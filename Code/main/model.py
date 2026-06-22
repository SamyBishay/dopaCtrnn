"""The two-area model — batched forward pass.

All tensors carry a batch dimension B (number of parallel environments).
Single-env rollouts use B=1 via reset_state(1).

Goal-directed (mPFC/DMS): CTRNN, split into a fast (widen) half at the
action/decision timescale and a slow (deepen) half at the maintenance
timescale. Cortical DA controls strength and temporal stability of decision
codes (Kutter et al.). Dopamine sets a MULTIPLICATIVE EXPRESSION GAIN with
two experimental placement modes (cfg.da_gain_mode):
  "output" (default): W_eff = f(DA) * W on the output readout r_out.
  "recurrent": gain applied inside dh — f(DA) multiplies the recurrent
    contribution directly, modelling Naudé et al. (2024) NMDA excitability.
DA can additionally modulate effective integration tau (cfg.da_tau=True):
  fast/widen units shorten with DA (faster decision dynamics);
  slow/deepen units lengthen with DA (stronger maintenance — Naudé deepen).
Emits action logits, a scalar DA-request, and value.

Habitual (DLS): CTRNN, opponent Go/NoGo readouts (policy = Go - NoGo).

Policy mixing: w_GD = sigmoid(alpha * DA + bias).
combined = w_GD * mot * pi_GD + (1 - w_GD) * pi_H
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tau(param):
    return 1.0 + F.softplus(param)


def _inv_softplus(y):
    return math.log(math.expm1(y))


def _mixed_tau_init(n, cfg):
    v = torch.empty(n)
    v[: n // 2] = _inv_softplus(cfg.tau_fast - 1.0)
    v[n // 2:] = _inv_softplus(cfg.tau_slow - 1.0)
    return v


class GDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n, a = cfg.n_gd, cfg.n_actions
        self.cfg = cfg
        self.W_in  = nn.Parameter(torch.randn(n, cfg.obs_dim) * 0.1)
        self.W     = nn.Parameter(torch.randn(n, n) * (0.9 / n ** 0.5))
        self.b     = nn.Parameter(torch.zeros(n))
        if getattr(cfg, "tau_mode", "mixed") == "uniform":
            _tau_init = torch.full((n,), _inv_softplus(getattr(cfg, "tau_uniform", 10.0) - 1.0))
        else:
            _tau_init = _mixed_tau_init(n, cfg)
        self.tau_p = nn.Parameter(_tau_init)
        self.W_out = nn.Parameter(torch.randn(a, n) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(a))
        self.w_da  = nn.Parameter(torch.randn(n) * 0.1)
        self.b_da  = nn.Parameter(torch.zeros(1))
        # Step 5 scalar split: an arbitration readout independent of da_request
        # (which drives the expression gain below). Only read when cfg.da_split
        # is True (DualSystemModel.step); unused parameters when False, so the
        # default forward pass is unaffected.
        self.w_arb = nn.Parameter(torch.randn(n) * 0.1)
        self.b_arb = nn.Parameter(torch.zeros(1))
        self.w_v   = nn.Parameter(torch.randn(n) * 0.1)
        self.b_v   = nn.Parameter(torch.zeros(1))
        self.gain_base = nn.Parameter(torch.tensor(cfg.gain_base))
        self.gain_da   = nn.Parameter(torch.tensor(cfg.gain_da))
        # n_fast = widen units (fast, action/decision timescale);
        # n_slow = deepen units (slow, maintenance timescale, tonic low-pass).
        self.n_fast = n // 2
        self.n_slow = n - self.n_fast
        fast_mask = torch.zeros(n); fast_mask[: self.n_fast] = 1.0
        self.register_buffer("fast_mask", fast_mask)

    def step(self, x, h, da_tonic, lesion=False):
        """x: [B, obs_dim], h: [B, n], da_tonic: [B]  →  all [B, ...]."""
        r  = torch.tanh(h)                                              # [B, n]

        da_gain_mode = getattr(self.cfg, "da_gain_mode", "output")
        da_tau_on    = getattr(self.cfg, "da_tau", False)

        # When either recurrent-gain or tau-modulation is active we need DA
        # signals computed from the PRE-UPDATE h (so that the step-t DA
        # influences step-t dynamics without a one-step lag).
        if da_gain_mode == "recurrent" or da_tau_on:
            da_req_pre = torch.sigmoid((h * self.w_da).sum(-1) + self.b_da.squeeze())  # [B]
            da_ton_pre = (1 - self.cfg.tonic_kappa) * da_tonic + self.cfg.tonic_kappa * da_req_pre
            # Per-unit DA component for recurrent/tau paths (same split as the gain path below)
            da_comp_pre = (self.fast_mask * da_req_pre.unsqueeze(-1)
                           + (1 - self.fast_mask) * da_ton_pre.unsqueeze(-1))         # [B, n]

        if da_gain_mode == "recurrent":
            # W_eff = f(DA) * W  —  gain is INSIDE the recurrence (Naudé NMDA excitability).
            rec_gain = self.gain_base + self.gain_da * da_comp_pre                    # [B, n]
            dh = -h + rec_gain * (r @ self.W.T) + x @ self.W_in.T + self.b           # [B, n]
        else:
            dh = -h + r @ self.W.T + x @ self.W_in.T + self.b                        # [B, n]

        if da_tau_on:
            tau_mode = getattr(self.cfg, "tau_mode", "mixed")
            da_tau_gain = getattr(self.cfg, "da_tau_gain", 0.5)
            if tau_mode == "uniform":
                # All units shorten with DA (widen only — no deepen group when tau is uniform).
                da_tau_comp = da_req_pre.unsqueeze(-1).expand(-1, h.shape[-1])        # [B, n]
                sign = torch.ones_like(self.fast_mask)                                # all +1
            else:
                # Mixed: fast units (sign=+1) shorten; slow units (sign=-1) lengthen.
                da_tau_comp = da_comp_pre
                sign = 2.0 * self.fast_mask - 1.0                                    # [n]
            # Log-space shift keeps eff_tau > 0 and makes da_tau_gain a fractional change.
            log_tau = torch.log(_tau(self.tau_p)) - da_tau_gain * sign * da_tau_comp  # [B, n]
            eff_tau = log_tau.exp().clamp_min(1.0)
            h = h + (self.cfg.dt / eff_tau) * dh
        else:
            h = h + (self.cfg.dt / _tau(self.tau_p)) * dh                            # original

        if lesion:
            h = torch.zeros_like(h)

        da_request = torch.sigmoid((h * self.w_da).sum(-1) + self.b_da.squeeze())    # [B]
        da_tonic   = (1 - self.cfg.tonic_kappa) * da_tonic + self.cfg.tonic_kappa * da_request
        # Fast (widen) units track phasic da_request; slow (deepen) units track tonic.
        da_comp    = self.fast_mask * da_request.unsqueeze(-1) + (1 - self.fast_mask) * da_tonic.unsqueeze(-1)  # [B, n]

        if da_gain_mode == "recurrent":
            # Gain already applied inside dh; output uses unscaled tanh.
            r_out = torch.tanh(h)
        else:
            # E3 (da_components="gain_only"): suppress DA expression gain; use base gain only.
            if getattr(self.cfg, "da_components", "both") == "gain_only":
                gain = self.gain_base
            else:
                gain = self.gain_base + self.gain_da * da_comp
            r_out = gain * torch.tanh(h)                                              # [B, n]

        pi    = r_out @ self.W_out.T + self.b_out                                    # [B, n_actions]
        value = (h * self.w_v).sum(-1) + self.b_v.squeeze()                          # [B]
        return pi, da_request, value, h, da_tonic

    def arbitration(self, h):
        """Independent arbitration readout (Step 5 split): same functional
        form as da_request but its own parameters, computed from the GD
        hidden state. Lets cfg.da_split=True decouple "which system drives
        behaviour" (-> w_gd) from "how strongly the GD policy is expressed"
        (-> da_request -> W_eff gain), as required for E2/E3."""
        return torch.sigmoid((h * self.w_arb).sum(-1) + self.b_arb.squeeze())


class HabNet(nn.Module):
    """Habitual CTRNN, opponent Go/NoGo readouts (policy = Go - NoGo).

    Recurrent weight can be full-rank (default) or low-rank. When
    cfg.hab_rank > 0 the recurrent matrix is parameterised as
    W = (m @ n.T) / n_hab  with m, n of shape [n_hab, rank]. This makes
    the "habit is low-dimensional" claim structural rather than something
    to be recovered post-hoc, and shrinks the parameter count of the
    system we most expect to be simple. Set cfg.hab_rank = 0 (default) to
    recover the original full-rank behaviour exactly.
    """
    def __init__(self, cfg):
        super().__init__()
        n, a = cfg.n_hab, cfg.n_actions
        self.cfg   = cfg
        self.rank  = int(getattr(cfg, "hab_rank", 0))
        self.W_in  = nn.Parameter(torch.randn(n, cfg.obs_dim_hab) * 0.1)
        if self.rank > 0:
            # Low-rank: W_rec = (m @ n_lr.T) / n.  m,n init small (cf. supervisor).
            self.m_lr = nn.Parameter(torch.randn(n, self.rank) * 0.1)
            self.n_lr = nn.Parameter(torch.randn(n, self.rank) * 0.1)
        else:
            self.W = nn.Parameter(torch.randn(n, n) * (0.9 / n ** 0.5))
        self.b     = nn.Parameter(torch.zeros(n))
        self.tau_p = nn.Parameter(_mixed_tau_init(n, cfg))
        self.W_go  = nn.Parameter(torch.randn(a, n) * 0.1)
        self.W_nogo= nn.Parameter(torch.randn(a, n) * 0.1)

    def _rec(self, r):
        """Recurrent contribution r @ W.T for either parameterisation."""
        if self.rank > 0:
            # r @ W.T = r @ (m n^T / N).T = ((r @ m) @ n.T) / N
            return (r @ self.m_lr) @ self.n_lr.T / self.cfg.n_hab
        return r @ self.W.T

    def rec_weight(self):
        """Materialise the recurrent matrix (for analysis / fixed points)."""
        if self.rank > 0:
            return (self.m_lr @ self.n_lr.T) / self.cfg.n_hab
        return self.W

    def step(self, x, h, lesion=False):
        """x: [B, obs_dim], h: [B, n]  →  pi [B, n_actions], h [B, n]."""
        r  = torch.tanh(h)
        dh = -h + self._rec(r) + x @ self.W_in.T + self.b
        h  = h + (self.cfg.dt / _tau(self.tau_p)) * dh
        if lesion:
            h = torch.zeros_like(h)
        r_out = torch.tanh(h)
        pi = r_out @ self.W_go.T - r_out @ self.W_nogo.T          # [B, n_actions]
        return pi, h


class DualSystemModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg  = cfg
        self.gd   = GDNet(cfg)
        self.hab  = HabNet(cfg)
        self.alpha = nn.Parameter(torch.tensor(cfg.wgd_alpha))
        self.bias  = nn.Parameter(torch.tensor(cfg.wgd_bias))
        # E2 (gate_mode="scheduled"): train.py fills this each iteration with a
        # fixed ramp value (never read from da_request). A buffer, not a plain
        # attribute, so it rides along in state_dict()/load_state_dict() — each
        # checkpoint (ckpt_learn/ckpt_maint) automatically freezes whatever
        # schedule value was active when that checkpoint was captured, with no
        # special-casing needed at any of analysis.py's many reload sites.
        # NaN = unset (gate_mode != "scheduled", or before the first ramp write).
        self.register_buffer("scheduled_w", torch.tensor(float("nan")))
        self.reset_state()

    def reset_state(self, batch_size=1):
        dev = self.alpha.device
        self.h_gd    = torch.zeros(batch_size, self.cfg.n_gd,  device=dev)
        self.h_hab   = torch.zeros(batch_size, self.cfg.n_hab, device=dev)
        self.da_tonic= torch.zeros(batch_size,                 device=dev)

    def gd_params(self):
        return list(self.gd.parameters()) + [self.alpha, self.bias]

    def hab_params(self):
        return list(self.hab.parameters())

    def step(self, obs, obs2, force_w=None, lesion=None, mot=1.0):
        """obs/obs2: [B, obs_dim].  All outputs carry batch dimension B."""
        les_gd  = lesion in ("gd",  "both")
        les_hab = lesion in ("hab", "both")
        pi_gd, da_request, value, self.h_gd, self.da_tonic = self.gd.step(
            obs, self.h_gd, self.da_tonic, lesion=les_gd)
        pi_h, self.h_hab = self.hab.step(obs2, self.h_hab, lesion=les_hab)

        # Scalar split: expression-gain path (→ W_eff, already applied inside
        # gd.step via da_request) vs the arbitration path (→ w_gd, below).
        # da_split=False (default, tied): both read da_request — bit-identical
        # to pre-split. da_split=True: arbitration is read from gd.arbitration(),
        # an independently-parameterised readout (model.py GDNet.w_arb/b_arb) of
        # the same hidden state — a genuinely separate signal, not a relabelling,
        # so E2 ("is expression-gating necessary") and E3 (Naudé decomposition)
        # can actually distinguish the two mechanisms.
        if self.cfg.da_split:
            da_expression  = da_request
            da_arbitration = self.gd.arbitration(self.h_gd)
        else:
            da_expression  = da_request
            da_arbitration = da_request

        # E3 (da_components): control which DA path is active.
        # "weights_only" or any mode that suppresses the gate: w_gd → 0 (fully habitual).
        # "gain_only": gate is DA-driven; expression gain is suppressed inside GDNet.
        # "both" (default): both paths active.
        da_comp_mode = getattr(self.cfg, "da_components", "both")
        if da_comp_mode == "weights_only":
            # DA drives expression gain only; arbitration gate is zeroed out.
            w_gd = torch.zeros_like(da_arbitration)
        elif force_w is not None:
            # Caller override (e.g. evaluation with force_w=0.0 or 1.0).
            w_gd = torch.full_like(da_arbitration, float(force_w))
        else:
            # E2 (gate_mode): expression gate (default) vs scheduled ramp.
            gate_mode = getattr(self.cfg, "gate_mode", "expression")
            if gate_mode == "scheduled" and not torch.isnan(self.scheduled_w):
                w_gd = torch.full_like(da_arbitration, float(self.scheduled_w))
            else:
                w_gd = torch.sigmoid(self.alpha * da_arbitration + self.bias)  # [B]

        w = w_gd.unsqueeze(-1)                                          # [B, 1]
        combined = w * mot * pi_gd + (1.0 - w) * pi_h                  # [B, n_actions]
        return {"combined": combined, "pi_gd": pi_gd, "pi_h": pi_h,
                "da_request": da_request, "value": value, "w_gd": w_gd,
                "da_expression": da_expression, "da_arbitration": da_arbitration}
