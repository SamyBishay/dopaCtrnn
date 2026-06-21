"""The two-area model — batched forward pass.

All tensors carry a batch dimension B (number of parallel environments).
Single-env rollouts use B=1 via reset_state(1).

Goal-directed (mPFC/DMS): CTRNN, split into a fast (widen) half at the
action/decision timescale and a slow (deepen) half at the maintenance
timescale. Cortical DA controls strength and temporal stability of decision
codes (Kutter et al.). Dopamine sets a MULTIPLICATIVE EXPRESSION GAIN:
W_eff = f(DA) * W. Emits action logits, a scalar DA-request, and value.

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
        self.tau_p = nn.Parameter(_mixed_tau_init(n, cfg))
        self.W_out = nn.Parameter(torch.randn(a, n) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(a))
        self.w_da  = nn.Parameter(torch.randn(n) * 0.1)
        self.b_da  = nn.Parameter(torch.zeros(1))
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
        r  = torch.tanh(h)                                          # [B, n]
        dh = -h + r @ self.W.T + x @ self.W_in.T + self.b         # [B, n]
        h  = h + (self.cfg.dt / _tau(self.tau_p)) * dh
        if lesion:
            h = torch.zeros_like(h)
        da_request = torch.sigmoid((h * self.w_da).sum(-1) + self.b_da.squeeze())  # [B]
        da_tonic   = (1 - self.cfg.tonic_kappa) * da_tonic + self.cfg.tonic_kappa * da_request
        # Fast (widen) units track the phasic da_request; slow (deepen) units
        # track the tonic low-pass — the timescale split, not a receptor label.
        da_comp    = self.fast_mask * da_request.unsqueeze(-1) + (1 - self.fast_mask) * da_tonic.unsqueeze(-1)  # [B, n]
        gain  = self.gain_base + self.gain_da * da_comp
        r_out = gain * torch.tanh(h)                               # [B, n]
        pi    = r_out @ self.W_out.T + self.b_out                  # [B, n_actions]
        value = (h * self.w_v).sum(-1) + self.b_v.squeeze()        # [B]
        return pi, da_request, value, h, da_tonic


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

        # Scalar split: expression-gain path (→ W_eff, inside gd.step) vs the
        # arbitration path (→ w_gd, below). In the tied default both equal
        # da_request, so the forward pass is bit-identical to pre-split. When
        # cfg.da_split is True the two are computed independently — Step 7 wires
        # them to separate sub-networks; here it is the structural scaffold only.
        if self.cfg.da_split:
            da_expression  = da_request   # stub: Step 7 will wire separate sub-networks
            da_arbitration = da_request
        else:
            da_expression  = da_request
            da_arbitration = da_request
        # NOTE: da_expression feeds W_eff. gd.step already used da_request for the
        # gain path; in the tied default this is identical, so no rewire needed.
        # Step 7 will thread da_expression into gd.step explicitly.

        if force_w is None:
            w_gd = torch.sigmoid(self.alpha * da_arbitration + self.bias)   # [B]
        else:
            w_gd = torch.full_like(da_arbitration, float(force_w))

        w = w_gd.unsqueeze(-1)                                          # [B, 1]
        combined = w * mot * pi_gd + (1.0 - w) * pi_h                  # [B, n_actions]
        return {"combined": combined, "pi_gd": pi_gd, "pi_h": pi_h,
                "da_request": da_request, "value": value, "w_gd": w_gd,
                "da_expression": da_expression, "da_arbitration": da_arbitration}
