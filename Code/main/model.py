"""The two-area model.

Goal-directed (mPFC/DMS): CTRNN, allocentric input, split into a D1/phasic-sensitive
and a D2/tonic-sensitive half. Dopamine sets a MULTIPLICATIVE EXPRESSION GAIN on the
units (W_eff = f(DA) * W): intact learned weights, scaled expression. It emits action
logits, a scalar DA-request, and a state value (critic).

Habitual (DLS): CTRNN, egocentric input, opponent Go/NoGo readouts, policy = Go - NoGo.

Policy mixing: w_GD = sigmoid(alpha * DA + bias); combined = w_GD*mot*pi_GD + (1-w_GD)*pi_H.
mot is a motivational/value multiplier on the goal-directed contribution (=1 normally;
devaluation sets it ~0, which is why only goal-directed behaviour is value-sensitive).
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
    """First half fast (action), second half slow (working memory).
    For the GD net this aligns with D1=phasic/fast, D2=tonic/slow."""
    v = torch.empty(n)
    v[: n // 2] = _inv_softplus(cfg.tau_fast - 1.0)
    v[n // 2:] = _inv_softplus(cfg.tau_slow - 1.0)
    return v


class GDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n, a = cfg.n_gd, cfg.n_actions
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.randn(n, cfg.obs_dim) * 0.1)
        self.W = nn.Parameter(torch.randn(n, n) * (0.9 / n ** 0.5))
        self.b = nn.Parameter(torch.zeros(n))
        self.tau_p = nn.Parameter(_mixed_tau_init(n, cfg))
        self.W_out = nn.Parameter(torch.randn(a, n) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(a))
        self.w_da = nn.Parameter(torch.randn(n) * 0.1)
        self.b_da = nn.Parameter(torch.zeros(1))
        self.w_v = nn.Parameter(torch.randn(n) * 0.1)
        self.b_v = nn.Parameter(torch.zeros(1))
        self.gain_base = nn.Parameter(torch.tensor(cfg.gain_base))
        self.gain_da = nn.Parameter(torch.tensor(cfg.gain_da))
        # D1 (phasic) = first half, D2 (tonic) = second half
        mask = torch.zeros(n)
        mask[: n // 2] = 1.0
        self.register_buffer("d1_mask", mask)

    def step(self, x, h, da_tonic, lesion=False):
        r = torch.tanh(h)
        dh = -h + self.W @ r + self.W_in @ x + self.b
        h = h + (self.cfg.dt / _tau(self.tau_p)) * dh
        if lesion:
            h = torch.zeros_like(h)
        da_request = torch.sigmoid((self.w_da * h).sum() + self.b_da).squeeze()
        da_tonic = (1 - self.cfg.tonic_kappa) * da_tonic + self.cfg.tonic_kappa * da_request
        da_comp = self.d1_mask * da_request + (1 - self.d1_mask) * da_tonic   # per-unit DA
        gain = self.gain_base + self.gain_da * da_comp                        # W_eff = gain * W
        r_out = gain * torch.tanh(h)
        pi = self.W_out @ r_out + self.b_out
        value = (self.w_v * h).sum() + self.b_v
        return pi, da_request, value.squeeze(), h, da_tonic


class HabNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n, a = cfg.n_hab, cfg.n_actions
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.randn(n, cfg.obs_dim) * 0.1)
        self.W = nn.Parameter(torch.randn(n, n) * (0.9 / n ** 0.5))
        self.b = nn.Parameter(torch.zeros(n))
        self.tau_p = nn.Parameter(_mixed_tau_init(n, cfg))
        self.W_go = nn.Parameter(torch.randn(a, n) * 0.1)
        self.W_nogo = nn.Parameter(torch.randn(a, n) * 0.1)

    def step(self, x, h, lesion=False):
        r = torch.tanh(h)
        dh = -h + self.W @ r + self.W_in @ x + self.b
        h = h + (self.cfg.dt / _tau(self.tau_p)) * dh
        if lesion:
            h = torch.zeros_like(h)
        r_out = torch.tanh(h)
        pi = self.W_go @ r_out - self.W_nogo @ r_out   # opponent: Go - NoGo
        return pi, h


class DualSystemModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gd = GDNet(cfg)
        self.hab = HabNet(cfg)
        self.alpha = nn.Parameter(torch.tensor(cfg.wgd_alpha))
        self.bias = nn.Parameter(torch.tensor(cfg.wgd_bias))
        self.reset_state()

    def reset_state(self):
        dev = self.alpha.device
        self.h_gd = torch.zeros(self.cfg.n_gd, device=dev)
        self.h_hab = torch.zeros(self.cfg.n_hab, device=dev)
        self.da_tonic = torch.zeros((), device=dev)

    def gd_params(self):
        return list(self.gd.parameters()) + [self.alpha, self.bias]

    def hab_params(self):
        return list(self.hab.parameters())

    def step(self, allo, ego, force_w=None, lesion=None, mot=1.0):
        """lesion in {None,'gd','hab','both'}; force_w overrides the DA gate (for solo evals)."""
        les_gd = lesion in ("gd", "both")
        les_hab = lesion in ("hab", "both")
        pi_gd, da_request, value, self.h_gd, self.da_tonic = self.gd.step(
            allo, self.h_gd, self.da_tonic, lesion=les_gd)
        pi_h, self.h_hab = self.hab.step(ego, self.h_hab, lesion=les_hab)
        w_gd = torch.sigmoid(self.alpha * da_request + self.bias) if force_w is None \
            else torch.as_tensor(float(force_w), device=da_request.device)
        combined = w_gd * mot * pi_gd + (1.0 - w_gd) * pi_h
        return {"combined": combined, "pi_gd": pi_gd, "pi_h": pi_h,
                "da_request": da_request, "value": value, "w_gd": w_gd}
