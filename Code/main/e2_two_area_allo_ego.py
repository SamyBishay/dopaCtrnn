"""Experiment 2 — two-area CTRNN with ego/allo observation split, no DA.

E1 + one change: the habitual area receives the position-free 4-dim obs
instead of the full 6-dim allocentric obs used by both areas in E1.

Architecture:
  GD  area: supervisor's FullRankCTRNN (512, alpha=0.2) + 2-layer actor
            + linear critic (E0 SingleAreaNet, unchanged from E1).
            Input: allocentric obs [x, y, 0, sig_L, sig_R, sig_choice] (6-dim).

  Hab area: rank-2 low-rank CTRNN, Go/NoGo readouts (our HabNet from model.py).
            Input: egocentric/position-free obs_hab [0, sig_L, sig_R, sig_choice] (4-dim).

Learning:
  GD  — A2C on task reward (same as E0/E1).
  Hab — APE: CE against GD-sampled actions + intrinsic efficiency RL (same as E1).

The split encodes the biological constraint that the habitual (DLS/dorsostriatal)
system has no access to allocentric position, only to the current task phase signal
(Packard & McGaugh, 1996). This is the same split used in the full DA model.

Ladder position:
  E0 — single-area CTRNN
  E1 — two-area, same 6D obs for both, APE
  E2 — two-area, ego/allo split (GD=6D, hab=4D), APE  (this file)
  Main model — E2 + DA expression gain + gating weight

Run:
    python e2_two_area_allo_ego.py --seed 0 --out results/e2/seed0/
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(__file__))
from config import Config
from e0_single_area_a2c import SingleAreaNet
from environment import TMazeFreeNav, TMazeVecEnv
from model import HabNet
from train import RunningNorm, discounted_batch, gae_batch


# ── two-area model (ego/allo split, no DA) ────────────────────────────

class TwoAreaAlloEgo(nn.Module):
    """GD (FullRankCTRNN, E0) + Hab (rank-2 LowRankCTRNN, our HabNet).

    GD receives allocentric obs (6-dim); hab receives position-free obs (4-dim).
    No gating — GD drives behaviour; hab imitates via APE.
    """
    def __init__(self, cfg):
        super().__init__()
        self.gd  = SingleAreaNet(cfg.obs_dim, cfg.n_actions,
                                 hidden=cfg.n_gd, alpha=0.2, noise_std=0.05)
        self.hab = HabNet(cfg)    # HabNet reads cfg.obs_dim_hab = 4

    def step(self, obs, obs_hab):
        """obs: [B, 6], obs_hab: [B, 4].  Returns dict with pi_gd, pi_h, value."""
        pi_gd, value, self.h_gd = self.gd.step(obs, self.h_gd)
        pi_h, self.h_hab         = self.hab.step(obs_hab, self.h_hab)
        return {"pi_gd": pi_gd, "pi_h": pi_h, "value": value}

    def reset_state(self, batch_size=1):
        dev = next(self.parameters()).device
        self.h_gd  = self.gd.init_hidden(batch_size, dev)
        self.h_hab = torch.zeros(batch_size, self.hab.cfg.n_hab, device=dev)

    def gd_params(self):
        return list(self.gd.parameters())

    def hab_params(self):
        return list(self.hab.parameters())


# ── evaluation ────────────────────────────────────────────────────────

def _eval(model, cfg, n_trials, fixed_delay=-1, force="gd"):
    """Greedy eval. force='gd' uses GD logits; force='hab' uses hab logits."""
    rng = np.random.default_rng(42)
    env = TMazeFreeNav(cfg, rng)
    saved = env.current_delay
    if fixed_delay >= 0:
        env.current_delay = fixed_delay

    model.eval()
    correct = 0
    dev = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(n_trials):
            model.reset_state(1)
            env.reset()
            done = False
            info = {"correct": False}
            while not done:
                obs     = torch.as_tensor(env.obs(),     dtype=torch.float32,
                                          device=dev).unsqueeze(0)
                obs_hab = torch.as_tensor(env.obs_hab(), dtype=torch.float32,
                                          device=dev).unsqueeze(0)
                out    = model.step(obs, obs_hab)
                logits = out["pi_gd"] if force == "gd" else out["pi_h"]
                a = int(logits[0].argmax())
                _, _, done, info = env.step(a)
            correct += int(info["correct"] or False)
    model.train()
    env.current_delay = saved
    return correct / n_trials


# ── training ──────────────────────────────────────────────────────────

def _train_batch(model, venv, cfg, opt_gd, opt_hab, ret_norm, ape_scale=1.0):
    """One vectorised A2C + APE iteration over B parallel episodes."""
    B   = venv.B
    dev = next(model.parameters()).device
    model.reset_state(B)
    venv.reset()
    model.train()

    all_pi_gd, all_pi_h, all_vals, all_acts = [], [], [], []
    all_task_r, all_int_r, all_valid = [], [], []
    active = np.ones(B, dtype=bool)

    for _ in range(cfg.max_episode_steps):
        obs     = torch.as_tensor(venv.obs(),     dtype=torch.float32, device=dev)
        obs_hab = torch.as_tensor(venv.obs_hab(), dtype=torch.float32, device=dev)
        out     = model.step(obs, obs_hab)

        acts = Categorical(logits=out["pi_gd"]).sample()

        valid_np = active.copy()
        task_r, int_r, done, _ = venv.step(acts.cpu().numpy())

        all_pi_gd.append(out["pi_gd"])
        all_pi_h.append(out["pi_h"])
        all_vals.append(out["value"])
        all_acts.append(acts)
        all_task_r.append(torch.from_numpy(task_r.astype(np.float32)).to(dev))
        all_int_r.append(torch.from_numpy(int_r.astype(np.float32)).to(dev))
        all_valid.append(torch.from_numpy(valid_np))
        active &= ~done
        if not active.any():
            break

    pi_gd_t = torch.stack(all_pi_gd)
    pi_h_t  = torch.stack(all_pi_h)
    vals_t  = torch.stack(all_vals)
    acts_t  = torch.stack(all_acts)
    task_t  = torch.stack(all_task_r)
    int_t   = torch.stack(all_int_r)
    valid_t = torch.stack(all_valid)

    valid_f     = valid_t.float()
    total_valid = valid_f.sum().clamp_min(1.0)

    # ── GD: A2C on task reward ─────────────────────────────────────────
    adv, returns = gae_batch(task_t, vals_t.detach(),
                             cfg.gamma, cfg.gae_lambda, valid_t)

    ret_vals = returns[valid_t].detach().cpu().numpy().tolist()
    ret_norm.update(ret_vals)
    r_mean, r_std = ret_norm.stats()
    returns_n = (returns - r_mean) / r_std

    adv_masked = adv[valid_t].detach()
    a_mean = adv_masked.mean() if adv_masked.numel() else adv.new_zeros(())
    a_std  = adv_masked.std()  if adv_masked.numel() > 1 else adv.new_ones(())
    adv_n  = (adv - a_mean) / (a_std + 1e-8)

    dist_gd  = Categorical(logits=pi_gd_t)
    logps_gd = dist_gd.log_prob(acts_t)
    ents_gd  = dist_gd.entropy()

    policy_l  = -((adv_n.detach() * logps_gd) * valid_f).sum()
    critic_l  = cfg.value_coef * (((returns_n - vals_t) ** 2) * valid_f).sum()
    entropy_l = -cfg.entropy_beta * (ents_gd * valid_f).sum()
    loss_gd   = (policy_l + critic_l + entropy_l) / total_valid

    opt_gd.zero_grad()
    loss_gd.backward()
    torch.nn.utils.clip_grad_norm_(model.gd_params(), cfg.grad_clip)
    opt_gd.step()

    # ── Hab: APE (CE) + intrinsic efficiency RL ───────────────────────
    A   = pi_h_t.shape[-1]
    ce  = F.cross_entropy(pi_h_t.reshape(-1, A), acts_t.reshape(-1),
                          reduction="none").reshape(acts_t.shape)
    ce_l = (ce * valid_f).sum() / total_valid

    returns_int = discounted_batch(int_t, cfg.gamma, valid_t)
    denom    = valid_f.sum(0).clamp_min(1.0)
    base_int = (returns_int * valid_f).sum(0) / denom
    adv_int  = returns_int - base_int.unsqueeze(0)
    logp_h   = Categorical(logits=pi_h_t).log_prob(acts_t)
    eff_l    = -((adv_int.detach() * logp_h) * valid_f).sum() / total_valid

    loss_hab = cfg.ape_weight * ape_scale * ce_l + cfg.eff_weight * eff_l

    opt_hab.zero_grad()
    loss_hab.backward()
    torch.nn.utils.clip_grad_norm_(model.hab_params(), cfg.grad_clip)
    opt_hab.step()

    return float(loss_gd.detach()), float(loss_hab.detach())


def train_e2(cfg, out_dir, verbose=True):
    """Train E2 and write results to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    B    = cfg.batch_size
    venv = TMazeVecEnv(cfg, np.random.default_rng(rng.integers(1 << 32)),
                       batch_size=B)

    model    = TwoAreaAlloEgo(cfg).to(cfg.device)
    opt_gd   = torch.optim.Adam(model.gd_params(),  lr=cfg.lr_gd)
    opt_hab  = torch.optim.Adam(model.hab_params(), lr=cfg.lr_hab)
    ret_norm = RunningNorm(cfg.ret_norm_window)

    logs = {
        "episode": [], "gd_acc": [], "hab_solo_acc": [], "delay": [],
        "fixed_delay_acc": [],
    }
    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())
    delay_advance_count = 0
    total_episodes = 0
    hab_solo_acc = 0.0
    gd_acc       = 0.0
    iterations   = (cfg.episodes + B - 1) // B

    for it in range(iterations):
        total_episodes += B

        if cfg.ape_decay:
            surplus = hab_solo_acc - gd_acc - cfg.ape_decay_margin
            ape_scale = (max(cfg.ape_min_scale, 1.0 - cfg.ape_decay_rate * surplus)
                         if surplus > 0 else 1.0)
        else:
            ape_scale = 1.0

        _train_batch(model, venv, cfg, opt_gd, opt_hab, ret_norm,
                     ape_scale=ape_scale)

        if total_episodes % cfg.eval_every < B or it == iterations - 1:
            gd_acc       = _eval(model, cfg, cfg.eval_trials, force="gd")
            hab_solo_acc = _eval(model, cfg, cfg.eval_trials, force="hab")
            at_ceiling   = venv.current_delay >= cfg.delay_max
            fd_acc = None
            if at_ceiling:
                fd_acc = _eval(model, cfg, cfg.eval_trials,
                               fixed_delay=cfg.fixed_eval_delay, force="gd")

            logs["episode"].append(total_episodes)
            logs["gd_acc"].append(gd_acc)
            logs["hab_solo_acc"].append(hab_solo_acc)
            logs["delay"].append(venv.current_delay)
            logs["fixed_delay_acc"].append(fd_acc)

            if gd_acc > best_acc:
                best_acc   = gd_acc
                best_state = copy.deepcopy(model.state_dict())

            if not at_ceiling:
                if gd_acc >= cfg.delay_advance_acc:
                    delay_advance_count += 1
                else:
                    delay_advance_count = 0
                if delay_advance_count >= cfg.delay_advance_evals:
                    venv.advance_delay()
                    delay_advance_count = 0
                    if verbose:
                        print(f"  -> delay → {venv.current_delay}", flush=True)

            if verbose:
                fd_str = f" fixedDA {fd_acc:.2f}" if fd_acc is not None else ""
                print(f"ep {total_episodes:6d} | GD {gd_acc:.2f} "
                      f"habSolo {hab_solo_acc:.2f} "
                      f"delay {venv.current_delay}{fd_str}", flush=True)

    final_gd  = _eval(model, cfg, cfg.final_trials, force="gd")
    final_hab = _eval(model, cfg, cfg.final_trials, force="hab")
    if verbose:
        print(f"\nfinal GD acc  ({cfg.final_trials} trials): {final_gd:.4f}", flush=True)
        print(f"final hab solo:                           {final_hab:.4f}", flush=True)

    results = {
        "experiment": "E2_two_area_allo_ego",
        "seed": cfg.seed,
        "final_gd_accuracy":  final_gd,
        "final_hab_accuracy": final_hab,
        "best_gd_accuracy":   best_acc,
        "logs": logs,
        "config": cfg.to_dict(),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    torch.save(best_state,         os.path.join(out_dir, "model_best.pt"))
    if verbose:
        print(f"results written to {out_dir}", flush=True)
    return model, logs, final_gd


# ── entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E2: two-area, ego/allo split, no DA")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--out",      default="results/e2/seed0")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    cfg = Config(seed=args.seed, device=args.device, hab_rank=2)
    if args.episodes is not None:
        cfg.episodes = args.episodes

    train_e2(cfg, out_dir=args.out, verbose=not args.quiet)
