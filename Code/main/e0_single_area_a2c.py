"""Experiment 0 — single-area CTRNN baseline trained with A2C.

One FullRankCTRNN (supervisor's PFC hyperparameters) + 2-layer actor +
linear critic. No habitual system, no DA gating, no expression gain.

This is the minimal baseline: can a single CTRNN solve TUNL at all, and
what does the training curve look like without the dual-system architecture?
Needed to interpret H1 (does the two-area model add anything over this) and
as a sanity check that the vectorised TUNL environment is correct.

Run:
    python e0_single_area_a2c.py --seed 0 --out results/e0/seed0/
    python e0_single_area_a2c.py --seed 1 --out results/e0/seed1/ --episodes 256000
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
from environment import TMazeFreeNav, TMazeVecEnv
from train import RunningNorm, gae_batch


# ── model ─────────────────────────────────────────────────────────────

class SingleAreaNet(nn.Module):
    """Supervisor's FullRankRNN with a 2-layer actor + linear critic.

    Dynamics (matches supervisor's FullRankRNN exactly):
        r   = tanh(h + b)
        h'  = h + noise + alpha * (-h + r @ W_rec.T + x @ W_in)
        r'  = tanh(h' + b)                   <- readout to actor/critic

    alpha = dt/tau = 0.2  (supervisor's ALPHA).
    noise_std = 0.05 during training, 0 during eval.
    """
    def __init__(self, obs_dim, n_actions, hidden=512,
                 alpha=0.2, noise_std=0.05):
        super().__init__()
        self.hidden    = hidden
        self.alpha     = alpha
        self.noise_std = noise_std

        # Recurrent weights — supervisor's init
        self.W_in  = nn.Parameter(torch.empty(obs_dim, hidden))
        self.W_rec = nn.Parameter(torch.empty(hidden, hidden))
        self.b     = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.W_in)
        nn.init.normal_(self.W_rec, std=1.0 / hidden ** 0.5)

        # Actor: 2-layer MLP on readout r (supervisor's pfc_actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, n_actions)
        )
        # Critic: linear on readout r (supervisor's pfc_critic)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def step(self, x, h):
        """One CTRNN step.

        x: [B, obs_dim], h: [B, hidden]
        returns logits [B, n_actions], value [B], h_new [B, hidden]
        """
        noise = torch.randn_like(h) * self.noise_std if self.training else 0.0
        r     = torch.tanh(h + self.b)
        h     = h + noise + self.alpha * (-h + r @ self.W_rec.T + x @ self.W_in)
        r_out = torch.tanh(h + self.b)
        logits = self.actor(r_out)
        value  = self.critic(r_out).squeeze(-1)
        return logits, value, h

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden, device=device)


# ── evaluation ────────────────────────────────────────────────────────

def _eval(model, cfg, n_trials, fixed_delay=-1):
    """Greedy eval, returns accuracy over n_trials."""
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
            h = model.init_hidden(1, dev)
            env.reset()
            done = False
            info = {"correct": False}
            while not done:
                x = torch.as_tensor(env.obs(), dtype=torch.float32,
                                    device=dev).unsqueeze(0)
                logits, _, h = model.step(x, h)
                a = int(logits[0].argmax())
                _, _, done, info = env.step(a)
            correct += int(info["correct"] or False)
    model.train()
    env.current_delay = saved
    return correct / n_trials


# ── training ──────────────────────────────────────────────────────────

def _train_batch(model, venv, cfg, optimizer, da_lambda_unused, ret_norm):
    """One vectorised A2C iteration over B parallel episodes."""
    B   = venv.B
    dev = next(model.parameters()).device
    h   = model.init_hidden(B, dev)
    venv.reset()
    model.train()

    all_logits, all_vals, all_acts = [], [], []
    all_task_r, all_valid = [], []
    active = np.ones(B, dtype=bool)

    for _ in range(cfg.max_episode_steps):
        x = torch.as_tensor(venv.obs(), dtype=torch.float32, device=dev)
        logits, value, h = model.step(x, h)
        h = h.detach()
        acts = Categorical(logits=logits).sample()

        valid_np = active.copy()
        task_r, _int_r, done, _ = venv.step(acts.cpu().numpy())

        all_logits.append(logits)
        all_vals.append(value)
        all_acts.append(acts)
        all_task_r.append(torch.from_numpy(task_r.astype(np.float32)).to(dev))
        all_valid.append(torch.from_numpy(valid_np))
        active &= ~done
        if not active.any():
            break

    T = len(all_logits)
    logits_t = torch.stack(all_logits)   # [T, B, n_actions]
    vals_t   = torch.stack(all_vals)     # [T, B]
    acts_t   = torch.stack(all_acts)     # [T, B]
    task_t   = torch.stack(all_task_r)   # [T, B]
    valid_t  = torch.stack(all_valid)    # [T, B] bool

    valid_f      = valid_t.float()
    total_valid  = valid_f.sum().clamp_min(1.0)

    adv, returns = gae_batch(task_t, vals_t.detach(),
                             cfg.gamma, cfg.gae_lambda, valid_t)

    # Return normalisation
    ret_vals = returns[valid_t].detach().cpu().numpy().tolist()
    ret_norm.update(ret_vals)
    r_mean, r_std = ret_norm.stats()
    returns_n = (returns - r_mean) / r_std

    # Advantage normalisation
    adv_masked = adv[valid_t].detach()
    a_mean = adv_masked.mean() if adv_masked.numel() else adv.new_zeros(())
    a_std  = adv_masked.std()  if adv_masked.numel() > 1 else adv.new_ones(())
    adv_n  = (adv - a_mean) / (a_std + 1e-8)

    dist   = Categorical(logits=logits_t)
    logps  = dist.log_prob(acts_t)        # [T, B]
    ents   = dist.entropy()               # [T, B]

    policy_l  = -((adv_n.detach() * logps) * valid_f).sum()
    critic_l  = cfg.value_coef * (((returns_n - vals_t) ** 2) * valid_f).sum()
    entropy_l = -cfg.entropy_beta * (ents * valid_f).sum()
    loss = (policy_l + critic_l + entropy_l) / total_valid

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return float(loss.detach())


def train_e0(cfg, out_dir, verbose=True):
    """Train the single-area baseline and write results to out_dir.

    Saves:
        results.json      — training log + final accuracy
        model_final.pt    — final model weights
        model_best.pt     — weights at best combined accuracy
    """
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    B    = cfg.batch_size
    dev  = cfg.device
    venv = TMazeVecEnv(cfg, np.random.default_rng(rng.integers(1 << 32)),
                       batch_size=B)

    model     = SingleAreaNet(cfg.obs_dim, cfg.n_actions,
                              hidden=cfg.n_gd, alpha=0.2, noise_std=0.05).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_gd)
    ret_norm  = RunningNorm(cfg.ret_norm_window)

    logs = {"episode": [], "combined_acc": [], "delay": [], "fixed_delay_acc": []}
    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())
    delay_advance_count = 0
    total_episodes = 0
    iterations = (cfg.episodes + B - 1) // B

    for it in range(iterations):
        total_episodes += B
        _train_batch(model, venv, cfg, optimizer, None, ret_norm)

        if total_episodes % cfg.eval_every < B or it == iterations - 1:
            acc = _eval(model, cfg, cfg.eval_trials)
            at_ceiling = venv.current_delay >= cfg.delay_max
            fd_acc = None
            if at_ceiling:
                fd_acc = _eval(model, cfg, cfg.eval_trials,
                               fixed_delay=cfg.fixed_eval_delay)

            logs["episode"].append(total_episodes)
            logs["combined_acc"].append(acc)
            logs["delay"].append(venv.current_delay)
            logs["fixed_delay_acc"].append(fd_acc)

            if acc > best_acc:
                best_acc   = acc
                best_state = copy.deepcopy(model.state_dict())

            if not at_ceiling:
                if acc >= cfg.delay_advance_acc:
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
                print(f"ep {total_episodes:6d} | acc {acc:.2f} "
                      f"delay {venv.current_delay}{fd_str}", flush=True)

    # Final high-n eval
    final_acc = _eval(model, cfg, cfg.final_trials)
    if verbose:
        print(f"\nfinal acc ({cfg.final_trials} trials): {final_acc:.4f}", flush=True)

    # Save
    results = {
        "experiment": "E0_single_area_a2c",
        "seed": cfg.seed,
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
        "logs": logs,
        "config": cfg.to_dict(),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    torch.save(best_state,         os.path.join(out_dir, "model_best.pt"))
    if verbose:
        print(f"results written to {out_dir}", flush=True)
    return model, logs, final_acc


# ── entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E0: single-area CTRNN A2C baseline")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--out",      default="results/e0/seed0",
                        help="output directory for results + weights")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override cfg.episodes")
    parser.add_argument("--hidden",   type=int, default=None,
                        help="override network size (default: cfg.n_gd = 512)")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    cfg = Config(seed=args.seed, device=args.device)
    if args.episodes is not None:
        cfg.episodes = args.episodes
    if args.hidden is not None:
        cfg.n_gd = args.hidden

    # E0 uses the same env config as the main training:
    # difficulty=2, len_edge=7, delay_start=15, delay_max=40.
    # No overrides needed — Config defaults match supervisor's params.

    train_e0(cfg, out_dir=args.out, verbose=not args.quiet)
