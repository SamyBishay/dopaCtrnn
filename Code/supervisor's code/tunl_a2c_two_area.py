"""
tunl_a2c_two_area.py
════════════════════
Two-area PFC + DLS model for the TUNL working-memory task.

Architecture
────────────
  obs → PFC (FullRankRNN, N=512) → π_pfc, V_pfc   goal-directed critic
  obs → DLS (LowRankRNN,  rank=r) → π_dls, V_dls   efficiency critic

Training (fully decoupled — two separate forward passes per episode)
────────────────────────────────────────────────────────────────────
  Phase 1 — Warmup (PFC only):
    PFC trains via standard A2C on full reward (correct +1, incorrect -1, step -0.02).
    Exits when: steps ≥ PFC_WARMUP_STEPS AND acc ≥ WARMUP_ACC_THRESH
                AND curriculum at max_delay.

  Phase 2 — DLS imitation (PFC still acts):
    DLS trains via CE imitation + efficiency RL:
      choice_ce : CE imitation of PFC greedy at choice step only (CHOICE_WEIGHT scaled)
      a2c_eff   : A2C on efficiency reward (step cost + wrong-side penalty, NO food)
    DLS never sees the food reward → outcome-insensitive habitual policy.

  Phase 3 — Handover (DLS acts, PFC frozen):
    DLS drives behavior. PFC parameters frozen — no more gradient updates.
    DLS continues receiving CE tutor signal from the now-fixed PFC policy.
    DLS can surpass frozen PFC because it optimises efficiency independently.
    Fallback: if dls_solo_acc < DLS_FALLBACK_THRESH, PFC unfreezes and resumes.

Key readouts
────────────
  dls_solo_acc  : accuracy when DLS acts alone (solo eval, ground truth for handover)
  dls_ce_agree  : CE agreement with PFC during training (diagnostic proxy only)
  gate_log      : 1=PFC acting, 0=DLS acting
  ep_lengths    : shorter after handover signals DLS efficiency gain
  value_gap     : |V_pfc − V_dls| (different scales by design; tracks convergence)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os, time, shutil, glob
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import argparse, importlib.util
from math import sqrt as _sqrt

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
def _load_config(path):
    spec = importlib.util.spec_from_file_location("params", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: getattr(mod, k) for k in dir(mod) if k.isupper()}

parser = argparse.ArgumentParser()
parser.add_argument("--config",      default="params.py")
parser.add_argument("--seed",        type=int, default=None)
parser.add_argument("--dls_rank",    type=int, default=None)
parser.add_argument("--difficulty",  type=int, default=None)
parser.add_argument("--start_delay", type=int, default=None)
parser.add_argument("--max_delay",   type=int, default=None)
parser.add_argument("--n_steps",     type=int, default=None)
parser.add_argument("--lesion_pfc",  action="store_true", default=None)
parser.add_argument("--lesion_dls",  action="store_true", default=None)
parser.add_argument("--run_mode",    default=None, choices=["single","pairs","sweep"])
parser.add_argument("--save_path",   default=None)
parser.add_argument("--resume",      default=None)
args = parser.parse_args()

cfg = _load_config(args.config)
_overrides = {"SEED": args.seed, "DLS_RANK": args.dls_rank,
              "DIFFICULTY": args.difficulty, "START_DELAY": args.start_delay,
              "MAX_DELAY": args.max_delay, "N_STEPS": args.n_steps,
              "LESION_PFC": args.lesion_pfc, "LESION_DLS": args.lesion_dls,
              "RUN_MODE": args.run_mode, "SAVE_PATH": args.save_path}
for k, v in _overrides.items():
    if v is not None: cfg[k] = v

SAVE_PATH         = cfg["SAVE_PATH"]
SEED              = cfg["SEED"]
DIFFICULTY        = cfg["DIFFICULTY"]
LEN_EDGE          = cfg["LEN_EDGE"]
START_DELAY       = cfg["START_DELAY"]
MAX_DELAY         = cfg["MAX_DELAY"]
N_STEPS           = cfg["N_STEPS"]
LESION_PFC        = cfg["LESION_PFC"]
LESION_DLS        = cfg["LESION_DLS"]
PFC_HIDDEN        = cfg["PFC_HIDDEN"]
DLS_HIDDEN        = cfg["DLS_HIDDEN"]
DLS_RANK          = cfg["DLS_RANK"]
ALPHA             = cfg["ALPHA"]
NOISE_STD         = cfg["NOISE_STD"]
DLS_FALLBACK_THRESH     = cfg["DLS_FALLBACK_THRESH"]
DLS_PENALTY_ERROR       = cfg["DLS_PENALTY_ERROR"]
DLS_COMPLETION_BONUS    = cfg["DLS_COMPLETION_BONUS"]
NAV_CE_WEIGHT           = cfg["NAV_CE_WEIGHT"]
CHOICE_WEIGHT           = cfg["CHOICE_WEIGHT"]
EFFICIENCY_WEIGHT       = cfg["EFFICIENCY_WEIGHT"]
DLS_HANDOVER_FRAC       = cfg["DLS_HANDOVER_FRAC"]
PFC_FREEZE_ACC          = cfg["PFC_FREEZE_ACC"]
PFC_FREEZE_REACH        = cfg["PFC_FREEZE_REACH"]
LR_PFC            = cfg["LR_PFC"]
LR_DLS            = cfg["LR_DLS"]
PFC_WARMUP_STEPS  = cfg["PFC_WARMUP_STEPS"]
WARMUP_ACC_THRESH = cfg["WARMUP_ACC_THRESH"]
CHECKPOINT_EVERY  = cfg["CHECKPOINT_EVERY"]
EVAL_EVERY        = cfg["EVAL_EVERY"]
RUN_MODE          = cfg["RUN_MODE"]
FIXED_PAIRS       = cfg["FIXED_PAIRS"]
RANK_SWEEP_LIST   = cfg["RANK_SWEEP_LIST"]
EVAL_DELAYS       = cfg["EVAL_DELAYS"]
RWD               = cfg["RWD"]
INC_RWD           = cfg["INC_RWD"]
STEP_RWD          = cfg["STEP_RWD"]
GAMMA             = cfg["GAMMA"]
GAE_LAMBDA        = cfg["GAE_LAMBDA"]
VALUE_COEFF       = cfg["VALUE_COEFF"]
GRAD_CLIP         = cfg["GRAD_CLIP"]
TARGET_ACC        = cfg["TARGET_ACC"]
MAX_EPISODE_STEPS = cfg["MAX_EPISODE_STEPS"]

N_ENVS     = 1
LOG_EVERY  = 100 * 20
PLOT_EVERY = 1000 * 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"PFC    : FullRankRNN  hidden={PFC_HIDDEN}")
print(f"DLS    : LowRankRNN   hidden={DLS_HIDDEN}  rank={DLS_RANK}")
print(f"Switch : dls_solo_ema >= {DLS_HANDOVER_FRAC:.0%} × pfc_solo  "
      f"(eval every {EVAL_EVERY:,} steps, fallback < {DLS_FALLBACK_THRESH:.0%} × pfc_solo)")
print(f"Choice weight: {CHOICE_WEIGHT}x")


# ─────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────
class VectorizedTunlVisual:
    def __init__(self, len_delay=0, len_edge=LEN_EDGE, rwd=RWD,
                 inc_rwd=INC_RWD, step_rwd=STEP_RWD,
                 rng_seed=SEED, difficulty=DIFFICULTY):
        assert len_edge % 2 == 1 and len_edge >= 5
        if   difficulty == 0: self.h = (len_edge - 1) // 2 + 1
        elif difficulty == 1: self.h = (len_edge + 1) // 2 + 1
        elif difficulty == 2: self.h = (len_edge + 1) // 2 + 2
        else: raise ValueError(f"Invalid difficulty: {difficulty}")
        print(f"Difficulty {difficulty} | grid h={self.h}")
        self.w = len_edge + 2; self.len_delay = len_delay
        self.rwd = rwd; self.inc_rwd = inc_rwd; self.step_rwd = step_rwd
        self.walls = np.ones((self.h, self.w), dtype=int)
        i = 1; self.walls[i, i:self.w - i] = 0; i += 1
        while i < self.h - 1:
            self.walls[i, (self.w - 1) // 2] = 0; i += 1
        if self.h > 6:#>=5:#
            self.walls[2, (self.w-1)//2-2:(self.w-1)//2+3] = 0
            self.walls[3, (self.w-1)//2-1:(self.w-1)//2+2] = 0
            
        self.default_walls = self.walls.copy()
        self.initiation_loc = (self.h - 2, (self.w - 1) // 2)
        self.left_loc  = (1, 1)
        self.right_loc = (1, self.w - 2)
        self.directions = [np.array([-1,0]), np.array([1,0]),
                           np.array([0,-1]), np.array([0,1]),
                           np.array([0,0])]
        self.color = {
            "agent":       np.array([  0,   0, 255], dtype=np.uint8),
            "initiation":  np.array([255,   0,   0], dtype=np.uint8),
            "touchscreen": np.array([  0, 255,   0], dtype=np.uint8),
            "wall":        np.array([255, 255, 255], dtype=np.uint8),
            "bg":          np.array([  0,   0,   0], dtype=np.uint8),
        }
        self.rng = np.random.RandomState(rng_seed)
        self.observation  = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.phase_signal = np.zeros(3, dtype=float)
        self.current_loc  = (0, 0); self.sample_loc = (0, 0)
        self.sample = "undefined"; self.phase = 1
        self.indelay = False; self.delay_t = 0; self.done = False
        self.correction_trial = False; self.prev_action = 4
        self.nav_reward = 0.0; self.episode_reward = 0.0
        self.episode_steps = 0; self.correctness = False
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.force_init_loc = False
        self.reset()

    def reset(self, force_init_loc=False):
        self.walls = self.default_walls.copy()
        open_y, open_x = np.where(self.walls == 0)
        idx = self.rng.randint(0, len(open_y), size=N_ENVS)
        self.current_loc = (open_y[idx], open_x[idx])
        if force_init_loc:
            iy, ix = self.initiation_loc
            self.current_loc = (np.array([iy]*N_ENVS), np.array([ix]*N_ENVS))
        if not self.correction_trial:
            self.sample_loc = (self.left_loc if self.rng.choice([0,1]) == 0
                               else self.right_loc)
        self.sample="undefined"; self.phase=1; self.indelay=False
        self.delay_t=0; self.done=False; self.prev_action=4
        self.nav_reward=0.0; self.episode_reward=0.0; self.episode_steps=0
        self.observation.fill(0)
        self.observation[self.walls == 1] = self.color["wall"]
        y, x = self.current_loc
        self.observation[y, x] += self.color["agent"]
        iy, ix = self.initiation_loc
        self.observation[iy, ix] += self.color["initiation"]
        self.phase_signal = np.array([0.0, 0.0, 0.0])
        self.correction_trial = False; self.correctness = False

    def warm_start(self):
        obs, _, _, _ = self.step(4)
        # print('within. warm start...', obs)
        return obs

    def step(self, action: int):
        reward = 0.0; info = {}
        if self.done:
            return self.get_vectorized_observation(), 0.0, True, info
        cy, cx = self.current_loc
        if not self.indelay:
            reward = self.step_rwd; self.nav_reward += self.step_rwd
        dy, dx = self.directions[action]
        ny, nx = cy + dy, cx + dx
        if 0 <= ny < self.h and 0 <= nx < self.w and self.walls[ny, nx] == 0:
            self.observation[cy, cx] -= self.color["agent"]
            self.current_loc = (ny, nx)
            self.observation[ny, nx] += self.color["agent"]
            cy, cx = ny, nx
        cell_sum = int(np.sum(self.observation[cy, cx]))
        at_lit   = cell_sum > 255
        at_init  = (cy, cx) == self.initiation_loc
        at_left  = (cy, cx) == self.left_loc
        at_right = (cy, cx) == self.right_loc
        if self.phase == 1 and at_init and at_lit:
            self.observation[self.initiation_loc] -= self.color["initiation"]
            if self.sample == "undefined":
                sy, sx = self.sample_loc
                self.observation[sy, sx] += self.color["touchscreen"]
                self.phase_signal = (np.array([1./(self.w-1), 0., 0.])
                                     if self.sample_loc == self.left_loc
                                     else np.array([0., 1./(self.w-1), 0.]))
                self.phase = 2
                ''' Block the opposite side of the sample, mimicing the experimental setup where the non-sample side is unlit and inaccessible during the delay. '''
                if self.sample_loc == self.left_loc:
                    self.walls[:, (self.w - 1) // 2+2:self.w-1] = 1
                elif self.sample_loc == self.right_loc:
                    self.walls[:, 1:(self.w - 1) // 2-1] = 1
                
            else:
                self.observation[self.left_loc]  += self.color["touchscreen"]
                self.observation[self.right_loc] += self.color["touchscreen"]
                self.phase_signal = np.array([0., 0., 1./(self.w-1)])
                self.phase = 4
                '''release both sides for choice, mimicking the experimental setup where both sides are lit and accessible during the choice phase.'''
                self.walls = self.default_walls.copy()
        elif self.phase == 2 and (cy, cx) == self.sample_loc and at_lit:
            self.sample = "L" if self.sample_loc == self.left_loc else "R"
            sy, sx = self.sample_loc
            self.observation[sy, sx] -= self.color["touchscreen"]
            self.indelay = True; self.phase = 3
        elif self.phase == 4 and (at_left or at_right) and at_lit:
            side    = "L" if at_left else "R"
            correct = (side != self.sample)
            reward  = self.rwd if correct else self.inc_rwd
            self.correction_trial = False; self.correctness = correct
            self.done = True
        if self.indelay:
            if self.delay_t < self.len_delay:
                self.delay_t += 1; self.phase_signal *= 0.10
            else:
                self.indelay = False; self.phase = 1
                self.observation[self.initiation_loc] += self.color["initiation"]
                self.phase_signal = np.array([0., 0., 1./(self.w-1)*0.1])
            reward = 0.0
        self.prev_action = action; self.episode_steps += 1
        self.episode_reward += reward
        if self.episode_steps >= self.max_episode_steps and not self.done:
            reward -= 10.0; self.episode_reward -= 10.0; self.done = True
        just_done = self.done
        if just_done:
            info = {'episode': {
                'r':       float(self.episode_reward),
                'l':       int(self.episode_steps),
                'correct': self.correctness and (self.sample != "undefined"),
                'timeout': self.episode_steps >= self.max_episode_steps,
                'choice':  (1 if self.phase==4 and at_left
                            else 2 if self.phase==4 and at_right else 0),
            }}
            self.reset(force_init_loc=self.force_init_loc)
        return self.get_vectorized_observation(), reward, just_done, info

    def get_vectorized_observation(self):
        obs = np.zeros(6, dtype=np.float32)
        y, x = self.current_loc

        # if these are 1-element arrays, convert to scalar
        if isinstance(x, np.ndarray): x = x.item()
        if isinstance(y, np.ndarray): y = y.item()

        obs[0] = x / (self.w - 1.0)
        obs[1] = y / (self.h - 1.0)
        obs[2] = 0.0   # prev_action zeroed — no action history in obs
        obs[3:6] = self.phase_signal
        return obs


# ─────────────────────────────────────────────────────────────────────
# RNN modules
# ─────────────────────────────────────────────────────────────────────
class FullRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, noise_std, alpha=0.2,
                 add_biases=False, non_linearity=torch.tanh):
        super().__init__()
        self.hidden_size = hidden_size; self.noise_std = noise_std
        self.alpha = alpha; self.non_linearity = non_linearity
        self.wi   = nn.Parameter(torch.empty(input_size, hidden_size))
        self.wrec = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b    = nn.Parameter(torch.zeros(hidden_size), requires_grad=add_biases)
        self.h0   = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        nn.init.normal_(self.wi)
        nn.init.normal_(self.wrec, std=1.0 / _sqrt(hidden_size))

    def forward(self, x, h):
        squeeze = (x.dim() == 2)
        if squeeze: x = x.unsqueeze(1)
        B, T, _ = x.shape
        noise_scale = self.noise_std if self.training else 0.0
        noise = torch.randn(B, T, self.hidden_size, device=x.device) * noise_scale
        inp   = x @ self.wi
        rs = []
        for i in range(T):
            h = (h + noise[:, i] + self.alpha * (
                -h + self.non_linearity(h + self.b) @ self.wrec.t() + inp[:, i]))
            rs.append(self.non_linearity(h + self.b))
        r_seq = torch.stack(rs, dim=1)
        if squeeze: return r_seq.squeeze(1), h
        return r_seq, h

    def init_hidden(self, batch_size, device):
        return self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(device)


class LowRankRNN(nn.Module):
    """W_rec = m @ n^T / N  — primary analysis target after training."""
    def __init__(self, input_size, hidden_size, noise_std, alpha=0.2, rank=2,
                 add_biases=False, non_linearity=torch.tanh):
        super().__init__()
        self.hidden_size = hidden_size; self.noise_std = noise_std
        self.alpha = alpha; self.rank = rank; self.non_linearity = non_linearity
        self.wi = nn.Parameter(torch.empty(input_size, hidden_size))
        self.si = nn.Parameter(torch.ones(input_size), requires_grad=False)
        self.m  = nn.Parameter(torch.empty(hidden_size, rank))
        self.n  = nn.Parameter(torch.empty(hidden_size, rank))
        self.b  = nn.Parameter(torch.zeros(hidden_size), requires_grad=add_biases)
        self.h0 = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        # nn.init.normal_(self.wi); nn.init.normal_(self.m); nn.init.normal_(self.n)
        # NEW:
        nn.init.xavier_normal_(self.wi, gain=1.0)
        nn.init.normal_(self.m, std=0.1)
        nn.init.normal_(self.n, std=0.1)

    def forward(self, x, h):
        squeeze = (x.dim() == 2)
        if squeeze: x = x.unsqueeze(1)
        B, T, _ = x.shape
        wi_full = (self.wi.t() * self.si).t()
        noise_scale = self.noise_std if self.training else 0.0
        noise   = torch.randn(B, T, self.hidden_size, device=x.device) * noise_scale
        inp     = x @ wi_full
        rs = []
        # scaling = np.sqrt(self.hidden_size)  # ~22.6 instead of 512
        for i in range(T):
            r = self.non_linearity(h + self.b)
            # h = (h + noise[:, i] + self.alpha * (
            #     -h + r @ self.n @ self.m.t() / self.hidden_size + inp[:, i]))
            # Use sqrt(N) scaling instead of N for better gradient flow
            h = (h + noise[:, i] + self.alpha * (
                -h + r @ self.n @ self.m.t()  + inp[:, i]))
            rs.append(self.non_linearity(h + self.b))
        r_seq = torch.stack(rs, dim=1)
        if squeeze: return r_seq.squeeze(1), h
        return r_seq, h

    def init_hidden(self, batch_size, device):
        return self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(device)

    def svd_reparametrize(self):
        with torch.no_grad():
            W = (self.m @ self.n.t()).cpu().numpy()
            u, s, vt = np.linalg.svd(W, full_matrices=False)
            u, s, vt = u[:, :self.rank], s[:self.rank], vt[:self.rank]
            self.m.set_(torch.from_numpy(u * np.sqrt(s)).to(self.m.device))
            self.n.set_(torch.from_numpy(vt.T * np.sqrt(s)).to(self.n.device))


# ─────────────────────────────────────────────────────────────────────
# Two-area model
# ─────────────────────────────────────────────────────────────────────
class TwoAreaACNet(nn.Module):
    """
    PFC : FullRankRNN → π_pfc, V_pfc   (outcome reward: +1/-1/step)
    DLS : LowRankRNN  → π_dls, V_dls   (efficiency reward: step cost only)

    PFC learns what to do (goal-directed).
    DLS learns to do it fast (habitual efficiency).
    DLS also imitates PFC at choice steps via CE so it knows which side.

    No soft mixing. Caller picks area via force_pfc / force_dls.
    """
    def __init__(self, input_dim=6, action_dim=5, dls_rank=None):
        super().__init__()
        _rank = dls_rank if dls_rank is not None else DLS_RANK
        self.input_dim  = input_dim
        self.action_dim = action_dim

        self.pfc_rnn  = FullRankRNN(input_dim, PFC_HIDDEN,
                                    noise_std=NOISE_STD, alpha=ALPHA)
        
        if _rank > 0:
            self.dls_rnn  = LowRankRNN(input_dim, DLS_HIDDEN,
                                    noise_std=NOISE_STD, alpha=ALPHA, rank=_rank)
        else:
            self.dls_rnn  = FullRankRNN(input_dim, DLS_HIDDEN,
                                    noise_std=NOISE_STD, alpha=ALPHA)
        self.pfc_actor  = nn.Sequential(nn.Linear(PFC_HIDDEN, 256), nn.ReLU(),
                                        nn.Linear(256, action_dim))
        self.pfc_critic = nn.Linear(PFC_HIDDEN, 1)
        self.dls_actor  = nn.Sequential(nn.Linear(DLS_HIDDEN, 256), nn.ReLU(),
                                        nn.Linear(256, action_dim))
        self.dls_critic = nn.Linear(DLS_HIDDEN, 1)   # predicts efficiency returns

        for critic in (self.pfc_critic, self.dls_critic):
            nn.init.orthogonal_(critic.weight, 1.0)
            nn.init.zeros_(critic.bias)

    def init_hidden(self, batch_size, device):
        return (self.pfc_rnn.init_hidden(batch_size, device),
                self.dls_rnn.init_hidden(batch_size, device))

    def forward(self, x, hidden,
                force_pfc=False, force_dls=False,
                lesion_pfc=False, lesion_dls=False,
                dls_only=False, pfc_only=False):
        """
        dls_only=True: run DLS only — PFC parameters never touched.
        pfc_only=True: run PFC only — DLS parameters never touched.
        """
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2: x = x.unsqueeze(1)
        B, T, _ = x.shape
        pfc_h, dls_h = hidden

        if pfc_only:
            pfc_out, new_pfc_h = self.pfc_rnn(x, pfc_h)
            if lesion_pfc: pfc_out = torch.zeros_like(pfc_out)
            fp = pfc_out.reshape(B * T, -1)
            pfc_logits = self.pfc_actor(fp).reshape(B, T, -1)
            pfc_policy = F.softmax(pfc_logits, dim=-1)
            pfc_value  = self.pfc_critic(fp).reshape(B, T)
            if T == 1:
                pfc_policy = pfc_policy.squeeze(1)
                pfc_value  = pfc_value.squeeze(1)
                pfc_logits = pfc_logits.squeeze(1)
            return (pfc_policy, pfc_value, None, None,
                    pfc_policy, new_pfc_h, dls_h,
                    pfc_logits, None)

        dls_out, new_dls_h = self.dls_rnn(x, dls_h)
        if lesion_dls: dls_out = torch.zeros_like(dls_out)
        fd = dls_out.reshape(B * T, -1)
        dls_logits = self.dls_actor(fd).reshape(B, T, -1)
        dls_policy = F.softmax(dls_logits, dim=-1)
        dls_value  = self.dls_critic(fd).reshape(B, T)

        if dls_only:
            if T == 1:
                dls_policy = dls_policy.squeeze(1)
                dls_value  = dls_value.squeeze(1)
                dls_logits = dls_logits.squeeze(1)
            return (None, None, dls_policy, dls_value,
                    dls_policy, pfc_h, new_dls_h,
                    None, dls_logits)

        pfc_out, new_pfc_h = self.pfc_rnn(x, pfc_h)
        if lesion_pfc: pfc_out = torch.zeros_like(pfc_out)
        fp = pfc_out.reshape(B * T, -1)
        pfc_logits = self.pfc_actor(fp).reshape(B, T, -1)
        pfc_policy = F.softmax(pfc_logits, dim=-1)
        pfc_value  = self.pfc_critic(fp).reshape(B, T)

        use_dls   = force_dls or lesion_pfc
        pi_active = dls_policy if use_dls else pfc_policy

        if T == 1:
            pfc_policy = pfc_policy.squeeze(1); dls_policy = dls_policy.squeeze(1)
            pi_active  = pi_active.squeeze(1)
            pfc_value  = pfc_value.squeeze(1)
            dls_value  = dls_value.squeeze(1)
            pfc_logits = pfc_logits.squeeze(1); dls_logits = dls_logits.squeeze(1)

        return (pfc_policy, pfc_value, dls_policy, dls_value,
                pi_active, new_pfc_h, new_dls_h,
                pfc_logits, dls_logits)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def compute_gae(rewards, dones, values):
    T   = len(rewards)
    adv = torch.zeros(T, device=rewards.device)
    ret = torch.zeros(T, device=rewards.device)
    gae = 0.0; nv = 0.0
    for t in reversed(range(T)):
        delta  = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        gae    = delta + GAMMA * GAE_LAMBDA * gae * (1 - dones[t])
        adv[t] = gae; ret[t] = gae + values[t]; nv = values[t].item()
    return adv, ret


def entropy_coeff(mean_acc, choice_bias, timeout_rate, curr_prog):
    e = (0.1*(1 - 0.5*curr_prog) + 0.4*choice_bias**2
         + 0.3*timeout_rate - 0.08*max(0., mean_acc-0.5))
    return float(np.clip(e, 0.05, 0.5))


def participation_ratio(hidden_states: np.ndarray,
                        noise_floor: float = None) -> float:
    h   = hidden_states - hidden_states.mean(axis=0, keepdims=True)
    cov = h.T @ h / max(len(h) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    n = len(eigvals)
    if noise_floor is None:
        noise_floor = float(np.median(eigvals[:n//2]))
    eigvals = eigvals - noise_floor
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0: return 1.0
    return float((eigvals.sum() ** 2) / (eigvals ** 2).sum())


def dls_singular_values(model: "TwoAreaACNet") -> np.ndarray:
    with torch.no_grad():
        W = (model.dls_rnn.m @ model.dls_rnn.n.t()).cpu().numpy()
    _, s, _ = np.linalg.svd(W, full_matrices=False)
    return s[:model.dls_rnn.rank]


def _cleanup_old_checkpoints(path: str, keep: int = 2):
    ckpts = sorted(glob.glob(os.path.join(path, "checkpoint_*.pt")),
                   key=os.path.getmtime)
    for old in ckpts[:-keep]:
        os.remove(old)


def save_rank_analysis(path, rank, step, pfc_solo, dls_solo,
                       gate_mean, pr_dls, singular_vals, pr_source='all'):
    csv_path   = os.path.join(os.path.dirname(path), "rank_sweep_summary.csv")
    sv_str     = ";".join(f"{v:.4f}" for v in singular_vals)
    row        = (f"{rank},{step},{pfc_solo:.4f},{dls_solo:.4f},"
                  f"{gate_mean:.4f},{pr_dls:.4f},{pr_source},{sv_str}\n")
    write_hdr  = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_hdr:
            f.write("rank,step,pfc_solo_acc,dls_solo_acc,gate_mean,"
                    "pr_dls,pr_source,singular_vals\n")
        f.write(row)
    print(f"  [rank_sweep] rank={rank} step={step} "
          f"pfc_solo={pfc_solo:.3f} dls_solo={dls_solo:.3f} "
          f"PR_dls={pr_dls:.2f}({pr_source}) "
          f"sv={np.array2string(singular_vals, precision=3)}")


# ─────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def solo_eval(model, envs, n_episodes=200,
              force_pfc=False, force_dls=False,
              fixed_delay=None, device=device):
    """Returns (accuracy, choice_reach_rate) tuple."""
    saved_delay = envs.len_delay
    if fixed_delay is not None:
        envs.len_delay = fixed_delay
    envs.reset()
    obs = envs.warm_start() if envs.force_init_loc else envs.get_vectorized_observation()

    hidden  = model.init_hidden(1, device)
    correct = 0; reached_choice = 0; completed = 0

    while completed < n_episodes:
        obs_t = torch.from_numpy(obs).float().to(device)
        (_, _, _, _, pi_active,
         new_pfc_h, new_dls_h, _, _) = model(
            obs_t, hidden,
            force_pfc=force_pfc, force_dls=force_dls)
        action = Categorical(pi_active.squeeze(0)).sample()
        hidden = (new_pfc_h, new_dls_h)
        obs, _, just_done, infos = envs.step(action.item())
        if just_done:
            hidden = (torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1]))
            if envs.force_init_loc:
                obs = envs.warm_start()
            if 'episode' in infos:
                correct        += int(infos['episode']['correct'])
                reached_choice += int(infos['episode'].get('choice', 0) != 0)
                completed      += 1

    envs.len_delay = saved_delay
    return correct / n_episodes, reached_choice / n_episodes


@torch.no_grad()
def delay_eval_grid(model, envs, path, rank, step,
                    n_episodes=100, delays=None, device=device):
    if delays is None: delays = EVAL_DELAYS
    results = {}
    print(f"\n  [delay_eval_grid]  rank={rank}  step={step:,}")
    print(f"  {'delay':>6}  {'PFC':>8}  {'DLS':>8}  {'mixed':>8}")
    for d in delays:
        p,  _ = solo_eval(model, envs, n_episodes, force_pfc=True,  fixed_delay=d, device=device)
        dl, _ = solo_eval(model, envs, n_episodes, force_dls=True,  fixed_delay=d, device=device)
        m,  _ = solo_eval(model, envs, n_episodes,                  fixed_delay=d, device=device)
        results[d] = {'pfc': p, 'dls': dl, 'mixed': m}
        print(f"  {d:>6}  {p:>8.3f}  {dl:>8.3f}  {m:>8.3f}")

    csv_path  = os.path.join(os.path.dirname(path), "delay_rank_grid.csv")
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_hdr: f.write("rank,step,delay,pfc_solo,dls_solo,mixed\n")
        for d, res in results.items():
            f.write(f"{rank},{step},{d},{res['pfc']:.4f},"
                    f"{res['dls']:.4f},{res['mixed']:.4f}\n")

    dl_list = list(results.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dl_list, [results[d]['pfc']   for d in dl_list], 'b-o', ms=6, label='PFC solo')
    ax.plot(dl_list, [results[d]['dls']   for d in dl_list], 'r-o', ms=6, label='DLS solo')
    ax.plot(dl_list, [results[d]['mixed'] for d in dl_list], 'k--s', ms=5, label='Mixed')
    ax.axhline(0.5, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Delay'); ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1.05)
    ax.set_title(f'Rank={rank} | Delay capacity profile'); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"delay_profile_rank{rank}.png"), dpi=150)
    plt.close(fig)
    return results


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────
def run_one(dls_rank_override=None, max_delay_override=None, difficulty_override=None):
    rank       = dls_rank_override   if dls_rank_override   is not None else DLS_RANK
    max_delay  = max_delay_override  if max_delay_override  is not None else MAX_DELAY
    difficulty = difficulty_override if difficulty_override is not None else DIFFICULTY
    ### I want to involve the override of len_edge : python ... --len_edge 7 for example, but can not distinguish len_edge = 5 from len_edge = 7 in the tag, since both will be recorded as len_edge 5 in the tag for now, solve this by recording the effective len_edge (which is 80% of the original len_edge) in the tag, since the effective len_edge determines the actual difficulty of the task. ###
    len_edge_use = LEN_EDGE
    
    best_rwd    = float('-inf')
    best_len    = float('inf')

    tag  = (datetime.now().strftime("%Y_%m_%d")
            + f"_two_area_RPEarb"
            + f"_pfc{PFC_HIDDEN}_dls{DLS_HIDDEN}rk{rank}"
            + f"_sd{START_DELAY}_md{max_delay}_diff{difficulty}_seed{SEED}_len{len_edge_use}"
            + ("_LPFC" if LESION_PFC else "")
            + ("_LDLS" if LESION_DLS else ""))
    path = os.path.join(SAVE_PATH, tag)
    os.makedirs(path, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    shutil.copy(args.config, os.path.join(path, "params.py"))
    with open(os.path.join(path, "config_summary.txt"), "w") as f:
        f.write(f"Run: {tag}\nConfig: {args.config}\n"
                f"CLI overrides: { {k:v for k,v in _overrides.items() if v is not None} }\n\n"
                "Effective parameters:\n")
        for k, v in sorted(cfg.items()):
            f.write(f"  {k:<22} = {v}\n")
        f.write(f"\n  DLS_RANK (this run)   = {rank}\n"
                f"  MAX_DELAY (this run)  = {max_delay}\n"
                f"  DIFFICULTY (this run) = {difficulty}\n"
                f"  LEN_EDGE (this run)    = {len_edge_use}\n")

    envs  = VectorizedTunlVisual(len_delay=START_DELAY, difficulty=difficulty)
    model = TwoAreaACNet(dls_rank=rank).to(device)

    opt_pfc = optim.Adam(list(model.pfc_rnn.parameters()) +
                         list(model.pfc_actor.parameters()) +
                         list(model.pfc_critic.parameters()), lr=LR_PFC)
    opt_dls = optim.Adam(list(model.dls_rnn.parameters()) +
                         list(model.dls_actor.parameters()) +
                         list(model.dls_critic.parameters()), lr=LR_DLS)

    # ── Training state ────────────────────────────────────────────────
    step_count    = 0; episode_count = 0
    current_delay = START_DELAY
    ema_fast = 0.5; ema_slow = 0.5; stable_steps = 0
    warmup_done  = (PFC_WARMUP_STEPS == 0)
    warmup_step  = 0
    pfc_frozen   = False   # True once pfc_solo >= PFC_FREEZE_ACC and reach >= PFC_FREEZE_REACH
    dls_acting   = False
    dls_solo_acc = 0.5
    dls_solo_ema = 0.5
    dls_reach    = 0.0
    dls_ce_agree = 0.5
    pfc_solo_acc = 0.0
    pfc_reach    = 0.0
    pfc_len_ema  = float(MAX_EPISODE_STEPS)
    dls_len_ema  = float(MAX_EPISODE_STEPS)
    best_acc     = 0.0
    ret_window   = deque(maxlen=1500)
    dret_window  = deque(maxlen=1500)
    suboptimal_pfc = True
    
    # ── Demonstration REPLAY Buffer ───────────────────────────────
    # Stores PFC's best successful episodes for DLS to learn from repeatedly
    # Quality criteria: correct choice + short episode length + high confidence
    demo_buffer = deque(maxlen=500)  # keep last 500 high-quality episodes
    demo_use_prob = 0.3  # 30% of DLS updates use replayed demos instead of live data

    ep_rewards=[]; ep_lengths=[]; ep_correct=[]; ep_choices=[]
    pfc_ep_correct=[]; pfc_ep_lengths=[]   # PFC-only episodes
    dls_ep_correct=[]; dls_ep_lengths=[]   # DLS-only episodes
    gate_log=[]; value_gap_log=[]
    dls_solo_log=[]; eval_steps=[]   # for plotting solo acc over time

    next_log_step  = LOG_EVERY
    next_eval_step = EVAL_EVERY
    next_plot_step = PLOT_EVERY
    next_ckpt_step = CHECKPOINT_EVERY

    # ── Resume ────────────────────────────────────────────────────────
    ckpt_files  = sorted(glob.glob(os.path.join(path, "checkpoint_*.pt")),
                         key=os.path.getmtime)
    resume_path = ckpt_files[-1] if ckpt_files else None
    ### if best_model exists, prefer the best model checkpoint for resuming, to avoid losing the best found solution due to a crash or interruption. ###
    best_model_path = os.path.join(path, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Found existing best model checkpoint: {best_model_path}")
        resume_path = best_model_path
    if args.resume: resume_path = args.resume

    if resume_path:
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        opt_pfc.load_state_dict(ckpt['opt_pfc'])
        opt_dls.load_state_dict(ckpt['opt_dls'])
        step_count    = ckpt['step_count']
        episode_count = ckpt['episode_count']
        best_acc      = ckpt['best_acc']
        current_delay = ckpt['current_delay']
        ema_fast      = ckpt['ema_fast']
        ema_slow      = ckpt['ema_slow']
        stable_steps  = ckpt['stable_steps']
        warmup_done   = ckpt['warmup_done']
        warmup_step   = ckpt.get('warmup_step', step_count)
        pfc_frozen    = ckpt.get('pfc_frozen', False)
        dls_acting    = ckpt.get('dls_acting', False)
        dls_solo_acc  = ckpt.get('dls_solo_acc', 0.5)
        dls_solo_ema  = ckpt.get('dls_solo_ema', 0.5)
        dls_reach     = ckpt.get('dls_reach', 0.0)
        dls_ce_agree  = ckpt.get('dls_ce_agree', 0.5)
        pfc_solo_acc  = ckpt.get('pfc_solo_acc', 0.0)
        pfc_reach     = ckpt.get('pfc_reach', 0.0)
        pfc_len_ema   = ckpt.get('pfc_len_ema', float(MAX_EPISODE_STEPS))
        dls_len_ema   = ckpt.get('dls_len_ema', float(MAX_EPISODE_STEPS))
        next_log_step  = ckpt.get('next_log_step',  step_count + LOG_EVERY)
        next_eval_step = ckpt.get('next_eval_step', step_count + EVAL_EVERY)
        next_plot_step = ckpt.get('next_plot_step', step_count + PLOT_EVERY)
        next_ckpt_step = ckpt.get('next_ckpt_step', step_count + CHECKPOINT_EVERY)
        gate_log       = ckpt.get('gate_log', [])
        value_gap_log  = list(ckpt.get('value_gap_log', []))
        dls_solo_log   = list(ckpt.get('dls_solo_log', []))
        eval_steps     = list(ckpt.get('eval_steps', []))
        ep_rewards     = list(ckpt.get('ep_rewards', []))
        ep_lengths     = list(ckpt.get('ep_lengths', []))
        ep_correct     = list(ckpt.get('ep_correct', []))
        ep_choices     = list(ckpt.get('ep_choices', []))
        suboptimal_pfc = ckpt.get('suboptimal_pfc',True) ### for recording significant time point data
        ### best_rwd and best_len are not saved in the checkpoint, but we can reconstruct them from the episode rewards and lengths if needed. ###
        if ep_rewards and dls_len_ema:
            best_rwd = np.mean(ep_rewards[-100:])
            best_len = dls_len_ema
        # Restore demo buffer
        if 'demo_buffer' in ckpt:
            demo_buffer = deque(ckpt['demo_buffer'], maxlen=500)
            print(f"  Restored demo buffer: {len(demo_buffer)} episodes")
        if warmup_done: 
            envs.force_init_loc = True
        if dls_acting:
            envs.max_episode_steps = int(MAX_EPISODE_STEPS/5)
        envs.len_delay = current_delay
        print(f"  step={step_count}  delay={current_delay}  "
              f"acting={'DLS' if dls_acting else 'PFC'}  "
              f"warmup={'done' if warmup_done else 'pending'}")

    # ── Main loop ─────────────────────────────────────────────────────
    hidden = model.init_hidden(N_ENVS, device)
    obs    = envs.get_vectorized_observation()
    start_time = time.time()
    print("Starting training…\n")
    just_start = False

    while step_count < N_STEPS:

        # ── Collect one episode ───────────────────────────────────────
        # PFC always acts until handover. DLS never touches the env pre-handover.
        buf_obs=[]; buf_acts=[]; buf_rews=[]; buf_dones=[]; buf_pfc_v=[]; buf_dls_v = []
        pfc_h0 = hidden[0].detach()
        dls_h0 = hidden[1].detach()
        done   = False

        while not done and step_count < N_STEPS:
            if warmup_done and len(buf_obs) == 0 and just_start:
                # print(envs.phase, envs.current_loc, envs.initiation_loc,obs)
                assert obs[0] == envs.initiation_loc[1] / (envs.w - 1.0)
                assert obs[1] == envs.initiation_loc[0] / (envs.h - 1.0)
            if dls_acting and len(buf_obs) == 0 and just_start:
                assert envs.max_episode_steps == int(MAX_EPISODE_STEPS/5)
            obs_t = torch.from_numpy(obs).float().to(device)
            just_start = True
            with torch.no_grad():
                if dls_acting:
                    # Post-handover: DLS drives behavior
                    (_, _, dls_pol, dls_val, _, new_pfc_h, new_dls_h, _, _) = model(
                        obs_t, hidden, dls_only=True)
                    action = Categorical(dls_pol.squeeze(0)).sample()
                    # When DLS is acting (Stage 2-3), use deterministic action selection:
                    # action = dls_pol.squeeze(0).argmax()
                    buf_dls_v.append(dls_val.squeeze(0))
                else:
                    # Pre-handover: PFC drives behavior, collect pfc_val for A2C
                    (pfc_pol, pfc_val, _, _, _, new_pfc_h, new_dls_h, _, _) = model(
                        obs_t, hidden, pfc_only=True)
                    action = Categorical(pfc_pol.squeeze(0)).sample()
                    buf_pfc_v.append(pfc_val.squeeze(0))
                hidden = (new_pfc_h, new_dls_h)

            buf_obs.append(obs_t.squeeze(0))
            buf_acts.append(action)
            buf_rews.append(0.0)   # placeholder; filled below
            buf_dones.append(0.0)  # placeholder; filled below

            obs, reward, just_done, infos = envs.step(action.item())
            buf_rews[-1]  = reward
            buf_dones[-1] = float(just_done)
            done = just_done

            if just_done:
                hidden = (torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1]))
                if envs.force_init_loc:
                    obs = envs.warm_start()
            if 'episode' in infos:
                ei = infos['episode']
                ep_rewards.append(ei['r']); ep_lengths.append(ei['l'])
                ep_correct.append(float(ei['correct']))
                ep_choices.append(ei.get('choice',0)-1 if ei.get('choice',0)>0 else -1)
                if dls_acting:
                    dls_ep_correct.append(float(ei['correct']))
                    dls_ep_lengths.append(ei['l'])
                else:
                    pfc_ep_correct.append(float(ei['correct']))
                    pfc_ep_lengths.append(ei['l'])

            step_count += 1

        if len(buf_obs) == 0: continue
        episode_count += 1

        obs_seq  = torch.stack(buf_obs).unsqueeze(0)
        acts_seq = torch.stack(buf_acts)
        rews_t   = torch.tensor(buf_rews,  device=device, dtype=torch.float32)
        dones_t  = torch.tensor(buf_dones, device=device, dtype=torch.float32)

        pfc_src = pfc_ep_correct if pfc_ep_correct else ep_correct
        src_lengths = pfc_ep_lengths if pfc_ep_lengths else ep_lengths
        m_acc  = np.mean(pfc_src[-100:]) if pfc_src else 0.5
        m_ch   = np.mean([c for c in ep_choices[-100:] if c >= 0]) if ep_choices else 0.5
        to_rt  = (sum(1 for l in src_lengths[-50:] if l >= MAX_EPISODE_STEPS-1)
                  / max(len(src_lengths[-50:]), 1))
        c_prog = (current_delay - START_DELAY) / max(max_delay - START_DELAY, 1)
        ent_c  = max(0.08, entropy_coeff(m_acc, abs(m_ch-0.5)*2, to_rt, c_prog))

        # ══ PFC update (only pre-freeze) ══════════════════════════════
        if not pfc_frozen and not dls_acting and buf_pfc_v:
            pfc_vals = torch.stack(buf_pfc_v)
            pfc_adv, pfc_ret = compute_gae(rews_t, dones_t, pfc_vals)
            ret_window.extend(pfc_ret.cpu().numpy().tolist())
            ret_mean = float(np.mean(ret_window))
            ret_std  = max(float(np.std(ret_window)) if len(ret_window) > 1 else 1.0, 1.0)
            pfc_ret  = (pfc_ret - ret_mean) / ret_std
            pfc_adv  = (pfc_adv - pfc_adv.mean()) / (pfc_adv.std() + 1e-8)

            (pfc_pol_s, pfc_val_s, _, _, _, _, _, _, _) = model(
                obs_seq, (pfc_h0, dls_h0), pfc_only=True)
            pfc_pol_s = pfc_pol_s.squeeze(0)
            pfc_val_s = pfc_val_s.squeeze(0)
            pfc_dist  = Categorical(pfc_pol_s)
            pfc_loss  = (-(pfc_dist.log_prob(acts_seq) * pfc_adv).mean()
                         + VALUE_COEFF * F.mse_loss(pfc_val_s, pfc_ret)
                         - ent_c * pfc_dist.entropy().mean())
            opt_pfc.zero_grad()
            pfc_loss.backward()
            nn.utils.clip_grad_norm_(list(model.pfc_rnn.parameters()) +
                                     list(model.pfc_actor.parameters()) +
                                     list(model.pfc_critic.parameters()), GRAD_CLIP)
            opt_pfc.step()

        # ── Store high-quality demonstrations for replay ──────────────
        # Only store when PFC is acting (not DLS) and episode succeeded
        if not dls_acting and ep_correct and ep_correct[-1]:
            ep_len = ep_lengths[-1] if ep_lengths else MAX_EPISODE_STEPS
            # Quality threshold: correct + reasonably efficient (< 80% of max length)
            is_high_quality = ep_len < 0.8 * MAX_EPISODE_STEPS
            
            # Extra quality check: compute average PFC confidence during this episode
            with torch.no_grad():
                (pfc_pol_temp, _, _, _, _, _, _, _, _) = model(
                    obs_seq, (pfc_h0, dls_h0), pfc_only=True)
                pfc_pol_temp = pfc_pol_temp.squeeze(0)
                pfc_ent = -(pfc_pol_temp * torch.log(pfc_pol_temp + 1e-8)).sum(dim=-1)
                avg_conf_episode = (1.0 - pfc_ent.mean().item() / np.log(pfc_pol_temp.shape[-1]))
            
            # Store if high quality (efficient) OR high confidence (>0.6)
            if is_high_quality or avg_conf_episode > 0.6:
                demo_buffer.append({
                    'obs_seq': obs_seq.detach().clone(),
                    'pfc_h0': pfc_h0.detach().clone(),
                    'dls_h0': dls_h0.detach().clone(),
                    'dones_t': dones_t.detach().clone(),
                    'quality': avg_conf_episode,  # for potential prioritized sampling
                    'length': ep_len
                })
        
        # ── Noise Filtering: Choose one configuration ─────────────────
        
        # OPTION A (Conservative): Always filter failures
        # Use if: PFC accuracy stays < 85% even after warmup
        pfc_succeeded = (ep_correct and ep_correct[-1]) if not dls_acting else True
        min_confidence = 1.0  # DISABLED - was 0.2, causing tiny gradients
        amplify_confidence = False
        
        # OPTION B (Balanced): Filter failures during warmup only
        # Use if: PFC reaches >85% accuracy after warmup
        # Assumption: After warmup, PFC is good enough that confidence weighting alone suffices
        # pfc_succeeded = (ep_correct and ep_correct[-1]) if (not dls_acting and not warmup_done) else True
        # Breakdown:
        #   - During warmup (warmup_done=False): pfc_succeeded = ep_correct[-1]  (filter failures)
        #   - After warmup (warmup_done=True):   pfc_succeeded = True            (no filter, rely on confidence)
        #   - After handover (dls_acting=True):  pfc_succeeded = True            (DLS acting anyway)
        # min_confidence = 1.0  # DISABLED - was 0.2, causing tiny gradients
        # amplify_confidence = False
    
        
        if not dls_acting and pfc_succeeded:  # ← Success filter applied here
            
            # ── Decide: use live episode or replay from buffer ───────────
            use_replay = (len(demo_buffer) >= 50 and 
                         np.random.random() < demo_use_prob and 
                         warmup_done)  # only use replay after warmup
            
            if use_replay:
                # Sample a high-quality demonstration from buffer
                # Prioritized sampling: favor higher quality episodes
                qualities = np.array([demo['quality'] for demo in demo_buffer])
                probs = qualities / qualities.sum()  # probability proportional to quality
                idx = np.random.choice(len(demo_buffer), p=probs)
                demo = demo_buffer[idx]
                
                # Use replayed demonstration
                obs_seq_use = demo['obs_seq']
                pfc_h0_use = demo['pfc_h0']
                dls_h0_use = demo['dls_h0']
                dones_t_use = demo['dones_t']
            else:
                # Use live episode
                obs_seq_use = obs_seq
                pfc_h0_use = pfc_h0
                dls_h0_use = dls_h0
                dones_t_use = dones_t
            
            # ── Imitation phase: DLS learns from PFC (frozen or not) ──
            # Get PFC's current policy as teaching signal
            with torch.no_grad():
                (pfc_pol_s, _, _, _, _, _, _, _, _) = model(
                    obs_seq_use, (pfc_h0_use, dls_h0_use), pfc_only=True)
                pfc_soft = pfc_pol_s.squeeze(0).detach()   # [T, A] soft targets
                
                # Compute PFC confidence: entropy of policy (lower = more confident)
                # High confidence (low entropy) → weight closer to 1
                # Low confidence (high entropy) → weight closer to 0
                pfc_entropy = -(pfc_soft * torch.log(pfc_soft + 1e-8)).sum(dim=-1)  # [T]
                max_entropy = np.log(pfc_soft.shape[-1])  # log(num_actions)
                confidence = 1.0 - (pfc_entropy / max_entropy)  # [T], range [0, 1]
                
                # Optional: amplify differences (makes filter more selective)
                if amplify_confidence:
                    confidence = torch.exp(3.0 * (confidence - 0.5))
                
                # Clamp to avoid learning from very noisy decisions
                confidence = torch.clamp(confidence, min=min_confidence)
                confidence = torch.clamp(confidence, min=min_confidence)

            # DLS learns PFC's full action distribution (not just argmax)
            # Weighted by PFC's confidence at each timestep
            (_, _, dls_pol_s, _, _, _, _, _, _) = model(
                obs_seq_use.detach(), (pfc_h0_use, dls_h0_use), dls_only=True)
            dls_pol_s = dls_pol_s.squeeze(0)           # [T, A]

            # KL(pfc || dls) = sum_a pfc*log(pfc/dls), weighted by confidence
            kl_per_step = (pfc_soft * torch.log((pfc_soft + 1e-8) / (dls_pol_s + 1e-8))).sum(dim=-1)
            dls_loss = (kl_per_step * confidence).mean()  # confidence-weighted
        

            # CE agreement diagnostic: terminal step
            with torch.no_grad():
                pfc_greedy = pfc_soft.argmax(dim=-1)
                terminal_mask = dones_t_use.bool()
                if terminal_mask.any():
                    agree = (dls_pol_s.detach().argmax(-1)[terminal_mask] ==
                             pfc_greedy[terminal_mask]).float().mean().item()
                    dls_ce_agree = 0.98 * dls_ce_agree + 0.02 * agree


            opt_dls.zero_grad()
            dls_loss.backward()
            
            nn.utils.clip_grad_norm_(list(model.dls_rnn.parameters()) +
                                     list(model.dls_actor.parameters()) +
                                     list(model.dls_critic.parameters()), GRAD_CLIP)
            opt_dls.step()

        elif dls_acting and buf_dls_v:
            # ── DLS acting phase: soft KL at terminal + efficiency RL ──
            # DLS acts in env. Nav KL invalid (different trajectory).
            # Soft KL at terminal step only — correct choice from frozen PFC.
            # Full efficiency RL drives path compression.
            for param in opt_dls.param_groups:
                param['lr'] = LR_DLS * 0.25
            with torch.no_grad():
                (pfc_pol_s, _, _, _, _, _, _, _, _) = model(
                    obs_seq, (pfc_h0, dls_h0), pfc_only=True)
                pfc_soft = pfc_pol_s.squeeze(0).detach()

            (_, _, dls_pol_s, dls_val_s, _, _, _, _, _) = model(
                obs_seq.detach(), (pfc_h0, dls_h0), dls_only=True)
            dls_pol_s = dls_pol_s.squeeze(0)
            dls_val_s = dls_val_s.squeeze(0)
            # dls_vals = torch.stack(buf_dls_v) ### similar to buf_pfc_v but for DLS values
            ### or on-policy values? but on-policy might be too noisy, since DLS is still learning and might produce bad trajectories, so using the values collected during acting seems more stable for now. ###
            dls_vals = dls_val_s.detach() # ensure no backprop through critic targets
            
            is_terminal = dones_t.bool()
            if is_terminal.any():
                dls_kl = F.kl_div(
                    torch.log(dls_pol_s[is_terminal] + 1e-8),
                    pfc_soft[is_terminal],
                    reduction='batchmean')
            else:
                dls_kl = torch.tensor(0.0, device=device)

            # Efficiency reward: step cost + completion bonus + error penalty
            
            ## for length_edge = 5, difficulty = 2, 0 * DLS_COMPLETION_BONUS works 
            # terminal_r = 0*DLS_COMPLETION_BONUS
            
            terminal_r = 1*DLS_COMPLETION_BONUS ### for hard task, we want to strongly incentivize completion, since random exploration is unlikely to succeed, and we want to make sure DLS learns to complete the task as soon as it starts learning. For easier tasks, we can reduce this bonus to encourage more exploration and gradual learning of efficient paths. ###
            if DLS_PENALTY_ERROR:
                # print('sensitive to error penalty')
                if ep_correct and not ep_correct[-1]:
                    terminal_r += 0.1 * INC_RWD   # penalise wrong choice
                    ### I add 0.1 to make the reaching more asymmetric### len_edge = 5-- 0.2
                    
            ## for length_edge = 5, difficulty = 2, this works
            # dls_eff_t = torch.tensor(
            #     [STEP_RWD] * (len(buf_rews)-1) + [0.2*terminal_r],
            #     device=device, dtype=torch.float32)
            
            dls_eff_t = torch.tensor(
                [STEP_RWD] * (len(buf_rews)-1) + [1.0*terminal_r],
                device=device, dtype=torch.float32)
            dls_adv, dls_ret = compute_gae(dls_eff_t, dones_t, dls_vals)
            
            dret_window.extend(dls_ret.cpu().numpy().tolist())
            dret_mean = float(np.mean(dret_window))
            dret_std  = max(float(np.std(dret_window)) if len(dret_window) > 1 else 1.0, 1.0)
            dls_ret  = (dls_ret - dret_mean) / dret_std
            
            dls_ret  = dls_ret.detach()  # value targets should not backprop through critic
            dls_adv  = (dls_adv - dls_adv.mean()) / (dls_adv.std() + 1e-8)
            dls_adv  = dls_adv.detach()  # policy advantages should not backprop through critic
            dls_dist = Categorical(dls_pol_s)
            energy_coeff = 1.0-dls_ce_agree#min(1.0-dls_ce_agree, 1.0-dls_solo_ema) if dls_ce_agree is not None else 1.0 - dls_solo_ema
            dls_a2c  = (-(dls_dist.log_prob(acts_seq) * dls_adv).mean()
                        + VALUE_COEFF * F.mse_loss(dls_val_s, dls_ret)
                        - ent_c * energy_coeff * dls_dist.entropy().mean()) # 
                        # - ent_c * (1.0 - dls_solo_ema) * dls_dist.entropy().mean()) #  
                        # - ent_c * (1.0 - dls_ce_agree) * dls_dist.entropy().mean()) # this is correct, as during the refinement, dls is still free of the correct answer but pfc...AND !!! dls_ce_agree is more temporal?
            dls_loss = 0*CHOICE_WEIGHT * dls_kl + EFFICIENCY_WEIGHT * 1.0 * dls_a2c#3.0 * dls_a2c

            with torch.no_grad():
                pfc_greedy = pfc_soft.argmax(dim=-1)
                if is_terminal.any():
                    agree = (dls_pol_s.detach().argmax(-1)[is_terminal] ==
                             pfc_greedy[is_terminal]).float().mean().item()
                    dls_ce_agree = 0.98 * dls_ce_agree + 0.02 * agree
            opt_dls.zero_grad()
            dls_loss.backward()
            nn.utils.clip_grad_norm_(list(model.dls_rnn.parameters()) +
                                     list(model.dls_actor.parameters()) +
                                     list(model.dls_critic.parameters()), GRAD_CLIP)
            opt_dls.step()
        pfc_src = pfc_ep_correct if pfc_ep_correct else ep_correct
        if pfc_src:
            nc       = float(pfc_src[-1])
            ema_fast = 0.95*ema_fast + 0.05*nc
            ema_slow = 0.99*ema_slow + 0.01*nc
            stable_steps += 1
        if (not warmup_done
                and step_count >= PFC_WARMUP_STEPS
                and ema_fast   >= WARMUP_ACC_THRESH
                # and ema_slow   >= WARMUP_ACC_THRESH ### ema_slow might be contaminated by early random successes, so only require ema_fast to trigger warmup end
                and current_delay == max_delay):
            warmup_done = True
            warmup_step = step_count
            envs.force_init_loc = True
            # ### set shorter max_seq 
            # envs.max_episode_steps = int(MAX_EPISODE_STEPS/5)
            ### save the warmup_done model 
            ckpt = {
                'model_state_dict': model.state_dict(),
                'opt_pfc':          opt_pfc.state_dict(),
                'opt_dls':          opt_dls.state_dict(),
                'step_count':       step_count,
                'episode_count':    episode_count,
                'best_acc':         best_acc,
                'current_delay':    current_delay,
                'ema_fast':         ema_fast,
                'ema_slow':         ema_slow,
                'stable_steps':     stable_steps,
                'warmup_done':      warmup_done,
                'warmup_step':      warmup_step,
                'pfc_frozen':       pfc_frozen,
                'dls_acting':       dls_acting,
                'dls_solo_acc':     dls_solo_acc,
                'dls_solo_ema':     dls_solo_ema,
                'dls_reach':        dls_reach,
                'dls_ce_agree':     dls_ce_agree,
                'pfc_solo_acc':     pfc_solo_acc,
                'pfc_reach':        pfc_reach,
                'next_eval_step':   next_eval_step,
                'dls_solo_log':     dls_solo_log,
                'eval_steps':       eval_steps,
                'pfc_len_ema':      pfc_len_ema,
                'dls_len_ema':      dls_len_ema,
                'next_log_step':    next_log_step,
                'next_plot_step':   next_plot_step,
                'next_ckpt_step':   next_ckpt_step,
                'gate_log':         gate_log[-1000:],
                'value_gap_log':    value_gap_log[-500:],
                'ep_rewards':       ep_rewards[-500:],
                'ep_lengths':       ep_lengths[-500:],
                'ep_correct':       ep_correct[-500:],
                'ep_choices':       ep_choices[-500:],
                'dls_rank':         rank,
                'suboptimal_pfc':   suboptimal_pfc,
            }
            ckpt_path = os.path.join(path, f"warmup_model.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  -> warmup model saved: {ckpt_path} with reward {best_rwd:.2f} and length {best_len:.1f}")
            print(f"\n  [WARMUP DONE] step={step_count}  "
                  f"pfc_acc={ema_fast:.3f}\n")

        # ── Curriculum (only while PFC is training) ───────────────────
        if pfc_src and not pfc_frozen:
            stab = abs(ema_fast - ema_slow)
            if (ema_fast >= TARGET_ACC and stab < 0.08
                    and stable_steps >= 200 and current_delay < max_delay):
                current_delay = max(1, current_delay+1) if current_delay > 0 else START_DELAY or 1
                envs.len_delay = current_delay; stable_steps = 0
                print(f"\n  PROMOTE delay → {current_delay}\n")
            elif (ema_fast < 0.35 and stab < 0.08
                    and stable_steps >= 200 and current_delay > START_DELAY):
                current_delay -= 1; envs.len_delay = current_delay; stable_steps = 0
                print(f"\n  DEMOTE  delay → {current_delay}\n")

        gate_log.append(0.0 if dls_acting else 1.0)

        # ── Episode length EMA (post-handover: tracks DLS efficiency) ─
        if ep_lengths:
            ep_len = float(ep_lengths[-1])
            if dls_acting:
                dls_len_ema = 0.95 * dls_len_ema + 0.05 * ep_len
            else:
                pfc_len_ema = 0.95 * pfc_len_ema + 0.05 * ep_len

        # ── Periodic DLS solo eval (ground truth for handover) ────────
        if warmup_done and step_count >= next_eval_step:
            next_eval_step = step_count + EVAL_EVERY
            saved_walls   = envs.walls.copy()
            saved_rng     = envs.rng.get_state()
            saved_hidden  = (hidden[0].clone(), hidden[1].clone())
            dls_solo_acc, dls_reach = solo_eval(model, envs, n_episodes=100,
                                                 force_dls=True, device=device)
            pfc_solo_acc, pfc_reach = solo_eval(model, envs, n_episodes=100,
                                                 force_pfc=True, device=device)
            # Fully restore env state so eval doesn't disrupt training trajectory
            envs.walls = saved_walls
            envs.rng.set_state(saved_rng)
            envs.reset(force_init_loc=envs.force_init_loc)
            assert envs.force_init_loc, "force_init_loc should be True after reset in eval"
            obs    = envs.warm_start() if envs.force_init_loc else envs.get_vectorized_observation()
            # print('warm start....', obs)
            hidden = saved_hidden
            dls_solo_ema = 0.7 * dls_solo_ema + 0.3 * dls_solo_acc
            dls_solo_log.append(dls_solo_ema)
            eval_steps.append(step_count)

            # PFC freeze: once PFC is good enough, stop training it
            # ema_slow to ensure stability, pfc_reach to ensure it's not just getting the choice right but also navigating well
            # also reward threshold to ensure overall performance is good (not just passing by chance)
            if (not pfc_frozen and warmup_done
                    and pfc_solo_acc >= PFC_FREEZE_ACC
                    and ema_slow >= PFC_FREEZE_ACC 
                    and np.mean(ep_rewards[-100:]) >= 0.4
                    and pfc_reach    >= PFC_FREEZE_REACH):
                pfc_frozen = True
                print(f"\n  [PFC FROZEN] step={step_count}  "
                      f"pfc_solo={pfc_solo_acc:.3f}  pfc_reach={pfc_reach:.2f}  "
                      f"PFC is now a fixed teacher\n")
                
            if (not pfc_frozen and warmup_done and suboptimal_pfc
                    and ema_slow >= DLS_HANDOVER_FRAC * PFC_FREEZE_ACC#
                    and np.mean(ep_rewards[-100:]) >= 0.4):
                suboptimal_pfc = False
                
                ckpt = {
                    'model_state_dict': model.state_dict(),
                    'opt_pfc':          opt_pfc.state_dict(),
                    'opt_dls':          opt_dls.state_dict(),
                    'step_count':       step_count,
                    'episode_count':    episode_count,
                    'best_acc':         best_acc,
                    'current_delay':    current_delay,
                    'ema_fast':         ema_fast,
                    'ema_slow':         ema_slow,
                    'stable_steps':     stable_steps,
                    'warmup_done':      warmup_done,
                    'warmup_step':      warmup_step,
                    'pfc_frozen':       pfc_frozen,
                    'dls_acting':       dls_acting,
                    'dls_solo_acc':     dls_solo_acc,
                    'dls_solo_ema':     dls_solo_ema,
                    'dls_reach':        dls_reach,
                    'dls_ce_agree':     dls_ce_agree,
                    'pfc_solo_acc':     pfc_solo_acc,
                    'pfc_reach':        pfc_reach,
                    'next_eval_step':   next_eval_step,
                    'dls_solo_log':     dls_solo_log,
                    'eval_steps':       eval_steps,
                    'pfc_len_ema':      pfc_len_ema,
                    'dls_len_ema':      dls_len_ema,
                    'next_log_step':    next_log_step,
                    'next_plot_step':   next_plot_step,
                    'next_ckpt_step':   next_ckpt_step,
                    'gate_log':         gate_log[-1000:],
                    'value_gap_log':    value_gap_log[-500:],
                    'ep_rewards':       ep_rewards[-500:],
                    'ep_lengths':       ep_lengths[-500:],
                    'ep_correct':       ep_correct[-500:],
                    'ep_choices':       ep_choices[-500:],
                    'dls_rank':         rank,
                    'suboptimal_pfc':   suboptimal_pfc,
                }
                ckpt_path = os.path.join(path, f"suboptimalpfc_model.pt")
                torch.save(ckpt, ckpt_path)
                print(f"  -> suboptimal pfc model saved: {ckpt_path}")

                print(f"\n  [Suboptimal PFC] step={step_count}  "
                      f"pfc_solo={pfc_solo_acc:.3f}  pfc_reach={pfc_reach:.2f}  "
                      f"PFC now reaches the suboptimal\n")
                

            print(f"\n  [EVAL] step={step_count}  "
                  f"dls_solo={dls_solo_acc:.3f}(reach={dls_reach:.2f})  "
                  f"dls_ema={dls_solo_ema:.3f}  "
                  f"pfc_solo={pfc_solo_acc:.3f}(reach={pfc_reach:.2f})  "
                  f"gap={dls_solo_ema-pfc_solo_acc:+.3f}  "
                  f"ce_agree={dls_ce_agree:.3f}  "
                  f"pfc={'FROZEN' if pfc_frozen else 'training'}  "
                  f"acting={'DLS' if dls_acting else 'PFC'}\n")

        # ── DLS handover: DLS reaches DLS_HANDOVER_FRAC of PFC performance ─
        if pfc_frozen and not dls_acting:
            handover_thresh = DLS_HANDOVER_FRAC * PFC_FREEZE_ACC#pfc_solo_acc
            if dls_solo_ema >= handover_thresh:
                dls_acting = True
                ### set shorter max_seq 
                envs.max_episode_steps = int(MAX_EPISODE_STEPS/5)
                ### make the learning rate of dls smaller to stabilise post-handover learning (optional)
                for param_group in opt_dls.param_groups:
                    param_group['lr'] = LR_DLS * 0.25
                print(f"\n  [HANDOVER → DLS] step={step_count}  "
                      f"dls_ema={dls_solo_ema:.3f} >= {handover_thresh:.3f} "
                      f"({DLS_HANDOVER_FRAC:.0%} of pfc={pfc_solo_acc:.3f})  "
                      f"pfc_len={pfc_len_ema:.1f}\n")

        if dls_acting:
            fallback_thresh = DLS_FALLBACK_THRESH * pfc_solo_acc
            if dls_solo_ema < fallback_thresh:
                dls_acting = False
                ### set shorter max_seq 
                envs.max_episode_steps = int(MAX_EPISODE_STEPS)
                ### restore PFC learning rate after fallback (optional)
                for param_group in opt_pfc.param_groups:
                    param_group['lr'] = LR_PFC
                for param_group in opt_dls.param_groups:
                    param_group['lr'] = LR_DLS
                print(f"\n  [FALLBACK → PFC] step={step_count}  "
                      f"dls_ema={dls_solo_ema:.3f} < {fallback_thresh:.3f}  "
                      f"PFC unfrozen\n")

        # ── Logging ───────────────────────────────────────────────────
        if step_count >= next_log_step and ep_correct:
            next_log_step += LOG_EVERY
            sps      = step_count / (time.time() - start_time)
            phase    = "WARMUP" if not warmup_done else ("DLS" if dls_acting else ("PFC❄" if pfc_frozen else "PFC"))
            pfc_acc  = (f"{pfc_solo_acc:.3f}❄" if pfc_frozen
                        else f"{np.mean(pfc_ep_correct[-100:]):.3f}" if pfc_ep_correct
                        else "nan")
            dls_acc  = (f"{np.mean(dls_ep_correct[-100:]):.3f}" if dls_ep_correct
                        else "nan")
            dls_info = (f"dls_ema={dls_solo_ema:.3f}  ce={dls_ce_agree:.3f}  dls_len={dls_len_ema:.0f}"
                        if warmup_done else f"warmup_steps={max(0, step_count-warmup_step):,}")
            print(f"Step {step_count:8d} | ep={episode_count:5d} | "
                  f"delay={current_delay} | pfc_acc={pfc_acc} | dls_acc={dls_acc} | "
                  f"acting={phase} | {dls_info} | pfc_len={pfc_len_ema:.0f} | "
                  f"rwd={np.mean(ep_rewards[-100:]):.2f} | {sps:.0f} sps")
            ## if dls is acting, save the best model with the smallest dls_len and largest rwd!!!
            if dls_acting:
                if np.mean(ep_rewards[-100:])>best_rwd and int(dls_len_ema)<best_len:
                    best_rwd = np.mean(ep_rewards[-100:])
                    best_len = int(dls_len_ema)
                    ### save the best model 
                    ckpt = {
                        'model_state_dict': model.state_dict(),
                        'opt_pfc':          opt_pfc.state_dict(),
                        'opt_dls':          opt_dls.state_dict(),
                        'step_count':       step_count,
                        'episode_count':    episode_count,
                        'best_acc':         best_acc,
                        'current_delay':    current_delay,
                        'ema_fast':         ema_fast,
                        'ema_slow':         ema_slow,
                        'stable_steps':     stable_steps,
                        'warmup_done':      warmup_done,
                        'warmup_step':      warmup_step,
                        'pfc_frozen':       pfc_frozen,
                        'dls_acting':       dls_acting,
                        'dls_solo_acc':     dls_solo_acc,
                        'dls_solo_ema':     dls_solo_ema,
                        'dls_reach':        dls_reach,
                        'dls_ce_agree':     dls_ce_agree,
                        'pfc_solo_acc':     pfc_solo_acc,
                        'pfc_reach':        pfc_reach,
                        'next_eval_step':   next_eval_step,
                        'dls_solo_log':     dls_solo_log,
                        'eval_steps':       eval_steps,
                        'pfc_len_ema':      pfc_len_ema,
                        'dls_len_ema':      dls_len_ema,
                        'next_log_step':    next_log_step,
                        'next_plot_step':   next_plot_step,
                        'next_ckpt_step':   next_ckpt_step,
                        'gate_log':         gate_log[-1000:],
                        'value_gap_log':    value_gap_log[-500:],
                        'ep_rewards':       ep_rewards[-500:],
                        'ep_lengths':       ep_lengths[-500:],
                        'ep_correct':       ep_correct[-500:],
                        'ep_choices':       ep_choices[-500:],
                        'dls_rank':         rank,
                        'suboptimal_pfc':   suboptimal_pfc,
                    }
                    ckpt_path = os.path.join(path, f"best_model.pt")
                    torch.save(ckpt, ckpt_path)
                    print(f"  -> best model saved: {ckpt_path} with reward {best_rwd:.2f} and length {best_len:.1f}")
                    

        # ── Plots ─────────────────────────────────────────────────────
        if step_count >= next_plot_step and len(ep_correct) > 10:
            next_plot_step += PLOT_EVERY
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"Two-Area | rank={rank} | Step {step_count:,} | Delay {current_delay}")

            axs[0,0].plot(ep_correct[-500:]); axs[0,0].axhline(TARGET_ACC, color='r', ls='--')
            axs[0,0].set_title('Accuracy (episode)'); axs[0,0].set_ylim(0, 1.05)

            axs[0,1].plot(ep_rewards[-500:]); axs[0,1].set_title('Episode reward (PFC signal)')

            axs[0,2].plot(ep_lengths[-500:])
            axs[0,2].set_title(f'Episode length  pfc={pfc_len_ema:.0f}  dls={dls_len_ema:.0f}')

            axs[1,0].plot(gate_log[-2000:]); axs[1,0].set_ylim(-0.05, 1.05)
            axs[1,0].set_title('Acting: 1=PFC  0=DLS')

            if dls_solo_log:
                axs[1,1].plot(eval_steps, dls_solo_log, 'r-o', ms=3, label='DLS solo')
                axs[1,1].axhline(DLS_HANDOVER_FRAC * pfc_solo_acc, color='g', ls='--', lw=0.8, label='handover')
                axs[1,1].axhline(DLS_FALLBACK_THRESH, color='r', ls='--', lw=0.8, label='fallback')
                axs[1,1].set_ylim(0, 1.05); axs[1,1].legend(fontsize=8)
            axs[1,1].set_title('DLS solo accuracy (ground truth)')

            if value_gap_log:
                axs[1,2].plot(value_gap_log[-500:])
                axs[1,2].set_title('Value gap |V_pfc − V_dls|  (different scales by design)')

            plt.tight_layout()
            # plt.savefig(os.path.join(path, f"progress_{step_count}.png"), dpi=120)
            plt.close(fig)

        # ── Checkpoint ────────────────────────────────────────────────
        if step_count >= next_ckpt_step:
            next_ckpt_step += CHECKPOINT_EVERY
            ckpt = {
                'model_state_dict': model.state_dict(),
                'opt_pfc':          opt_pfc.state_dict(),
                'opt_dls':          opt_dls.state_dict(),
                'step_count':       step_count,
                'episode_count':    episode_count,
                'best_acc':         best_acc,
                'current_delay':    current_delay,
                'ema_fast':         ema_fast,
                'ema_slow':         ema_slow,
                'stable_steps':     stable_steps,
                'warmup_done':      warmup_done,
                'warmup_step':      warmup_step,
                'pfc_frozen':       pfc_frozen,
                'dls_acting':       dls_acting,
                'dls_solo_acc':     dls_solo_acc,
                'dls_solo_ema':     dls_solo_ema,
                'dls_reach':        dls_reach,
                'dls_ce_agree':     dls_ce_agree,
                'pfc_solo_acc':     pfc_solo_acc,
                'pfc_reach':        pfc_reach,
                'next_eval_step':   next_eval_step,
                'dls_solo_log':     dls_solo_log,
                'eval_steps':       eval_steps,
                'pfc_len_ema':      pfc_len_ema,
                'dls_len_ema':      dls_len_ema,
                'next_log_step':    next_log_step,
                'next_plot_step':   next_plot_step,
                'next_ckpt_step':   next_ckpt_step,
                'gate_log':         gate_log[-1000:],
                'value_gap_log':    value_gap_log[-500:],
                'ep_rewards':       ep_rewards[-500:],
                'ep_lengths':       ep_lengths[-500:],
                'ep_correct':       ep_correct[-500:],
                'ep_choices':       ep_choices[-500:],
                'dls_rank':         rank,
                'demo_buffer':      list(demo_buffer),  # save replay buffer
            }
            ckpt_path = os.path.join(path, f"checkpoint_{step_count}.pt")
            torch.save(ckpt, ckpt_path)
            # _cleanup_old_checkpoints(path, keep=2)
            print(f"  -> checkpoint: {ckpt_path}")

        if step_count >= N_STEPS - 1:
            delay_eval_grid(model, envs, path, rank, step_count, device=device)

    print(f"\nDone.  steps={step_count}  episodes={episode_count}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if RUN_MODE == "single":
        run_one()
    elif RUN_MODE == "pairs":
        for rank, md, diff in FIXED_PAIRS:
            run_one(dls_rank_override=rank,
                    max_delay_override=md,
                    difficulty_override=diff)
    elif RUN_MODE == "sweep":
        for rank in RANK_SWEEP_LIST:
            run_one(dls_rank_override=rank)