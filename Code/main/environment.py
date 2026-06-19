"""DNMTP T-maze with free navigation — non-trivial working memory task.

The agent navigates freely throughout all phases. Arm identity must be held
in recurrent hidden state across the delay; it cannot be read from the
observation during delay or choice.

Layout (row, col):

    (0,0) (0,1) (0,2) (0,3) (0,4)   <- L-end .. JUNCTION .. R-end
                (1,2)                <- stem
                (2,2)  = START

Phases (in order):
  pre_sample  : navigate from START to JUNCTION (triggers sample display)
  sample      : one arm blocked; navigate to the open arm end
  delay       : free movement; phase signal decays ×0.1 per step
  choice      : navigate to the non-match arm (correct) or wrong arm

Observation (6D):
  [col/(COLS-1), row/(ROWS-1), 0,
   sig_L, sig_R, sig_choice]

  sig_L/R = 0.25 during sample (indicating which arm was open), else 0
  sig_choice = 0.25 during choice phase
  All three decay ×0.1 per step during delay, so by step 5 they are < 3e-5.

Actions: 0=N  1=S  2=E  3=W  4=WAIT
"""
import numpy as np

ROWS, COLS = 3, 5
START    = (2, 2)
STEM     = (1, 2)
JUNCTION = (0, 2)
L_ARM1   = (0, 1)
L_END    = (0, 0)
R_ARM1   = (0, 3)
R_END    = (0, 4)
TOP      = [(0, c) for c in range(COLS)]
PASSABLE_ALL = set(TOP) | {STEM, START}

DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (0, 0)}  # 4=WAIT

OBS_DIM = 6
_SIG_VAL = 1.0 / (COLS - 1)   # 0.25 — phase signal amplitude


def maze_layout():
    return {"rows": ROWS, "cols": COLS,
            "passable": [list(c) for c in sorted(PASSABLE_ALL)],
            "start": list(START), "junction": list(JUNCTION),
            "l_end": list(L_END), "r_end": list(R_END)}


def _blocked_cells(blocked_side):
    if blocked_side == "L":
        return {L_ARM1, L_END}
    return {R_ARM1, R_END}


class TMazeFreeNav:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.current_delay = cfg.delay_start
        self.reward_scale = 1.0   # set to ~0 for devaluation (H3)
        self.reset()

    def advance_delay(self):
        self.current_delay = min(self.current_delay + 1, self.cfg.delay_max)

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.blocked   = self.rng.choice(["L", "R"])
        self.open_side = "R" if self.blocked == "L" else "L"
        self.phase     = "pre_sample"
        self.pos       = START
        self.delay_idx = 0
        self.step_count = 0
        self.done      = False
        self._phase_signal = np.zeros(3, dtype=np.float32)
        return self.obs()

    # ----------------------------------------------------------- passability
    def _passable(self, cell):
        if cell not in PASSABLE_ALL:
            return False
        if self.phase == "sample" and cell in _blocked_cells(self.blocked):
            return False
        return True

    # --------------------------------------------------------------- observe
    def obs(self):
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[0] = self.pos[1] / (COLS - 1)   # normalized column (x)
        o[1] = self.pos[0] / (ROWS - 1)   # normalized row (y)
        # o[2] = 0  (prev-action slot, left zero like supervisor)
        o[3:6] = self._phase_signal
        return o

    @property
    def agent_controlled(self):
        return True   # agent navigates throughout

    # ------------------------------------------------------------------ step
    def step(self, action):
        task_r, int_r, correct = 0.0, 0.0, None
        self.step_count += 1

        # --- move (WAIT keeps position) ---
        dr, dc = DELTAS[action]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        if nxt in PASSABLE_ALL and self._passable(nxt):
            self.pos = nxt

        # --- per-step efficiency cost ---
        cost = self.cfg.wait_cost if action == 4 else self.cfg.step_cost
        int_r -= cost
        task_r -= cost   # step-cost shaping for goal-directed value system

        # --- phase transitions ---
        if self.phase == "pre_sample" and self.pos == JUNCTION:
            self.phase = "sample"
            # set phase signal: L_open → sig_L, R_open → sig_R
            self._phase_signal[:] = 0.0
            if self.open_side == "L":
                self._phase_signal[0] = _SIG_VAL
            else:
                self._phase_signal[1] = _SIG_VAL

        elif self.phase == "sample":
            sample_end = L_END if self.open_side == "L" else R_END
            if self.pos == sample_end:
                self.phase = "delay"
                self.delay_idx = 0

        elif self.phase == "delay":
            self._phase_signal *= 0.1   # exponential decay
            self.delay_idx += 1
            if self.delay_idx >= self.current_delay:
                self.phase = "choice"
                self._phase_signal[:] = 0.0
                self._phase_signal[2] = _SIG_VAL   # "go-time" signal

        elif self.phase == "choice":
            if self.pos in (L_END, R_END):
                reached = "L" if self.pos == L_END else "R"
                correct = (reached == self.blocked)   # non-match rule
                if correct:
                    task_r += self.cfg.test_reward * self.reward_scale
                int_r += self.cfg.completion_bonus
                self.done = True

        # --- timeout ---
        if not self.done and self.step_count >= self.cfg.max_episode_steps:
            correct = False
            self.done = True

        info = {"agent_step": True, "correct": correct, "phase": self.phase}
        return task_r, int_r, self.done, info
