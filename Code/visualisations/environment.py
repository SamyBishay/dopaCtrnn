"""DNMTP T-maze (faithful in spirit to Villet et al., 2025).

Layout (row, col), passable cells marked; walls elsewhere:

    (0,0) (0,1) (0,2) (0,3) (0,4)      <- top row: L-end .. JUNCTION .. R-end
                (1,2)                  <- stem
                (2,2)  = START

Trial = SAMPLE (forced into the open arm, reward at its end)
      -> DELAY (held at start, no cue, must hold "which arm was sampled")
      -> TEST  (both arms open; reward for the PREVIOUSLY BLOCKED arm = non-match).

Two observation streams implement the modelling asymmetry:
  - allocentric (goal-directed): full position map + phase + cues  -> can plan
  - egocentric  (habitual): local wall sensors + place-type + prev action + clock -> cannot plan
"""
import numpy as np

ROWS, COLS = 3, 5
START = (2, 2)
STEM = (1, 2)
JUNCTION = (0, 2)
L_END = (0, 0)
R_END = (0, 4)
TOP = [(0, c) for c in range(COLS)]
PASSABLE_ALL = set(TOP) | {STEM, START}

# action deltas: 0=N 1=S 2=E 3=W
DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}

ALLO_DIM = 22
EGO_DIM = 14


def maze_layout():
    """Static geometry for the visualizer (JSON-friendly)."""
    return {"rows": ROWS, "cols": COLS,
            "passable": [list(c) for c in sorted(PASSABLE_ALL)],
            "start": list(START), "junction": list(JUNCTION),
            "l_end": list(L_END), "r_end": list(R_END)}


def _blocked_cells(blocked_side):
    """Cells made impassable during the SAMPLE phase to force the open arm."""
    if blocked_side == "L":
        return {(0, 0), (0, 1)}
    else:
        return {(0, 3), (0, 4)}


class TMazeDNMTP:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.reward_scale = 1.0   # devaluation can set this to ~0 (see analysis.py)
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.blocked = self.rng.choice(["L", "R"])
        self.open_side = "R" if self.blocked == "L" else "L"
        self.phase = "sample"
        self.pos = START
        self.prev_action = -1          # -1 = none
        self.delay_idx = 0
        self.test_steps = 0
        self.done = False
        # scripted forced path to the open arm end
        if self.open_side == "R":
            self._sample_path = [0, 0, 2, 2]   # N N E E -> (0,4)
        else:
            self._sample_path = [0, 0, 3, 3]   # N N W W -> (0,0)
        self._sample_idx = 0
        return self.obs()

    # ----------------------------------------------------------- passability
    def _passable(self, cell):
        if cell not in PASSABLE_ALL:
            return False
        if self.phase == "sample" and cell in _blocked_cells(self.blocked):
            return False
        return True

    def _wall(self, direction):
        dr, dc = DELTAS[direction]
        return not self._passable((self.pos[0] + dr, self.pos[1] + dc))

    def _place_type(self):
        if self.pos == START:
            return 0
        if self.pos == STEM:
            return 1
        if self.pos == JUNCTION:
            return 2
        return 3  # arm

    # --------------------------------------------------------------- observe
    def obs(self):
        # ---- allocentric (22) ----
        allo = np.zeros(ALLO_DIM, dtype=np.float32)
        allo[self.pos[0] * COLS + self.pos[1]] = 1.0           # occupancy (15)
        phase_idx = {"sample": 0, "delay": 1, "test": 2}[self.phase]
        allo[15 + phase_idx] = 1.0                              # phase one-hot (3)
        if self.phase == "sample":                             # blocked cue (2), sample only
            allo[18 + (0 if self.blocked == "L" else 1)] = 1.0
        # open-arm cue (2): sample = open arm; test = both; delay = none
        if self.phase == "sample":
            allo[20 + (0 if self.open_side == "L" else 1)] = 1.0
        elif self.phase == "test":
            allo[20] = 1.0
            allo[21] = 1.0

        # ---- egocentric (14) ----
        ego = np.zeros(EGO_DIM, dtype=np.float32)
        for d in range(4):
            ego[d] = 1.0 if self._wall(d) else 0.0             # wall sensors (4)
        ego[4 + self._place_type()] = 1.0                      # place type (4)
        if self.prev_action >= 0:
            ego[8 + self.prev_action] = 1.0                    # prev action (4)
        ego[12] = 1.0 if self.phase == "delay" else 0.0        # clock (2)
        ego[13] = 1.0 if self.phase == "test" else 0.0
        return allo, ego

    @property
    def agent_controlled(self):
        return self.phase == "test"

    # ------------------------------------------------------------------ step
    def step(self, action):
        task_r, int_r, agent_step, correct = 0.0, 0.0, False, None

        if self.phase == "sample":
            a = self._sample_path[self._sample_idx]
            self._move(a)
            self.prev_action = a
            self._sample_idx += 1
            int_r -= self.cfg.step_cost
            if self._sample_idx >= len(self._sample_path):
                task_r += self.cfg.sample_reward
                self.phase = "delay"
                self.pos = START          # return to start box before the delay

        elif self.phase == "delay":
            self.prev_action = -1
            self.delay_idx += 1
            int_r -= self.cfg.wait_cost
            if self.delay_idx >= self.cfg.delay_steps:
                self.phase = "test"

        elif self.phase == "test":
            agent_step = True
            self._move(action)             # blocked moves keep position
            self.prev_action = action
            self.test_steps += 1
            int_r -= self.cfg.step_cost
            task_r -= self.cfg.step_cost      # shaping for the value system (GD); devaluable path
            if self.pos in (L_END, R_END):
                reached = "L" if self.pos == L_END else "R"
                correct = (reached == self.blocked)     # non-match rule
                if correct:
                    task_r += self.cfg.test_reward * self.reward_scale
                int_r += self.cfg.completion_bonus
                self.done = True
            elif self.test_steps >= self.cfg.max_test_steps:
                correct = False
                self.done = True

        info = {"agent_step": agent_step, "correct": correct, "phase": self.phase}
        return task_r, int_r, self.done, info

    def _move(self, action):
        dr, dc = DELTAS[action]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        if self._passable(nxt):
            self.pos = nxt
