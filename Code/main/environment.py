"""DNMTP T-maze with free navigation — vectorised over B parallel envs.

This is the vectorised rewrite. All environment state carries a leading
batch dimension B; one `step(actions)` call advances every environment in
C-level NumPy (boolean masks instead of per-env Python branches), removing
the per-environment Python loop that was the single-process bottleneck.

Parametric grid (matches the supervisor's difficulty system)
─────────────────────────────────────────────────────────────
The maze size is set by two knobs, exactly as in the reference code:

  len_edge (odd, >=5)  -> grid WIDTH:  w = len_edge + 2
  difficulty (0/1/2)   -> grid HEIGHT: longer stem = harder
        difficulty 0 (easy)  : h = (len_edge - 1)//2 + 1
        difficulty 1 (medium): h = (len_edge + 1)//2 + 1
        difficulty 2 (hard)  : h = (len_edge + 1)//2 + 2

Difficulty controls the length of the central STEM the agent walks between
START and the arm row. A taller grid = a longer stem = more steps (and more
chance for the held memory to be disturbed) between sampling and choosing.
The arms and the non-match rule are identical across difficulties; only the
distance/time from sample to choice grows.

Layout (row, col), arms on row 1 like the reference env:

    row 0   : ###############        (wall border)
    row 1   : # L .. JCT .. R #      <- arm row: L_END .. JUNCTION .. R_END
    row 2   : #      |       #       <- stem
     ...            |
    row h-2 : #     S       #        <- START (bottom of stem)
    row h-1 : ###############

Phases (TUNL protocol):
  pre_sample : navigate from START up the stem to JUNCTION (shows sample)
  sample     : one arm blocked; navigate to the open arm end
  delay      : agent teleported back to START and held there (no movement, no step cost);
               phase signal decays x0.1 per step — models the inter-trial holding box
  choice     : agent released from START; navigate to the non-match arm (correct) or wrong arm

Observation — goal-directed (obs, dim 6):
  [col/(w-1), row/(h-1), 0, sig_L, sig_R, sig_choice]
Observation — habitual (obs_hab, dim 4):
  [0, sig_L, sig_R, sig_choice]   (no position; reactive system)

Actions: 0=N  1=S  2=E  3=W  4=WAIT
"""
import numpy as np

PRE_SAMPLE, SAMPLE, DELAY, CHOICE = 0, 1, 2, 3
_PHASE_NAME = {PRE_SAMPLE: "pre_sample", SAMPLE: "sample",
               DELAY: "delay", CHOICE: "choice"}

_DR = np.array([-1, 1, 0, 0, 0])
_DC = np.array([0, 0, 1, -1, 0])

OBS_DIM     = 6
OBS_DIM_HAB = 4


def grid_dims(len_edge, difficulty):
    """Return (h, w) for a given len_edge / difficulty — the supervisor's rule.

    Raises if the configuration yields a degenerate grid with no stem: START
    sits at row h-2 and the arm row is row 1, so the stem length is h-3. A
    stem length < 1 means START is on (or above) the arm row — there is no
    corridor to traverse and the working-memory demand collapses. Use a larger
    len_edge or a higher difficulty.
    """
    assert len_edge % 2 == 1 and len_edge >= 5, "len_edge must be odd and >= 5"
    w = len_edge + 2
    if   difficulty == 0: h = (len_edge - 1) // 2 + 1
    elif difficulty == 1: h = (len_edge + 1) // 2 + 1
    elif difficulty == 2: h = (len_edge + 1) // 2 + 2
    else: raise ValueError(f"Invalid difficulty: {difficulty}")
    stem = h - 3   # rows strictly between START (h-2) and the arm row (1)
    if stem < 1:
        raise ValueError(
            f"Degenerate grid: len_edge={len_edge}, difficulty={difficulty} "
            f"gives h={h} (stem length {stem}). START would sit on the arm row "
            f"with no corridor. Use len_edge>=7, or raise difficulty "
            f"(len_edge=5 needs difficulty>=2; len_edge>=7 works at any difficulty).")
    return int(h), int(w)


def _geometry(len_edge, difficulty):
    """Static cell coordinates + passable mask for the parametric grid."""
    h, w = grid_dims(len_edge, difficulty)
    cx = (w - 1) // 2
    arm_row = 1
    junction = (arm_row, cx)
    l_end = (arm_row, 1)
    r_end = (arm_row, w - 2)
    start = (h - 2, cx)

    passable = np.zeros((h, w), dtype=bool)
    passable[arm_row, 1:w - 1] = True       # arm corridor
    passable[arm_row:h - 1, cx] = True       # central stem

    if h > 6:                                # large-grid antechamber (reference)
        passable[2, cx - 2:cx + 3] = True
        passable[3, cx - 1:cx + 2] = True

    return {"h": h, "w": w, "cx": cx, "arm_row": arm_row,
            "junction": junction, "l_end": l_end, "r_end": r_end,
            "start": start, "passable": passable}


def maze_layout(cfg=None):
    """Layout dict for the visualiser. Uses cfg's len_edge/difficulty if given."""
    le   = getattr(cfg, "len_edge", 5) if cfg is not None else 5
    diff = getattr(cfg, "difficulty", 0) if cfg is not None else 0
    g = _geometry(le, diff)
    ys, xs = np.where(g["passable"])
    return {"rows": g["h"], "cols": g["w"],
            "passable": [[int(y), int(x)] for y, x in zip(ys, xs)],
            "start": list(g["start"]), "junction": list(g["junction"]),
            "l_end": list(g["l_end"]), "r_end": list(g["r_end"])}


class TMazeVecEnv:
    """Vectorised env over B parallel trials. step() takes a [B] action array
    and returns [B] reward/done arrays."""
    def __init__(self, cfg, rng, batch_size):
        self.cfg = cfg
        self.rng = rng
        self.B   = int(batch_size)
        self.current_delay = getattr(cfg, "delay_start", cfg.delay)
        self.reward_scale  = 1.0

        self.g = _geometry(getattr(cfg, "len_edge", 5),
                           getattr(cfg, "difficulty", 0))
        self.h, self.w = self.g["h"], self.g["w"]
        self._passable = self.g["passable"]
        self._junction = np.array(self.g["junction"])
        self._l_end    = np.array(self.g["l_end"])
        self._r_end    = np.array(self.g["r_end"])
        self._start    = np.array(self.g["start"])
        cx = self.g["cx"]
        self._l_arm_cols = np.arange(1, cx)
        self._r_arm_cols = np.arange(cx + 1, self.w - 1)
        self._arm_row = self.g["arm_row"]
        # Pushback path: arm end → junction (cx-1 horizontal steps) → start (h-3 vertical steps)
        self._pushback_len = (self.g["cx"] - 1) + (self.h - 3)
        delay_start = getattr(cfg, "delay_start", cfg.delay)
        assert delay_start >= self._pushback_len, (
            f"delay_start={delay_start} must be >= pushback_len={self._pushback_len} "
            f"for grid len_edge={cfg.len_edge}, difficulty={cfg.difficulty}")
        assert cfg.delay >= delay_start, \
            f"delay ceiling={cfg.delay} must be >= delay_start={delay_start}"
        self.pos = None
        self.reset()

    def advance_delay(self):
        """Increment current_delay by one step toward cfg.delay (curriculum ceiling)."""
        step = max(1, (self.cfg.delay - getattr(self.cfg, "delay_start", self.cfg.delay)) // 8)
        self.current_delay = min(self.cfg.delay, self.current_delay + step)

    def reset(self, mask=None):
        """Reset all envs, or only those where mask is True."""
        B = self.B
        if mask is None:
            mask = np.ones(B, dtype=bool)
        n = int(mask.sum())
        if self.pos is None:
            self.pos        = np.tile(self._start, (B, 1)).astype(np.int64)
            self.blocked    = np.empty(B, dtype="<U1")
            self.open_side  = np.empty(B, dtype="<U1")
            self.phase      = np.zeros(B, dtype=np.int64)
            self.delay_idx  = np.zeros(B, dtype=np.int64)
            self.step_count = np.zeros(B, dtype=np.int64)
            self.done       = np.zeros(B, dtype=bool)
            self.correct    = np.zeros(B, dtype=bool)
            self._sig       = np.zeros((B, 3), dtype=np.float32)
            self._confined  = np.zeros(B, dtype=np.float32)
        if n == 0:
            return self.obs()
        idx = np.where(mask)[0]
        blk = self.rng.integers(0, 2, size=n)
        self.blocked[idx]    = np.where(blk == 0, "L", "R")
        self.open_side[idx]  = np.where(blk == 0, "R", "L")
        self.pos[idx]        = self._start
        self.phase[idx]      = PRE_SAMPLE
        self.delay_idx[idx]  = 0
        self.step_count[idx] = 0
        self.done[idx]       = False
        self.correct[idx]    = False
        self._sig[idx]       = 0.0
        self._confined[idx]  = 0.0
        return self.obs()

    def _passable_target(self, ny, nx, phase, blocked):
        B = self.B
        inb = (ny >= 0) & (ny < self.h) & (nx >= 0) & (nx < self.w)
        ok = np.zeros(B, dtype=bool)
        ok[inb] = self._passable[ny[inb], nx[inb]]
        in_sample = phase == SAMPLE
        if in_sample.any():
            on_arm_row = ny == self._arm_row
            blk_L = (blocked == "L")
            left_cols  = np.isin(nx, self._l_arm_cols)
            right_cols = np.isin(nx, self._r_arm_cols)
            seal = in_sample & on_arm_row & (
                (blk_L & left_cols) | (~blk_L & right_cols))
            ok &= ~seal
        return ok

    def obs(self):
        o = np.zeros((self.B, OBS_DIM), dtype=np.float32)
        o[:, 0] = self.pos[:, 1] / (self.w - 1)
        o[:, 1] = self.pos[:, 0] / (self.h - 1)
        o[:, 2] = self._confined  # confinement signal: 1.0 when agent is held at START in DELAY
        o[:, 3:6] = self._sig
        return o

    def obs_hab(self):
        # E5 (habit_obs="allocentric"): hab sees the same obs as GD (includes position).
        if getattr(self.cfg, "habit_obs", "position_free") == "allocentric":
            return self.obs()
        # Default: position-free [0, sig_L, sig_R, sig_choice]
        o = np.zeros((self.B, OBS_DIM_HAB), dtype=np.float32)
        o[:, 1:4] = self._sig
        return o

    @property
    def agent_controlled(self):
        return True

    def step(self, actions):
        a = np.asarray(actions).reshape(self.B)
        active = ~self.done
        B = self.B
        task_r = np.zeros(B, dtype=np.float32)
        int_r  = np.zeros(B, dtype=np.float32)

        self.step_count += active.astype(np.int64)

        # Agents in the delay phase are held at START (holding-box model): no movement,
        # no step cost. Mirrors the supervisor's `if not self.indelay: reward = step_rwd`.
        in_delay = (self.phase == DELAY)

        ny = self.pos[:, 0] + _DR[a]
        nx = self.pos[:, 1] + _DC[a]
        legal = self._passable_target(ny, nx, self.phase, self.blocked) & active & ~in_delay
        self.pos[legal, 0] = ny[legal]
        self.pos[legal, 1] = nx[legal]

        is_wait = (a == 4)
        cost = np.where(is_wait, self.cfg.wait_cost, self.cfg.step_cost).astype(np.float32)
        nav_active = active & ~in_delay
        int_r  -= cost * nav_active
        task_r -= cost * nav_active

        py, px = self.pos[:, 0], self.pos[:, 1]

        # PRE_SAMPLE -> SAMPLE
        m = active & (self.phase == PRE_SAMPLE)
        at_jct = m & (py == self._junction[0]) & (px == self._junction[1])
        just_entered_sample = np.zeros(self.B, dtype=bool)
        if at_jct.any():
            self.phase[at_jct] = SAMPLE
            task_r[at_jct] += self.cfg.junction_bonus
            self._sig[at_jct] = 0.0
            open_L = at_jct & (self.open_side == "L")
            open_R = at_jct & (self.open_side == "R")
            sample_sig = 1.0 / (self.w - 1)
            self._sig[open_L, 0] = sample_sig
            self._sig[open_R, 1] = sample_sig
            just_entered_sample = at_jct

        # SAMPLE -> DELAY (skip envs that just entered SAMPLE this step)
        m = active & (self.phase == SAMPLE) & (~just_entered_sample)
        end_y = self._l_end[0]
        open_is_L = (self.open_side == "L")
        tgt_x = np.where(open_is_L, self._l_end[1], self._r_end[1])
        at_open_end = m & (py == end_y) & (px == tgt_x)
        just_entered_delay = np.zeros(self.B, dtype=bool)
        if at_open_end.any():
            self.phase[at_open_end] = DELAY
            task_r[at_open_end] += self.cfg.arm_end_bonus
            self.delay_idx[at_open_end] = 0
            just_entered_delay = at_open_end

        # DELAY -> CHOICE (skip envs that just entered DELAY this step)
        m = active & (self.phase == DELAY) & (~just_entered_delay)
        just_entered_choice = np.zeros(self.B, dtype=bool)
        if m.any():
            self._sig[m] *= 0.1
            self.delay_idx[m] += 1

            # PUSHBACK: programmatically walk agents from arm end back to START.
            # delay_idx has just been incremented, so steps 1..pushback_len are pushback.
            in_pushback = m & (self.delay_idx <= self._pushback_len)
            if in_pushback.any():
                idx_pb = np.where(in_pushback)[0]
                py_pb = self.pos[idx_pb, 0]
                px_pb = self.pos[idx_pb, 1]
                open_pb = self.open_side[idx_pb]

                on_arm_row = (py_pb == self._arm_row)
                at_jct_col = (px_pb == self._junction[1])
                in_arm_not_jct = on_arm_row & ~at_jct_col

                dy = np.zeros(len(idx_pb), dtype=np.int64)
                dx = np.zeros(len(idx_pb), dtype=np.int64)

                # Arm phase: move toward junction horizontally
                dx[in_arm_not_jct & (open_pb == "L")] = 1    # L arm → East
                dx[in_arm_not_jct & (open_pb == "R")] = -1   # R arm → West

                # Stem phase (at junction or below): move South toward START
                in_stem_or_jct = ~in_arm_not_jct
                at_start_row = (py_pb == self._start[0])
                dy[in_stem_or_jct & ~at_start_row] = 1

                self.pos[idx_pb, 0] = py_pb + dy
                self.pos[idx_pb, 1] = px_pb + dx

            # CONFINED: agent has reached START (pushback complete), hold there.
            # Set obs[2]=1.0 as a confinement signal. Agents can't move anyway
            # (in_delay mask already blocks all action-driven movement in step()).
            in_confined = m & (self.delay_idx > self._pushback_len)
            self._confined[in_confined] = 1.0

            # CHOICE transition
            to_choice = m & (self.delay_idx >= self.current_delay)
            if to_choice.any():
                self.phase[to_choice] = CHOICE
                self._confined[to_choice] = 0.0   # clear confinement signal
                self._sig[to_choice] = 0.0
                choice_sig = (1.0 / (self.w - 1)) * 0.1
                self._sig[to_choice, 2] = choice_sig
                just_entered_choice = to_choice

        # CHOICE -> terminal. An env that *just* entered CHOICE this step is
        # excluded (it is at START and could not have reached an arm end yet);
        # it gets a free step to start navigating first.
        m = active & (self.phase == CHOICE) & (~just_entered_choice)
        at_L = m & (py == self._l_end[0]) & (px == self._l_end[1])
        at_R = m & (py == self._r_end[0]) & (px == self._r_end[1])
        reached = at_L | at_R
        if reached.any():
            reached_side = np.where(at_L, "L", "R")
            corr = reached & (reached_side == self.blocked)
            self.correct[reached] = corr[reached]
            task_r[corr] += self.cfg.test_reward * self.reward_scale
            int_r[reached] += self.cfg.completion_bonus
            self.done[reached] = True

        timeout = active & (~self.done) & (self.step_count >= self.cfg.max_episode_steps)
        if timeout.any():
            self.correct[timeout] = False
            self.done[timeout] = True

        info = {"correct": self.correct.copy(), "phase": self.phase.copy()}
        return task_r, int_r, self.done.copy(), info


class TMazeFreeNav:
    """B=1 facade with the original scalar interface. Wraps TMazeVecEnv so the
    analysis and trajectory code needs no changes."""
    def __init__(self, cfg, rng):
        self._env = TMazeVecEnv(cfg, rng, batch_size=1)
        self.cfg = cfg

    @property
    def current_delay(self): return self._env.current_delay
    @current_delay.setter
    def current_delay(self, v): self._env.current_delay = v
    def advance_delay(self): self._env.advance_delay()

    @property
    def reward_scale(self): return self._env.reward_scale
    @reward_scale.setter
    def reward_scale(self, v): self._env.reward_scale = v

    @property
    def pos(self): return (int(self._env.pos[0, 0]), int(self._env.pos[0, 1]))
    @property
    def phase(self): return _PHASE_NAME[int(self._env.phase[0])]
    @property
    def blocked(self): return str(self._env.blocked[0])
    @property
    def open_side(self): return str(self._env.open_side[0])

    @property
    def _phase_signal(self): return self._env._sig[0]
    @_phase_signal.setter
    def _phase_signal(self, v): self._env._sig[0] = v

    def reset(self):
        self._env.reset()
        return self.obs()

    def obs(self): return self._env.obs()[0]
    def obs_hab(self): return self._env.obs_hab()[0]

    def step(self, action):
        task_r, int_r, done, info = self._env.step(np.array([action]))
        terminal = bool(done[0])
        correct = bool(info["correct"][0]) if terminal else None
        scalar_info = {"agent_step": True, "correct": correct,
                       "phase": _PHASE_NAME[int(info["phase"][0])]}
        return float(task_r[0]), float(int_r[0]), terminal, scalar_info
