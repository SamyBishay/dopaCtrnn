import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config import Config
from environment import TMazeFreeNav

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def env(cfg, rng):
    return TMazeFreeNav(cfg, rng)


@pytest.fixture
def start_pos(env):
    """Expected (row, col) START cell, derived from the env's own geometry
    rather than hard-coded — the grid size is parametric (len_edge/difficulty),
    so START moves with the default Config()."""
    s = env._env.g["start"]
    return (int(s[0]), int(s[1]))


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_current_delay_equals_delay_start(self, env, cfg):
        assert env.current_delay == cfg.delay_start

    def test_delay_start_default(self, cfg):
        assert cfg.delay_start == 15

    def test_reward_scale(self, env):
        assert env.reward_scale == 1.0

    def test_initial_pos_is_start(self, env, start_pos):
        assert env.pos == start_pos

    def test_initial_phase(self, env):
        assert env.phase == "pre_sample"

    def test_initial_done_false(self, env):
        # TMazeFreeNav has no scalar `done` attribute; the underlying
        # vectorised env's [B=1] done array is the source of truth.
        assert bool(env._env.done[0]) is False

    def test_blocked_is_L_or_R(self, env):
        assert env.blocked in ("L", "R")

    def test_open_side_is_opposite_of_blocked(self, env):
        if env.blocked == "L":
            assert env.open_side == "R"
        else:
            assert env.open_side == "L"

    def test_blocked_and_open_side_differ(self, env):
        assert env.blocked != env.open_side


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_numpy_array(self, env):
        obs = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_reset_obs_shape(self, env, cfg):
        obs = env.reset()
        assert obs.shape == (cfg.obs_dim,)

    def test_reset_obs_dtype(self, env):
        obs = env.reset()
        assert obs.dtype == np.float32

    def test_reset_pos(self, env, start_pos):
        env.reset()
        assert env.pos == start_pos

    def test_reset_phase(self, env):
        env.reset()
        assert env.phase == "pre_sample"

    def test_reset_done(self, env):
        env.reset()
        assert bool(env._env.done[0]) is False

    def test_reset_step_count(self, env):
        env.reset()
        assert int(env._env.step_count[0]) == 0

    def test_reset_delay_idx(self, env):
        env.reset()
        assert int(env._env.delay_idx[0]) == 0

    def test_reset_obs_col_norm(self, env, start_pos):
        # col = start_pos[1], normalised by (w - 1)
        obs = env.reset()
        w = env._env.w
        assert obs[0] == pytest.approx(start_pos[1] / (w - 1), abs=1e-6)

    def test_reset_obs_row_norm(self, env, start_pos):
        # row = start_pos[0], normalised by (h - 1)
        obs = env.reset()
        h = env._env.h
        assert obs[1] == pytest.approx(start_pos[0] / (h - 1), abs=1e-6)

    def test_reset_obs_dim2_zero(self, env):
        obs = env.reset()
        assert obs[2] == pytest.approx(0.0, abs=1e-6)

    def test_reset_obs_sig_L_zero(self, env):
        obs = env.reset()
        assert obs[3] == pytest.approx(0.0, abs=1e-6)

    def test_reset_obs_sig_R_zero(self, env):
        obs = env.reset()
        assert obs[4] == pytest.approx(0.0, abs=1e-6)

    def test_reset_obs_sig_choice_zero(self, env):
        obs = env.reset()
        assert obs[5] == pytest.approx(0.0, abs=1e-6)

    def test_reset_twice_same_shape(self, env):
        obs1 = env.reset()
        obs2 = env.reset()
        assert obs1.shape == obs2.shape

    def test_reset_twice_same_dtype(self, env):
        obs1 = env.reset()
        obs2 = env.reset()
        assert obs1.dtype == obs2.dtype

    def test_reset_twice_first_two_dims_identical(self, env):
        # position-derived dims are deterministic regardless of blocked side
        obs1 = env.reset()
        obs2 = env.reset()
        assert obs1[0] == pytest.approx(obs2[0], abs=1e-6)
        assert obs1[1] == pytest.approx(obs2[1], abs=1e-6)


# ---------------------------------------------------------------------------
# advance_delay() tests
# ---------------------------------------------------------------------------

class TestAdvanceDelay:
    def test_advance_delay_increments_by_one(self, env, cfg):
        initial = env.current_delay
        env.advance_delay()
        assert env.current_delay == initial + 1

    def test_advance_delay_from_default_start(self, env, cfg):
        # default delay_start = 15
        env.advance_delay()
        assert env.current_delay == cfg.delay_start + 1

    def test_advance_delay_caps_at_delay_max(self, env, cfg):
        # Advance until we hit delay_max
        for _ in range(cfg.delay_max + 10):
            env.advance_delay()
        assert env.current_delay == cfg.delay_max

    def test_advance_delay_does_not_exceed_delay_max(self, env, cfg):
        for _ in range(cfg.delay_max + 20):
            env.advance_delay()
        assert env.current_delay <= cfg.delay_max

    def test_advance_delay_max_default(self, cfg):
        assert cfg.delay_max == 40

    def test_advance_delay_one_before_max_reaches_max(self, env, cfg):
        # Set current_delay to delay_max - 1 manually, then advance once
        env.current_delay = cfg.delay_max - 1
        env.advance_delay()
        assert env.current_delay == cfg.delay_max

    def test_advance_delay_at_max_stays_at_max(self, env, cfg):
        env.current_delay = cfg.delay_max
        env.advance_delay()
        assert env.current_delay == cfg.delay_max

    def test_advance_delay_multiple_steps_below_max(self, env, cfg):
        # Starting from delay_start=15, advance 3 times → should be delay_start+3
        steps = 3
        expected = min(cfg.delay_start + steps, cfg.delay_max)
        for _ in range(steps):
            env.advance_delay()
        assert env.current_delay == expected


# ---------------------------------------------------------------------------
# agent_controlled property tests
#
# TMazeFreeNav (the B=1 scalar facade) does not itself expose
# `agent_controlled` -- that property lives on the vectorised TMazeVecEnv it
# wraps. Test it there instead of on the facade.
# ---------------------------------------------------------------------------

class TestAgentControlled:
    def test_agent_controlled_is_true(self, env):
        assert env._env.agent_controlled == True

    def test_agent_controlled_after_reset(self, env):
        env.reset()
        assert env._env.agent_controlled == True

    def test_agent_controlled_is_bool(self, env):
        result = env._env.agent_controlled
        assert isinstance(result, bool)
