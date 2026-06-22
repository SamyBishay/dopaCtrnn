"""
Tests for TMazeFreeNav.obs()/obs_hab() — rewritten against the vectorised
environment (environment.py's TMazeVecEnv, wrapped by a B=1 facade).

The facade exposes pos/phase/blocked/open_side as READ-ONLY properties (no
setters) — there is no way to poke internal state directly as the old
(pre-vectorisation) tests did. These tests instead drive the env purely
through the public reset()/step(action) API and read back obs()/obs_hab(),
deriving expected positions from the env's own parametric geometry
(env._env.g) rather than hard-coding the old 5-wide/3-tall grid's numbers.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from config import Config
from environment import TMazeFreeNav

N, S, E, W, WAIT = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    c = Config()
    c.delay_start = 2  # keep delay phase short for tests that must pass through it
    return c


@pytest.fixture
def env(cfg):
    rng = np.random.default_rng(42)
    e = TMazeFreeNav(cfg, rng)
    e.current_delay = 2
    return e


def goto_junction(env):
    """Walk straight up the stem from START to JUNCTION (triggers
    pre_sample -> sample). Returns the last step's info dict."""
    info = None
    for _ in range(50):
        if env.phase != "pre_sample":
            break
        _, _, _, info = env.step(N)
    return info


def goto_open_arm_end(env):
    """From JUNCTION in 'sample' phase, walk toward the open arm until the
    sample -> delay transition fires. Returns the last step's info dict."""
    move = W if env.open_side == "L" else E
    info = None
    for _ in range(50):
        if env.phase != "sample":
            break
        _, _, _, info = env.step(move)
    return info


# ===========================================================================
# obs() — shape / dtype / static structure
# ===========================================================================

class TestObsShapeDtype:
    def test_shape(self, env, cfg):
        obs = env.obs()
        assert obs.shape == (cfg.obs_dim,)

    def test_dtype(self, env):
        assert env.obs().dtype == np.float32

    def test_obs2_always_zero(self, env):
        """obs[2] is reserved/unused — must always be 0.0."""
        assert env.obs()[2] == 0.0
        env.step(N)
        assert env.obs()[2] == 0.0

    def test_obs_position_dims_in_unit_interval(self, env):
        for _ in range(10):
            obs = env.obs()
            assert 0.0 <= obs[0] <= 1.0
            assert 0.0 <= obs[1] <= 1.0
            env.step(N)

    def test_phase_signal_slots_length_three(self, env):
        assert len(env.obs()[3:6]) == 3


# ===========================================================================
# obs() — position encoding, derived from the env's own geometry
# ===========================================================================

class TestObsPositionEncoding:
    def test_start_position(self, env):
        env.reset()
        s = env._env.g["start"]
        h, w = env._env.h, env._env.w
        obs = env.obs()
        assert obs[0] == pytest.approx(s[1] / (w - 1), abs=1e-6)
        assert obs[1] == pytest.approx(s[0] / (h - 1), abs=1e-6)

    def test_junction_position(self, env):
        env.reset()
        goto_junction(env)
        assert env.phase == "sample"
        j = env._env.g["junction"]
        h, w = env._env.h, env._env.w
        obs = env.obs()
        assert obs[0] == pytest.approx(j[1] / (w - 1), abs=1e-6)
        assert obs[1] == pytest.approx(j[0] / (h - 1), abs=1e-6)

    def test_position_changes_after_move(self, env):
        env.reset()
        obs0 = env.obs()
        env.step(N)
        obs1 = env.obs()
        assert not np.allclose(obs0[:2], obs1[:2])


# ===========================================================================
# obs() — phase signal
# ===========================================================================

class TestObsPhaseSignal:
    def test_no_phase_signal_at_start(self, env):
        env.reset()
        obs = env.obs()
        assert obs[3] == pytest.approx(0.0, abs=1e-6)
        assert obs[4] == pytest.approx(0.0, abs=1e-6)
        assert obs[5] == pytest.approx(0.0, abs=1e-6)

    def test_phase_signal_set_on_entering_sample(self, env):
        env.reset()
        goto_junction(env)
        assert env.phase == "sample"
        obs = env.obs()
        sig_val = env.cfg.sig_val
        if env.open_side == "L":
            np.testing.assert_allclose(obs[3:6], [sig_val, 0.0, 0.0], atol=1e-6)
        else:
            np.testing.assert_allclose(obs[3:6], [0.0, sig_val, 0.0], atol=1e-6)

    def test_phase_signal_set_on_entering_choice(self, env):
        env.reset()
        goto_junction(env)
        goto_open_arm_end(env)
        assert env.phase == "delay"
        sig_val = env.cfg.sig_val
        # advance through the (short, current_delay=2) delay window
        for _ in range(5):
            if env.phase == "choice":
                break
            env.step(WAIT)
        assert env.phase == "choice"
        obs = env.obs()
        np.testing.assert_allclose(obs[3:6], [0.0, 0.0, sig_val], atol=1e-6)


# ===========================================================================
# obs_hab() — position-free by default (E5 default: "position_free")
# ===========================================================================

class TestObsHabDefault:
    def test_shape(self, env, cfg):
        assert env.obs_hab().shape == (cfg.obs_dim_hab,)

    def test_no_position_information(self, env):
        """Position-free hab obs must not change with position (only the
        phase-signal slots can vary)."""
        env.reset()
        obs_hab_0 = env.obs_hab()
        env.step(N)
        obs_hab_1 = env.obs_hab()
        # slot 0 is the reserved/unused zero; hab obs carries no x/y at all
        assert obs_hab_0[0] == pytest.approx(0.0, abs=1e-6)
        assert obs_hab_1[0] == pytest.approx(0.0, abs=1e-6)

    def test_matches_sig_slots_of_full_obs(self, env):
        """obs_hab()[1:4] must equal obs()[3:6] (same phase-signal source)."""
        env.reset()
        goto_junction(env)
        obs = env.obs()
        obs_hab = env.obs_hab()
        np.testing.assert_allclose(obs_hab[1:4], obs[3:6], atol=1e-6)


# ===========================================================================
# obs_hab() — allocentric variant (E5)
# ===========================================================================

class TestObsHabAllocentric:
    def test_obs_dim_hab_equals_obs_dim(self):
        cfg = Config(habit_obs="allocentric")
        assert cfg.obs_dim_hab == cfg.obs_dim

    def test_obs_hab_equals_obs(self):
        cfg = Config(habit_obs="allocentric")
        rng = np.random.default_rng(1)
        env = TMazeFreeNav(cfg, rng)
        env.reset()
        np.testing.assert_allclose(env.obs_hab(), env.obs(), atol=1e-9)
        env.step(N)
        np.testing.assert_allclose(env.obs_hab(), env.obs(), atol=1e-9)
