"""
Comprehensive tests for TMazeFreeNav.step() — rewritten against the
vectorised environment (environment.py's TMazeVecEnv, B=1 facade).

The pre-vectorisation tests drove the env by directly assigning to
env.pos / env.phase / env.blocked etc., and called a scalar env._passable(
cell) helper. None of that exists anymore: TMazeFreeNav exposes pos / phase /
blocked / open_side as READ-ONLY properties, and passability is computed
internally on batched arrays (TMazeVecEnv._passable_target), not as a public
per-cell method. These tests instead drive the env purely through the public
reset()/step(action) API, using small `current_delay` overrides to keep the
delay phase short, and deriving expected geometry (START/JUNCTION/arm ends)
from env._env.g rather than hard-coded 5-wide/3-tall coordinates.

Maze layout (default Config(): len_edge=7, difficulty=2 -> h=6, w=9):
  start    = (4, 4)
  junction = (1, 4)   (top of the stem, arm row)
  l_end    = (1, 1)   r_end = (1, 7)
Actions: 0=N, 1=S, 2=E, 3=W, 4=WAIT
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
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed=42, delay_start=2):
    cfg = Config()
    cfg.delay_start = delay_start
    rng = np.random.default_rng(seed)
    env = TMazeFreeNav(cfg, rng)
    env.current_delay = delay_start
    return cfg, env


def goto_junction(env):
    """Walk straight up the stem from START to JUNCTION."""
    info = None
    for _ in range(50):
        if env.phase != "pre_sample":
            break
        _, _, _, info = env.step(N)
    return info


def goto_open_arm_end(env):
    """From JUNCTION in 'sample' phase, walk to the open arm's end."""
    move = W if env.open_side == "L" else E
    info = None
    for _ in range(50):
        if env.phase != "sample":
            break
        _, _, _, info = env.step(move)
    return info


def goto_choice(env):
    """Drain the (short) delay window via WAIT until 'choice' phase."""
    info = None
    for _ in range(50):
        if env.phase != "delay":
            break
        _, _, _, info = env.step(WAIT)
    return info


def goto_arm_end_in_choice(env, side):
    """In 'choice' phase the agent starts back at START (stem column); walk
    N to the junction/arm row first, then toward the named ('L'/'R') arm end."""
    move = W if side == "L" else E
    info = None
    # climb the stem to the arm row
    for _ in range(50):
        if env.pos[0] == env._env.g["arm_row"]:
            break
        _, _, done, info = env.step(N)
        if done:
            return info
    # walk along the arm row to the chosen end
    for _ in range(50):
        _, _, done, info = env.step(move)
        if done:
            break
    return info


# ---------------------------------------------------------------------------
# 1. Basic movement
# ---------------------------------------------------------------------------

class TestMovement:
    def test_valid_move_updates_position(self):
        cfg, env = make_env()
        start = env.pos
        env.step(N)
        assert env.pos != start

    def test_move_toward_junction_reaches_it(self):
        cfg, env = make_env()
        goto_junction(env)
        j = env._env.g["junction"]
        assert env.pos == (int(j[0]), int(j[1]))
        assert env.phase == "sample"

    def test_wall_move_position_unchanged(self):
        """Moving south from START (into a wall) keeps position."""
        cfg, env = make_env()
        old_pos = env.pos
        env.step(S)
        assert env.pos == old_pos

    def test_wait_position_unchanged(self):
        cfg, env = make_env()
        old_pos = env.pos
        env.step(WAIT)
        assert env.pos == old_pos

    def test_move_east_from_start_into_wall_unchanged(self):
        cfg, env = make_env()
        old_pos = env.pos
        env.step(E)
        assert env.pos == old_pos


# ---------------------------------------------------------------------------
# 2. Per-step costs
# ---------------------------------------------------------------------------

class TestPerStepCosts:
    def test_wait_applies_wait_cost_to_task_r(self):
        cfg, env = make_env()
        task_r, int_r, done, info = env.step(WAIT)
        assert task_r == pytest.approx(-cfg.wait_cost)

    def test_wait_applies_wait_cost_to_int_r(self):
        cfg, env = make_env()
        task_r, int_r, done, info = env.step(WAIT)
        assert int_r == pytest.approx(-cfg.wait_cost)

    def test_movement_applies_step_cost_to_task_r(self):
        cfg, env = make_env()
        task_r, int_r, done, info = env.step(N)   # valid move toward junction
        assert task_r == pytest.approx(-cfg.step_cost)

    def test_movement_applies_step_cost_to_int_r(self):
        cfg, env = make_env()
        task_r, int_r, done, info = env.step(N)
        assert int_r == pytest.approx(-cfg.step_cost)

    def test_wait_cost_default_value(self):
        cfg, _ = make_env()
        assert cfg.wait_cost == pytest.approx(0.02)

    def test_step_cost_default_value(self):
        cfg, _ = make_env()
        assert cfg.step_cost == pytest.approx(0.02)

    def test_no_cost_during_delay(self):
        """Delay phase holds the agent at START with no step cost (TUNL
        holding-box model)."""
        cfg, env = make_env()
        goto_junction(env)
        goto_open_arm_end(env)
        assert env.phase == "delay"
        task_r, int_r, done, info = env.step(WAIT)
        assert task_r == pytest.approx(0.0)
        assert int_r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. info dict
# ---------------------------------------------------------------------------

class TestInfoDict:
    def test_agent_step_always_true(self):
        cfg, env = make_env()
        _, _, _, info = env.step(WAIT)
        assert info.get("agent_step") is True

    def test_correct_none_during_non_terminal(self):
        cfg, env = make_env()
        _, _, done, info = env.step(WAIT)
        assert done is False
        assert info.get("correct") is None

    def test_phase_key_present(self):
        cfg, env = make_env()
        _, _, _, info = env.step(WAIT)
        assert "phase" in info


# ---------------------------------------------------------------------------
# 4. Phase transition: pre_sample -> sample
# ---------------------------------------------------------------------------

class TestPreSampleToSample:
    def test_reaching_junction_triggers_transition(self):
        cfg, env = make_env()
        info = goto_junction(env)
        assert info["phase"] == "sample"

    def test_junction_bonus_added_on_transition(self):
        cfg, env = make_env()
        # walk to one step before junction, then take the triggering step
        j = env._env.g["junction"]
        while env.pos != (int(j[0]) + 1, int(j[1])):
            env.step(N)
        task_r, _, _, info = env.step(N)
        assert info["phase"] == "sample"
        assert task_r == pytest.approx(cfg.junction_bonus - cfg.step_cost)

    def test_phase_signal_set_on_transition(self):
        cfg, env = make_env()
        goto_junction(env)
        obs = env.obs()
        sig_val = cfg.sig_val
        if env.open_side == "L":
            np.testing.assert_allclose(obs[3:6], [sig_val, 0.0, 0.0], atol=1e-6)
        else:
            np.testing.assert_allclose(obs[3:6], [0.0, sig_val, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Phase transition: sample -> delay
# ---------------------------------------------------------------------------

class TestSampleToDelay:
    def test_reaching_open_arm_end_triggers_transition(self):
        cfg, env = make_env()
        goto_junction(env)
        info = goto_open_arm_end(env)
        assert info["phase"] == "delay"

    def test_arm_end_bonus_given_on_sample_to_delay(self):
        cfg, env = make_env()
        goto_junction(env)
        move = W if env.open_side == "L" else E
        # step until just before the transition, then capture the
        # transitioning step's reward
        task_r = None
        for _ in range(50):
            if env.phase != "sample":
                break
            task_r, _, _, info = env.step(move)
        assert info["phase"] == "delay"
        assert task_r == pytest.approx(cfg.arm_end_bonus - cfg.step_cost)

    def test_delay_idx_resets_to_zero_on_transition(self):
        cfg, env = make_env()
        goto_junction(env)
        goto_open_arm_end(env)
        assert env.phase == "delay"
        assert int(env._env.delay_idx[0]) == 0

    def test_position_teleports_to_start_on_delay_entry(self):
        """TUNL protocol: entering delay teleports the agent back to START."""
        cfg, env = make_env()
        goto_junction(env)
        goto_open_arm_end(env)
        assert env.phase == "delay"
        start = env._env.g["start"]
        assert env.pos == (int(start[0]), int(start[1]))

    def test_blocked_arm_does_not_transition(self):
        """Walking toward the BLOCKED arm from the junction should not reach
        an arm end (the path is sealed) and should not enter delay."""
        cfg, env = make_env()
        goto_junction(env)
        blocked_move = E if env.open_side == "L" else W
        for _ in range(10):
            env.step(blocked_move)
        assert env.phase == "sample"


# ---------------------------------------------------------------------------
# 6. Blocked arm during sample phase
# ---------------------------------------------------------------------------

class TestBlockedArmDuringSample:
    def test_blocked_side_impassable_during_sample(self):
        cfg, env = make_env()
        goto_junction(env)
        blocked_move = W if env.blocked == "L" else E
        old_pos = env.pos
        env.step(blocked_move)
        assert env.pos == old_pos, "agent should not move into the blocked arm"

    def test_open_side_passable_during_sample(self):
        cfg, env = make_env()
        goto_junction(env)
        old_pos = env.pos
        open_move = W if env.open_side == "L" else E
        env.step(open_move)
        assert env.pos != old_pos, "agent should be able to move into the open arm"


# ---------------------------------------------------------------------------
# 7. Delay phase: delay_idx and signal decay
# ---------------------------------------------------------------------------

class TestDelayPhase:
    def test_delay_idx_increments_each_step(self):
        cfg, env = make_env(delay_start=10)
        goto_junction(env)
        goto_open_arm_end(env)
        assert env.phase == "delay"
        assert int(env._env.delay_idx[0]) == 0
        env.step(WAIT)
        assert int(env._env.delay_idx[0]) == 1

    def test_phase_signal_decays_each_delay_step(self):
        cfg, env = make_env(delay_start=10)
        goto_junction(env)
        goto_open_arm_end(env)
        assert env.phase == "delay"
        sig_before = env.obs()[3:6].copy()
        assert sig_before.sum() > 0  # carried over from the sample cue
        env.step(WAIT)
        sig_after = env.obs()[3:6]
        np.testing.assert_allclose(sig_after, sig_before * 0.1, atol=1e-6)

    def test_delay_to_choice_transition_after_current_delay_steps(self):
        cfg, env = make_env(delay_start=3)
        goto_junction(env)
        goto_open_arm_end(env)
        assert env.phase == "delay"
        info = goto_choice(env)
        assert info["phase"] == "choice"

    def test_choice_phase_signal_set_on_transition(self):
        cfg, env = make_env(delay_start=1)
        goto_junction(env)
        goto_open_arm_end(env)
        goto_choice(env)
        assert env.phase == "choice"
        obs = env.obs()
        np.testing.assert_allclose(obs[3:6], [0.0, 0.0, cfg.sig_val], atol=1e-6)

    def test_no_transition_before_delay_complete(self):
        cfg, env = make_env(delay_start=5)
        goto_junction(env)
        goto_open_arm_end(env)
        assert env.phase == "delay"
        for _ in range(4):
            _, _, _, info = env.step(WAIT)
            assert info["phase"] == "delay"


# ---------------------------------------------------------------------------
# 8. Choice phase: correct / wrong outcome
# ---------------------------------------------------------------------------

class TestChoicePhase:
    def test_correct_choice_reaches_blocked_arm(self):
        """Non-match rule: correct = go to the originally-BLOCKED arm."""
        cfg, env = make_env(delay_start=1)
        goto_junction(env)
        goto_open_arm_end(env)
        goto_choice(env)
        assert env.phase == "choice"
        blocked = env.blocked
        info = goto_arm_end_in_choice(env, blocked)
        assert info["correct"] is True

    def test_wrong_choice_reaches_open_arm(self):
        """Going to the originally-open (non-blocked) arm during choice is wrong."""
        cfg, env = make_env(delay_start=1)
        goto_junction(env)
        goto_open_arm_end(env)
        goto_choice(env)
        assert env.phase == "choice"
        open_side = env.open_side
        info = goto_arm_end_in_choice(env, open_side)
        assert info["correct"] is False

    def test_correct_choice_task_reward(self):
        cfg, env = make_env(delay_start=1)
        goto_junction(env)
        goto_open_arm_end(env)
        goto_choice(env)
        blocked = env.blocked
        # climb the stem to the arm row, then capture the reward of the
        # final (terminating) step toward the blocked (= correct) arm.
        move = W if blocked == "L" else E
        task_r = int_r = done = None
        for _ in range(50):
            if env.pos[0] == env._env.g["arm_row"]:
                break
            task_r, int_r, done, info = env.step(N)
            if done:
                break
        for _ in range(50):
            task_r, int_r, done, info = env.step(move)
            if done:
                break
        expected = cfg.test_reward * env.reward_scale - cfg.step_cost
        assert task_r == pytest.approx(expected)

    def test_correct_choice_completion_bonus_in_int_r(self):
        cfg, env = make_env(delay_start=1)
        goto_junction(env)
        goto_open_arm_end(env)
        goto_choice(env)
        blocked = env.blocked
        move = W if blocked == "L" else E
        task_r = int_r = done = None
        for _ in range(50):
            if env.pos[0] == env._env.g["arm_row"]:
                break
            task_r, int_r, done, info = env.step(N)
            if done:
                break
        for _ in range(50):
            task_r, int_r, done, info = env.step(move)
            if done:
                break
        assert int_r == pytest.approx(cfg.completion_bonus - cfg.step_cost)

    def test_not_done_if_not_at_arm_end_in_choice(self):
        cfg, env = make_env(delay_start=1)
        goto_junction(env)
        goto_open_arm_end(env)
        goto_choice(env)
        assert env.phase == "choice"
        _, _, done, info = env.step(WAIT)
        assert done is False
        assert info["correct"] is None

    def test_completion_bonus_default_value(self):
        cfg, _ = make_env()
        assert cfg.completion_bonus == pytest.approx(0.10)

    def test_test_reward_default_value(self):
        cfg, _ = make_env()
        assert cfg.test_reward == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_sets_done_true(self):
        cfg, env = make_env()
        done = False
        for _ in range(cfg.max_episode_steps + 5):
            _, _, done, info = env.step(WAIT)
            if done:
                break
        assert done is True
        assert info["correct"] is False  # never reached an arm end

    def test_no_timeout_before_max_steps(self):
        cfg, env = make_env()
        for _ in range(cfg.max_episode_steps - 2):
            _, _, done, info = env.step(WAIT)
            assert done is False

    def test_max_episode_steps_default_value(self):
        cfg, _ = make_env()
        assert cfg.max_episode_steps == 200


# ---------------------------------------------------------------------------
# 10. Bonus defaults
# ---------------------------------------------------------------------------

class TestBonusDefaults:
    def test_junction_bonus_default(self):
        cfg, _ = make_env()
        assert cfg.junction_bonus == pytest.approx(0.30)

    def test_arm_end_bonus_default(self):
        cfg, _ = make_env()
        assert cfg.arm_end_bonus == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# 11. Full episode smoke test
# ---------------------------------------------------------------------------

class TestFullEpisodeSmoke:
    def test_full_episode_runs_to_a_terminal_state(self):
        """Walk a complete episode through every phase via the public API and
        confirm it terminates with a sensible info dict, regardless of which
        side ends up blocked (randomised by the env's own rng)."""
        cfg, env = make_env(delay_start=2)

        info = goto_junction(env)
        assert info["phase"] == "sample"

        info = goto_open_arm_end(env)
        assert info["phase"] == "delay"

        info = goto_choice(env)
        assert info["phase"] == "choice"

        # Navigate deliberately to the correct (originally-blocked) arm.
        blocked = env.blocked
        info = goto_arm_end_in_choice(env, blocked)
        assert info["correct"] is True
