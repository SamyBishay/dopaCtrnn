"""
Comprehensive tests for TMazeFreeNav.step()

Spec reference: DNMTP T-maze with 4 phases: pre_sample → sample → delay → choice

Maze layout:
    (0,0) (0,1) (0,2) (0,3) (0,4)   L_END L_ARM1 JUNCTION R_ARM1 R_END
                (1,2)                 STEM
                (2,2)                 START

Actions: 0=N, 1=S, 2=E, 3=W, 4=WAIT
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config import Config
from environment import TMazeFreeNav


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed=42):
    cfg = Config()
    rng = np.random.default_rng(seed)
    env = TMazeFreeNav(cfg, rng)
    return cfg, env


def step(env, action):
    """Return (task_r, int_r, done, info) from env.step(action)."""
    return env.step(action)


# Positions (row, col)
START     = (2, 2)
STEM      = (1, 2)
JUNCTION  = (0, 2)
L_ARM1    = (0, 1)
L_END     = (0, 0)
R_ARM1    = (0, 3)
R_END     = (0, 4)

# Actions
N, S, E, W, WAIT = 0, 1, 2, 3, 4

_SIG_VAL = 0.25   # 1/(COLS-1) with COLS=5


# ---------------------------------------------------------------------------
# 1. Basic movement
# ---------------------------------------------------------------------------

class TestMovement:
    def test_valid_move_north_updates_position(self):
        cfg, env = make_env()
        env.pos = list(STEM)   # (1,2)
        step(env, N)            # → (0,2) JUNCTION  — may trigger pre_sample→sample
        # We just need position to have moved; JUNCTION is passable
        assert tuple(env.pos) == JUNCTION

    def test_valid_move_south_updates_position(self):
        cfg, env = make_env()
        env.pos = list(STEM)
        step(env, S)            # → (2,2) START
        assert tuple(env.pos) == START

    def test_valid_move_east_from_junction(self):
        cfg, env = make_env()
        env.phase = "delay"     # avoid phase-transition side-effects
        env.pos = list(JUNCTION)
        step(env, E)            # → (0,3) R_ARM1
        assert tuple(env.pos) == R_ARM1

    def test_valid_move_west_from_junction(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(JUNCTION)
        step(env, W)            # → (0,1) L_ARM1
        assert tuple(env.pos) == L_ARM1

    def test_wall_move_position_unchanged(self):
        """Moving into an impassable cell keeps position."""
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(START)   # (2,2)
        old_pos = list(env.pos)
        step(env, E)            # (2,3) is not passable → stay
        assert tuple(env.pos) == tuple(old_pos)

    def test_wait_position_unchanged(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(STEM)
        step(env, WAIT)
        assert tuple(env.pos) == STEM

    def test_move_out_of_bounds_north_from_top_row(self):
        """Moving north from top row (0,*) is a wall — position stays."""
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(JUNCTION)   # (0,2)
        step(env, N)               # row -1 → out of bounds / wall
        assert tuple(env.pos) == JUNCTION

    def test_move_into_wall_below_stem(self):
        """Row 3 doesn't exist; moving S from START stays."""
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(START)
        step(env, S)
        assert tuple(env.pos) == START


# ---------------------------------------------------------------------------
# 2. Per-step costs
# ---------------------------------------------------------------------------

class TestPerStepCosts:
    def test_wait_applies_wait_cost_to_task_r(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(STEM)
        task_r, int_r, done, info = step(env, WAIT)
        assert task_r == pytest.approx(-cfg.wait_cost)

    def test_wait_applies_wait_cost_to_int_r(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(STEM)
        task_r, int_r, done, info = step(env, WAIT)
        assert int_r == pytest.approx(-cfg.wait_cost)

    def test_movement_applies_step_cost_to_task_r(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(STEM)
        task_r, int_r, done, info = step(env, N)   # valid move → JUNCTION
        # step cost is the only contribution in delay phase (no bonus expected here
        # because delay phase doesn't react to JUNCTION)
        assert task_r == pytest.approx(-cfg.step_cost)

    def test_movement_applies_step_cost_to_int_r(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(STEM)
        task_r, int_r, done, info = step(env, N)
        assert int_r == pytest.approx(-cfg.step_cost)

    def test_wait_cost_default_value(self):
        cfg, _ = make_env()
        assert cfg.wait_cost == pytest.approx(0.02)

    def test_step_cost_default_value(self):
        cfg, _ = make_env()
        assert cfg.step_cost == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 3. info dict
# ---------------------------------------------------------------------------

class TestInfoDict:
    def test_agent_step_always_true(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(STEM)
        _, _, _, info = step(env, WAIT)
        assert info.get("agent_step") is True

    def test_correct_none_during_non_terminal(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(STEM)
        _, _, done, info = step(env, WAIT)
        assert done is False
        assert info.get("correct") is None

    def test_phase_key_present(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.pos = list(STEM)
        _, _, _, info = step(env, WAIT)
        assert "phase" in info


# ---------------------------------------------------------------------------
# 4. Phase transition: pre_sample → sample
# ---------------------------------------------------------------------------

class TestPreSampleToSample:
    def test_reaching_junction_triggers_transition(self):
        cfg, env = make_env()
        env.phase = "pre_sample"
        env.pos = list(STEM)       # one step north → JUNCTION
        _, _, _, info = step(env, N)
        assert info["phase"] == "sample"

    def test_junction_bonus_added_on_transition(self):
        cfg, env = make_env()
        env.phase = "pre_sample"
        env.pos = list(STEM)
        task_r, _, _, _ = step(env, N)
        # task_r = junction_bonus - step_cost
        assert task_r == pytest.approx(cfg.junction_bonus - cfg.step_cost)

    def test_phase_signal_set_open_L_on_transition(self):
        """When open_side=='L', phase signal obs[3:6] becomes [0.25, 0.0, 0.0]."""
        cfg, env = make_env()
        env.phase = "pre_sample"
        env.open_side = "L"
        env.pos = list(STEM)
        step(env, N)
        obs = env.obs()
        np.testing.assert_allclose(obs[3:6], [_SIG_VAL, 0.0, 0.0], atol=1e-6)

    def test_phase_signal_set_open_R_on_transition(self):
        """When open_side=='R', phase signal obs[3:6] becomes [0.0, 0.25, 0.0]."""
        cfg, env = make_env()
        env.phase = "pre_sample"
        env.open_side = "R"
        env.pos = list(STEM)
        step(env, N)
        obs = env.obs()
        np.testing.assert_allclose(obs[3:6], [0.0, _SIG_VAL, 0.0], atol=1e-6)

    def test_no_transition_if_not_at_junction(self):
        cfg, env = make_env()
        env.phase = "pre_sample"
        env.pos = list(START)
        _, _, _, info = step(env, N)   # → STEM, not JUNCTION
        assert info["phase"] == "pre_sample"


# ---------------------------------------------------------------------------
# 5. Phase transition: sample → delay
# ---------------------------------------------------------------------------

class TestSampleToDelay:
    def test_reaching_L_END_with_open_L_triggers_transition(self):
        cfg, env = make_env()
        env.phase = "sample"
        env.open_side = "L"
        env.blocked = "R"
        env.pos = list(L_ARM1)   # one step west → L_END
        _, _, _, info = step(env, W)
        assert info["phase"] == "delay"

    def test_reaching_R_END_with_open_R_triggers_transition(self):
        cfg, env = make_env()
        env.phase = "sample"
        env.open_side = "R"
        env.blocked = "L"
        env.pos = list(R_ARM1)   # one step east → R_END
        _, _, _, info = step(env, E)
        assert info["phase"] == "delay"

    def test_arm_end_bonus_given_on_sample_to_delay(self):
        cfg, env = make_env()
        env.phase = "sample"
        env.open_side = "L"
        env.blocked = "R"
        env.pos = list(L_ARM1)
        task_r, _, _, _ = step(env, W)
        assert task_r == pytest.approx(cfg.arm_end_bonus - cfg.step_cost)

    def test_delay_idx_resets_to_zero_on_transition(self):
        cfg, env = make_env()
        env.phase = "sample"
        env.open_side = "L"
        env.blocked = "R"
        env.pos = list(L_ARM1)
        # set delay_idx to something non-zero first
        env.delay_idx = 7
        step(env, W)
        assert env.delay_idx == 0

    def test_reaching_wrong_end_in_sample_does_not_transition(self):
        """Reaching R_END when open_side=='L' should not trigger sample→delay."""
        cfg, env = make_env()
        env.phase = "sample"
        env.open_side = "L"
        env.blocked = "R"
        # Place agent at R_ARM1; R cells should be blocked during sample
        # Moving east would hit blocked arm — position stays
        env.pos = list(JUNCTION)
        step(env, E)   # tries to enter R_ARM1 — blocked
        assert env.phase == "sample"


# ---------------------------------------------------------------------------
# 6. Blocked arm during sample phase
# ---------------------------------------------------------------------------

class TestBlockedArmDuringSample:
    def test_blocked_L_cells_impassable_during_sample(self):
        """When blocked=='L', cells (0,0) and (0,1) are impassable in sample."""
        cfg, env = make_env()
        env.phase = "sample"
        env.blocked = "L"
        env.open_side = "R"
        env.pos = list(JUNCTION)   # (0,2)
        step(env, W)               # tries to enter L_ARM1 (0,1) — should be blocked
        assert tuple(env.pos) == JUNCTION

    def test_blocked_R_cells_impassable_during_sample(self):
        """When blocked=='R', cells (0,3) and (0,4) are impassable in sample."""
        cfg, env = make_env()
        env.phase = "sample"
        env.blocked = "R"
        env.open_side = "L"
        env.pos = list(JUNCTION)
        step(env, E)               # tries to enter R_ARM1 (0,3) — should be blocked
        assert tuple(env.pos) == JUNCTION

    def test_open_L_arm_passable_during_sample(self):
        """When blocked=='R', L_ARM1 is passable."""
        cfg, env = make_env()
        env.phase = "sample"
        env.blocked = "R"
        env.open_side = "L"
        env.pos = list(JUNCTION)
        step(env, W)               # → L_ARM1 (0,1)
        assert tuple(env.pos) == L_ARM1

    def test_open_R_arm_passable_during_sample(self):
        """When blocked=='L', R_ARM1 is passable."""
        cfg, env = make_env()
        env.phase = "sample"
        env.blocked = "L"
        env.open_side = "R"
        env.pos = list(JUNCTION)
        step(env, E)               # → R_ARM1 (0,3)
        assert tuple(env.pos) == R_ARM1


# ---------------------------------------------------------------------------
# 7. Delay phase: delay_idx and signal decay
# ---------------------------------------------------------------------------

class TestDelayPhase:
    def test_delay_idx_increments_each_step(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.delay_idx = 0
        env.current_delay = 10   # ensure no transition yet
        env.pos = list(STEM)
        step(env, WAIT)
        assert env.delay_idx == 1

    def test_phase_signal_decays_each_delay_step(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.delay_idx = 0
        env.current_delay = 10
        env.pos = list(STEM)
        # Set a known non-zero phase signal
        env._phase_signal = np.array([0.25, 0.0, 0.0], dtype=float)
        step(env, WAIT)
        np.testing.assert_allclose(env._phase_signal[:2], [0.025, 0.0], atol=1e-9)

    def test_delay_to_choice_transition_after_current_delay_steps(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.current_delay = 3
        env.delay_idx = 2   # one more step → delay_idx=3 >= current_delay=3
        env.pos = list(STEM)
        _, _, _, info = step(env, WAIT)
        assert info["phase"] == "choice"

    def test_choice_phase_signal_set_on_transition(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.current_delay = 1
        env.delay_idx = 0
        env.pos = list(STEM)
        step(env, WAIT)   # delay_idx becomes 1 >= current_delay=1 → choice
        obs = env.obs()
        np.testing.assert_allclose(obs[3:6], [0.0, 0.0, _SIG_VAL], atol=1e-6)

    def test_no_transition_before_delay_complete(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.current_delay = 5
        env.delay_idx = 3   # after step: 4 < 5 → stays in delay
        env.pos = list(STEM)
        _, _, _, info = step(env, WAIT)
        assert info["phase"] == "delay"

    def test_delay_default_current_delay_is_5(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.delay_idx = 0
        env.pos = list(STEM)
        # Take 4 steps — should still be in delay
        for _ in range(4):
            env.pos = list(STEM)  # keep in place
            _, _, _, info = step(env, WAIT)
        assert info["phase"] == "delay"
        # 5th step → choice
        env.pos = list(STEM)
        _, _, _, info = step(env, WAIT)
        assert info["phase"] == "choice"


# ---------------------------------------------------------------------------
# 8. Choice phase: correct / wrong outcome
# ---------------------------------------------------------------------------

class TestChoicePhase:
    def _setup_choice(self, env, blocked):
        env.phase = "choice"
        env.blocked = blocked
        env.open_side = "R" if blocked == "L" else "L"

    def test_correct_choice_L_blocked_go_L_END(self):
        """Non-match rule: blocked=='L' → correct if agent reaches L_END."""
        cfg, env = make_env()
        self._setup_choice(env, "L")
        env.pos = list(L_ARM1)
        task_r, int_r, done, info = step(env, W)   # → L_END
        assert info["correct"] is True
        assert done is True

    def test_correct_choice_R_blocked_go_R_END(self):
        """Non-match rule: blocked=='R' → correct if agent reaches R_END."""
        cfg, env = make_env()
        self._setup_choice(env, "R")
        env.pos = list(R_ARM1)
        task_r, int_r, done, info = step(env, E)   # → R_END
        assert info["correct"] is True
        assert done is True

    def test_wrong_choice_L_blocked_go_R_END(self):
        """Non-match rule: blocked=='L' → wrong if agent goes to R_END."""
        cfg, env = make_env()
        self._setup_choice(env, "L")
        env.pos = list(R_ARM1)
        task_r, int_r, done, info = step(env, E)   # → R_END (wrong arm)
        assert info["correct"] is False
        assert done is True

    def test_wrong_choice_R_blocked_go_L_END(self):
        """Non-match rule: blocked=='R' → wrong if agent goes to L_END."""
        cfg, env = make_env()
        self._setup_choice(env, "R")
        env.pos = list(L_ARM1)
        task_r, int_r, done, info = step(env, W)   # → L_END (wrong arm)
        assert info["correct"] is False
        assert done is True

    def test_correct_choice_task_reward(self):
        cfg, env = make_env()
        self._setup_choice(env, "L")
        env.pos = list(L_ARM1)
        task_r, _, _, info = step(env, W)
        # task_r = test_reward * reward_scale - step_cost
        expected = cfg.test_reward * env.reward_scale - cfg.step_cost
        assert task_r == pytest.approx(expected)

    def test_correct_choice_completion_bonus_in_int_r(self):
        cfg, env = make_env()
        self._setup_choice(env, "L")
        env.pos = list(L_ARM1)
        _, int_r, _, info = step(env, W)
        # int_r on correct: completion_bonus - step_cost
        assert int_r == pytest.approx(cfg.completion_bonus - cfg.step_cost)

    def test_wrong_choice_completion_bonus_in_int_r(self):
        """Even wrong choices yield completion_bonus in int_r."""
        cfg, env = make_env()
        self._setup_choice(env, "L")
        env.pos = list(R_ARM1)
        _, int_r, _, info = step(env, E)   # wrong arm
        assert int_r == pytest.approx(cfg.completion_bonus - cfg.step_cost)

    def test_wrong_choice_no_test_reward_in_task_r(self):
        cfg, env = make_env()
        self._setup_choice(env, "L")
        env.pos = list(R_ARM1)
        task_r, _, _, _ = step(env, E)   # wrong arm
        # task_r = 0 (no test_reward) - step_cost
        assert task_r == pytest.approx(-cfg.step_cost)

    def test_not_done_if_not_at_arm_end_in_choice(self):
        cfg, env = make_env()
        self._setup_choice(env, "L")
        env.pos = list(STEM)
        _, _, done, info = step(env, N)   # → JUNCTION, not an arm end
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
        env.phase = "delay"
        env.current_delay = 999   # prevent delay→choice transition
        env.step_count = cfg.max_episode_steps - 1
        env.pos = list(STEM)
        _, _, done, info = step(env, WAIT)
        assert done is True

    def test_timeout_correct_is_false(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.current_delay = 999
        env.step_count = cfg.max_episode_steps - 1
        env.pos = list(STEM)
        _, _, _, info = step(env, WAIT)
        assert info["correct"] is False

    def test_no_timeout_before_max_steps(self):
        cfg, env = make_env()
        env.phase = "delay"
        env.current_delay = 999
        env.step_count = cfg.max_episode_steps - 2
        env.pos = list(STEM)
        _, _, done, _ = step(env, WAIT)
        assert done is False

    def test_max_episode_steps_default_value(self):
        cfg, _ = make_env()
        assert cfg.max_episode_steps == 80


# ---------------------------------------------------------------------------
# 10. Junction bonus default
# ---------------------------------------------------------------------------

class TestBonusDefaults:
    def test_junction_bonus_default(self):
        cfg, _ = make_env()
        assert cfg.junction_bonus == pytest.approx(0.30)

    def test_arm_end_bonus_default(self):
        cfg, _ = make_env()
        assert cfg.arm_end_bonus == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# 11. Full episode smoke test (open_side = 'L', blocked = 'R')
# ---------------------------------------------------------------------------

class TestFullEpisodeSmoke:
    def test_full_episode_L_open_correct(self):
        """
        Walk the agent through a complete episode manually:
          START → STEM → JUNCTION  (pre_sample→sample)
          → L_ARM1 → L_END         (sample→delay)
          5 WAITs                   (delay→choice)
          JUNCTION → L_ARM1 → L_END (choice, correct because blocked=='R'... wait)

        Non-match rule: correct = went to BLOCKED arm.
        If open_side=='L', blocked=='R', correct arm is R_END.
        We'll go L_END for the wrong case to verify correct=False.
        """
        cfg, env = make_env()
        # Override so we control open_side
        env.phase = "pre_sample"
        env.open_side = "L"
        env.blocked = "R"
        env.pos = list(START)

        # START → STEM
        _, _, done, info = step(env, N)
        assert not done
        assert info["phase"] == "pre_sample"

        # STEM → JUNCTION  (pre_sample → sample)
        _, _, done, info = step(env, N)
        assert not done
        assert info["phase"] == "sample"

        # JUNCTION → L_ARM1
        _, _, done, info = step(env, W)
        assert not done

        # L_ARM1 → L_END  (sample → delay)
        task_r, _, done, info = step(env, W)
        assert not done
        assert info["phase"] == "delay"
        assert task_r == pytest.approx(cfg.arm_end_bonus - cfg.step_cost)

        # 5 delay steps
        for i in range(4):
            env.pos = list(STEM)
            _, _, done, info = step(env, WAIT)
            assert not done, f"should not be done after {i+1} delay WAITs"

        env.pos = list(STEM)
        _, _, done, info = step(env, WAIT)
        assert info["phase"] == "choice"

        # Navigate to correct arm (blocked='R' → R_END is correct)
        # First go to junction, then R_ARM1, then R_END
        env.pos = list(JUNCTION)
        _, _, done, info = step(env, E)   # → R_ARM1
        assert not done

        env.pos = list(R_ARM1)
        task_r, int_r, done, info = step(env, E)   # → R_END (correct!)
        assert done is True
        assert info["correct"] is True
        assert task_r == pytest.approx(cfg.test_reward * env.reward_scale - cfg.step_cost)
        assert int_r == pytest.approx(cfg.completion_bonus - cfg.step_cost)
