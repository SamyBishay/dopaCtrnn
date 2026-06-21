"""
Tests for rollout(), evaluate(), and eval_trajectories() in analysis.py.
Written purely from the function spec — no implementation reading.
"""
import sys
import os
import pytest
import numpy as np
import torch

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from environment import TMazeFreeNav
from model import DualSystemModel
from analysis import rollout, evaluate, eval_trajectories


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return Config()


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def env(cfg, rng):
    return TMazeFreeNav(cfg, rng)


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return DualSystemModel(cfg)


# ---------------------------------------------------------------------------
# rollout() — basic return shape and types
# ---------------------------------------------------------------------------

class TestRolloutBasic:
    def test_returns_dict(self, model, env, cfg):
        out = rollout(model, env, cfg)
        assert isinstance(out, dict)

    def test_required_keys_present(self, model, env, cfg):
        out = rollout(model, env, cfg)
        for key in ("correct", "da_mean", "w_mean", "delay_state", "blocked", "traj"):
            assert key in out, f"Missing key: {key}"

    def test_correct_is_bool(self, model, env, cfg):
        out = rollout(model, env, cfg)
        assert isinstance(out["correct"], bool)

    def test_da_mean_is_float(self, model, env, cfg):
        out = rollout(model, env, cfg)
        assert isinstance(out["da_mean"], float)

    def test_w_mean_is_float(self, model, env, cfg):
        out = rollout(model, env, cfg)
        assert isinstance(out["w_mean"], float)

    def test_blocked_is_L_or_R(self, model, env, cfg):
        out = rollout(model, env, cfg)
        assert out["blocked"] in ("L", "R")

    def test_traj_is_none_by_default(self, model, env, cfg):
        out = rollout(model, env, cfg)
        assert out["traj"] is None

    def test_delay_state_is_none_by_default(self, model, env, cfg):
        out = rollout(model, env, cfg)
        assert out["delay_state"] is None

    def test_w_mean_in_unit_interval(self, model, env, cfg):
        out = rollout(model, env, cfg)
        assert 0.0 <= out["w_mean"] <= 1.0

    def test_da_mean_finite(self, model, env, cfg):
        out = rollout(model, env, cfg)
        assert np.isfinite(out["da_mean"])


# ---------------------------------------------------------------------------
# rollout() — record=True
# ---------------------------------------------------------------------------

class TestRolloutRecord:
    @pytest.fixture(autouse=True)
    def run_rollout(self, model, env, cfg):
        self.out = rollout(model, env, cfg, record=True)

    def test_traj_not_none(self):
        assert self.out["traj"] is not None

    def test_traj_is_dict(self):
        assert isinstance(self.out["traj"], dict)

    def test_traj_has_required_keys(self):
        traj = self.out["traj"]
        for key in ("pos", "w", "a", "correct", "blocked"):
            assert key in traj, f"traj missing key: {key}"

    def test_traj_pos_is_list(self):
        assert isinstance(self.out["traj"]["pos"], list)

    def test_traj_pos_elements_are_two_element(self):
        pos = self.out["traj"]["pos"]
        assert len(pos) > 0
        for p in pos:
            assert len(p) == 2, f"pos entry {p} does not have 2 elements"

    def test_traj_pos_length_is_steps_plus_one(self):
        traj = self.out["traj"]
        # pos has length steps+1, a has length steps
        assert len(traj["pos"]) == len(traj["a"]) + 1

    def test_traj_w_length_equals_steps(self):
        traj = self.out["traj"]
        assert len(traj["w"]) == len(traj["a"])

    def test_traj_w_elements_are_floats(self):
        for v in self.out["traj"]["w"]:
            assert isinstance(v, float)

    def test_traj_correct_matches_top_level(self):
        assert self.out["traj"]["correct"] == self.out["correct"]

    def test_traj_blocked_matches_top_level(self):
        assert self.out["traj"]["blocked"] == self.out["blocked"]

    def test_traj_blocked_is_L_or_R(self):
        assert self.out["traj"]["blocked"] in ("L", "R")

    def test_traj_correct_is_bool(self):
        assert isinstance(self.out["traj"]["correct"], bool)


# ---------------------------------------------------------------------------
# rollout() — collect_delay=True
# ---------------------------------------------------------------------------

class TestRolloutCollectDelay:
    def test_delay_state_is_array_or_none(self, model, env, cfg):
        out = rollout(model, env, cfg, collect_delay=True)
        # It's either None (never reached delay) or a numpy array
        assert out["delay_state"] is None or isinstance(out["delay_state"], np.ndarray)

    def test_delay_state_array_is_finite(self, model, env, cfg):
        out = rollout(model, env, cfg, collect_delay=True)
        if out["delay_state"] is not None:
            assert np.all(np.isfinite(out["delay_state"]))


# ---------------------------------------------------------------------------
# rollout() — force_w and lesion parameters
# ---------------------------------------------------------------------------

class TestRolloutForceW:
    def test_force_w_zero_runs_without_error(self, model, env, cfg):
        out = rollout(model, env, cfg, force_w=0.0)
        assert isinstance(out["correct"], bool)

    def test_force_w_one_runs_without_error(self, model, env, cfg):
        out = rollout(model, env, cfg, force_w=1.0)
        assert isinstance(out["correct"], bool)

    def test_force_w_half_runs_without_error(self, model, env, cfg):
        out = rollout(model, env, cfg, force_w=0.5)
        assert isinstance(out["correct"], bool)


class TestRolloutLesion:
    @pytest.mark.parametrize("lesion", [None, "gd", "hab", "both"])
    def test_lesion_runs_without_error(self, model, env, cfg, lesion):
        out = rollout(model, env, cfg, lesion=lesion)
        assert isinstance(out["correct"], bool)
        assert out["blocked"] in ("L", "R")


# ---------------------------------------------------------------------------
# evaluate() — basic contract
# ---------------------------------------------------------------------------

class TestEvaluateBasic:
    N = 5

    @pytest.fixture(autouse=True)
    def run_evaluate(self, model, env, cfg):
        self.out = evaluate(model, env, cfg, n=self.N)

    def test_returns_dict(self):
        assert isinstance(self.out, dict)

    def test_required_keys(self):
        for key in ("acc", "k", "n", "da_mean", "w_mean"):
            assert key in self.out, f"Missing key: {key}"

    def test_n_equals_requested(self):
        assert self.out["n"] == self.N

    def test_k_is_int(self):
        assert isinstance(self.out["k"], int)

    def test_acc_is_float(self):
        assert isinstance(self.out["acc"], float)

    def test_acc_in_unit_interval(self):
        assert 0.0 <= self.out["acc"] <= 1.0

    def test_acc_equals_k_over_n(self):
        assert self.out["acc"] == pytest.approx(self.out["k"] / self.out["n"])

    def test_k_consistent_with_n(self):
        assert 0 <= self.out["k"] <= self.out["n"]

    def test_da_mean_finite(self):
        assert np.isfinite(self.out["da_mean"])

    def test_w_mean_finite(self):
        assert np.isfinite(self.out["w_mean"])

    def test_w_mean_in_unit_interval(self):
        assert 0.0 <= self.out["w_mean"] <= 1.0


# ---------------------------------------------------------------------------
# evaluate() — keyword argument forwarding
# ---------------------------------------------------------------------------

class TestEvaluateKwargs:
    N = 3

    def test_force_w_one_runs(self, model, env, cfg):
        out = evaluate(model, env, cfg, n=self.N, force_w=1.0)
        assert 0.0 <= out["acc"] <= 1.0

    def test_force_w_zero_runs(self, model, env, cfg):
        out = evaluate(model, env, cfg, n=self.N, force_w=0.0)
        assert 0.0 <= out["acc"] <= 1.0

    def test_lesion_gd_runs(self, model, env, cfg):
        out = evaluate(model, env, cfg, n=self.N, lesion="gd")
        assert out["n"] == self.N

    def test_lesion_hab_runs(self, model, env, cfg):
        out = evaluate(model, env, cfg, n=self.N, lesion="hab")
        assert out["n"] == self.N

    def test_lesion_both_runs(self, model, env, cfg):
        out = evaluate(model, env, cfg, n=self.N, lesion="both")
        assert out["n"] == self.N

    def test_mot_kwarg_forwarded(self, model, env, cfg):
        out = evaluate(model, env, cfg, n=self.N, mot=0.5)
        assert isinstance(out["acc"], float)


# ---------------------------------------------------------------------------
# evaluate() — n=1 edge case
# ---------------------------------------------------------------------------

class TestEvaluateEdgeCases:
    def test_n_equals_one(self, model, env, cfg):
        out = evaluate(model, env, cfg, n=1)
        assert out["n"] == 1
        assert out["k"] in (0, 1)
        assert out["acc"] in (0.0, 1.0)

    def test_acc_k_n_consistency_n1(self, model, env, cfg):
        out = evaluate(model, env, cfg, n=1)
        assert out["acc"] == pytest.approx(out["k"] / out["n"])


# ---------------------------------------------------------------------------
# eval_trajectories() — basic contract
# ---------------------------------------------------------------------------

class TestEvalTrajectoriesBasic:
    N = 3

    @pytest.fixture(autouse=True)
    def run_eval_traj(self, model, env, cfg):
        self.result = eval_trajectories(model, env, cfg, n=self.N)

    def test_returns_list(self):
        assert isinstance(self.result, list)

    def test_length_equals_n(self):
        assert len(self.result) == self.N

    def test_no_none_elements(self):
        for i, t in enumerate(self.result):
            assert t is not None, f"Element {i} is None"

    def test_each_element_is_dict(self):
        for i, t in enumerate(self.result):
            assert isinstance(t, dict), f"Element {i} is not a dict"

    def test_each_traj_has_required_keys(self):
        for i, t in enumerate(self.result):
            for key in ("pos", "w", "a", "correct", "blocked"):
                assert key in t, f"Traj {i} missing key: {key}"

    def test_each_traj_pos_has_entries(self):
        for i, t in enumerate(self.result):
            assert len(t["pos"]) > 0, f"Traj {i} has empty pos"

    def test_each_traj_blocked_valid(self):
        for i, t in enumerate(self.result):
            assert t["blocked"] in ("L", "R"), f"Traj {i} blocked={t['blocked']!r}"

    def test_each_traj_correct_is_bool(self):
        for i, t in enumerate(self.result):
            assert isinstance(t["correct"], bool), f"Traj {i} correct is not bool"

    def test_each_traj_pos_steps_consistent(self):
        for i, t in enumerate(self.result):
            assert len(t["pos"]) == len(t["a"]) + 1, (
                f"Traj {i}: pos length {len(t['pos'])} != steps {len(t['a'])} + 1"
            )

    def test_each_traj_w_steps_consistent(self):
        for i, t in enumerate(self.result):
            assert len(t["w"]) == len(t["a"]), (
                f"Traj {i}: w length {len(t['w'])} != steps {len(t['a'])}"
            )


# ---------------------------------------------------------------------------
# eval_trajectories() — greedy and kwargs forwarding
# ---------------------------------------------------------------------------

class TestEvalTrajectoriesKwargs:
    N = 3

    def test_greedy_false_runs(self, model, env, cfg):
        result = eval_trajectories(model, env, cfg, n=self.N, greedy=False)
        assert len(result) == self.N

    def test_force_w_kwarg_forwarded(self, model, env, cfg):
        result = eval_trajectories(model, env, cfg, n=self.N, force_w=1.0)
        assert len(result) == self.N
        assert all(t is not None for t in result)

    def test_lesion_kwarg_forwarded(self, model, env, cfg):
        result = eval_trajectories(model, env, cfg, n=self.N, lesion="gd")
        assert len(result) == self.N
        assert all(t is not None for t in result)
