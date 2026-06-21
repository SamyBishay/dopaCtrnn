"""
Tests for h1_learning() and h2_handoff() in analysis.py.
Written from spec only — implementation files were not read.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from environment import TMazeFreeNav
from model import DualSystemModel
from analysis import h1_learning, h2_handoff


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_objects():
    cfg = Config()
    rng = np.random.default_rng(0)
    env = TMazeFreeNav(cfg, rng)
    model = DualSystemModel(cfg)
    return cfg, env, model


# ---------------------------------------------------------------------------
# h1_learning — structural / type checks
# ---------------------------------------------------------------------------

class TestH1LearningStructure:
    """Verify the shape and types of h1_learning's output dict."""

    @pytest.fixture(autouse=True)
    def run_h1(self, base_objects):
        cfg, env, model = base_objects
        cfg.final_trials = 20        # keep test fast
        self.out = h1_learning(model, env, cfg)

    def test_returns_dict(self):
        assert isinstance(self.out, dict)

    def test_required_keys_present(self):
        required = {"accuracy", "k", "n", "p_value", "ci95", "pass"}
        assert required.issubset(self.out.keys())

    def test_n_equals_final_trials(self, base_objects):
        cfg, _, _ = base_objects
        assert self.out["n"] == cfg.final_trials

    def test_k_is_int(self):
        assert isinstance(self.out["k"], (int, np.integer))

    def test_k_in_range(self):
        assert 0 <= self.out["k"] <= self.out["n"]

    def test_accuracy_equals_k_over_n(self):
        assert self.out["accuracy"] == pytest.approx(self.out["k"] / self.out["n"])

    def test_accuracy_in_unit_interval(self):
        assert 0.0 <= self.out["accuracy"] <= 1.0

    def test_p_value_in_unit_interval(self):
        assert 0.0 <= self.out["p_value"] <= 1.0

    def test_ci95_is_length_two(self):
        ci = self.out["ci95"]
        assert hasattr(ci, "__len__") and len(ci) == 2

    def test_ci95_values_in_unit_interval(self):
        lo, hi = self.out["ci95"]
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_ci95_lo_leq_hi(self):
        lo, hi = self.out["ci95"]
        assert lo <= hi

    def test_ci95_brackets_accuracy(self):
        lo, hi = self.out["ci95"]
        acc = self.out["accuracy"]
        # CI should bracket (or nearly bracket) the observed proportion
        assert lo <= acc + 1e-9
        assert hi >= acc - 1e-9

    def test_pass_is_bool(self):
        assert isinstance(self.out["pass"], (bool, np.bool_))


class TestH1LearningPassLogic:
    """Verify the pass criterion semantics for an untrained model."""

    def test_untrained_model_does_not_pass(self, base_objects):
        """An untrained model should have accuracy ~0.5 and should NOT pass."""
        cfg, env, _ = base_objects
        cfg.final_trials = 20
        # fresh model with a fresh rng so we are not affected by prior fixture state
        fresh_rng = np.random.default_rng(42)
        fresh_env = TMazeFreeNav(cfg, fresh_rng)
        fresh_model = DualSystemModel(cfg)
        out = h1_learning(fresh_model, fresh_env, cfg)
        # An untrained model should not satisfy acc>=0.8 AND p<1e-3 simultaneously
        assert out["pass"] is False or out["pass"] == False  # noqa: E712

    def test_pass_requires_high_accuracy_and_low_pvalue(self, base_objects):
        """If pass is True, accuracy must be >= 0.80 and p_value < 1e-3."""
        cfg, env, model = base_objects
        cfg.final_trials = 20
        out = h1_learning(model, env, cfg)
        if out["pass"]:
            assert out["accuracy"] >= 0.80
            assert out["p_value"] < 1e-3

    def test_pass_false_when_acc_below_threshold(self, base_objects):
        """If accuracy < 0.80, pass must be False."""
        cfg, env, model = base_objects
        cfg.final_trials = 20
        out = h1_learning(model, env, cfg)
        if out["accuracy"] < 0.80:
            assert not out["pass"]

    def test_pass_false_when_pvalue_too_high(self, base_objects):
        """If p_value >= 1e-3, pass must be False."""
        cfg, env, model = base_objects
        cfg.final_trials = 20
        out = h1_learning(model, env, cfg)
        if out["p_value"] >= 1e-3:
            assert not out["pass"]


# ---------------------------------------------------------------------------
# h2_handoff — synthetic log tests
# ---------------------------------------------------------------------------

# Synthetic logs defined at module level for clarity and reuse.
LOGS_PASS = {
    "episode":      [100, 200, 300, 400, 500],
    "hab_solo_acc": [0.4, 0.5, 0.85, 0.9, 0.95],
    # peak w_gd = 0.9, half-peak = 0.45
    # first drop below 0.45: index 3 (w=0.3) → episode 400
    # hab_onset: index 2 (0.85 >= 0.8) → episode 300
    "w_gd":         [0.8, 0.9, 0.85, 0.3, 0.2],
    "combined_acc": [0.6, 0.7, 0.85, 0.9, 0.9],
}

LOGS_FAIL_NO_HAB = {
    "episode":      [100, 200, 300],
    "hab_solo_acc": [0.4, 0.5, 0.6],   # never reaches 0.8
    "w_gd":         [0.8, 0.7, 0.6],
    "combined_acc": [0.6, 0.7, 0.8],
}


class TestH2HandoffStructure:
    """Verify the shape and types of h2_handoff's output dict."""

    @pytest.fixture(autouse=True)
    def run_h2_pass(self):
        self.out = h2_handoff(LOGS_PASS)

    def test_returns_dict(self):
        assert isinstance(self.out, dict)

    def test_required_keys_present(self):
        required = {
            "hab_onset_episode",
            "wgd_drop_onset_episode",
            "wgd_peak",
            "wgd_final",
            "combined_final",
            "pass",
        }
        assert required.issubset(self.out.keys())

    def test_pass_is_bool(self):
        assert isinstance(self.out["pass"], (bool, np.bool_))

    def test_wgd_peak_is_float_like(self):
        assert isinstance(self.out["wgd_peak"], (float, int, np.floating, np.integer))

    def test_wgd_final_is_float_like(self):
        assert isinstance(self.out["wgd_final"], (float, int, np.floating, np.integer))

    def test_combined_final_is_float_like(self):
        assert isinstance(self.out["combined_final"], (float, int, np.floating, np.integer))


class TestH2HandoffClearHandoff:
    """LOGS_PASS should produce a passing result with correct field values."""

    @pytest.fixture(autouse=True)
    def run_h2(self):
        self.out = h2_handoff(LOGS_PASS)

    def test_pass_is_true(self):
        assert self.out["pass"] is True or self.out["pass"] == True  # noqa: E712

    def test_hab_onset_episode(self):
        # First episode where hab_solo_acc >= 0.8 is index 2 → episode 300
        assert self.out["hab_onset_episode"] == 300

    def test_wgd_peak(self):
        # Maximum w_gd = 0.9
        assert self.out["wgd_peak"] == pytest.approx(0.9)

    def test_wgd_drop_onset_episode(self):
        # First w_gd < 0.5 * 0.9 = 0.45 is index 3 (w=0.3) → episode 400
        assert self.out["wgd_drop_onset_episode"] == 400

    def test_wgd_drop_after_hab_onset(self):
        assert self.out["wgd_drop_onset_episode"] >= self.out["hab_onset_episode"]

    def test_wgd_final(self):
        assert self.out["wgd_final"] == pytest.approx(0.2)

    def test_wgd_final_leq_half_peak(self):
        assert self.out["wgd_final"] <= 0.5 * self.out["wgd_peak"] + 1e-9

    def test_combined_final(self):
        assert self.out["combined_final"] == pytest.approx(0.9)

    def test_combined_final_geq_threshold(self):
        assert self.out["combined_final"] >= 0.8


class TestH2HandoffNoHabitualOnset:
    """LOGS_FAIL_NO_HAB: hab_solo_acc never reaches 0.8 → pass must be False."""

    @pytest.fixture(autouse=True)
    def run_h2(self):
        self.out = h2_handoff(LOGS_FAIL_NO_HAB)

    def test_pass_is_false(self):
        assert self.out["pass"] is False or self.out["pass"] == False  # noqa: E712

    def test_hab_onset_is_none(self):
        assert self.out["hab_onset_episode"] is None

    def test_wgd_peak_is_max(self):
        assert self.out["wgd_peak"] == pytest.approx(max(LOGS_FAIL_NO_HAB["w_gd"]))

    def test_wgd_final_is_last(self):
        assert self.out["wgd_final"] == pytest.approx(LOGS_FAIL_NO_HAB["w_gd"][-1])

    def test_combined_final_is_last(self):
        assert self.out["combined_final"] == pytest.approx(
            LOGS_FAIL_NO_HAB["combined_acc"][-1]
        )


class TestH2HandoffPassLogicInvariant:
    """If pass is True the three sub-conditions must all hold."""

    @pytest.mark.parametrize("logs", [LOGS_PASS, LOGS_FAIL_NO_HAB])
    def test_pass_implies_conditions(self, logs):
        out = h2_handoff(logs)
        if out["pass"]:
            assert out["hab_onset_episode"] is not None
            assert out["wgd_drop_onset_episode"] is not None
            assert out["wgd_drop_onset_episode"] >= out["hab_onset_episode"]
            assert out["wgd_final"] <= 0.5 * out["wgd_peak"] + 1e-9
            assert out["combined_final"] >= 0.8

    @pytest.mark.parametrize("logs", [LOGS_PASS, LOGS_FAIL_NO_HAB])
    def test_not_pass_implies_at_least_one_condition_fails(self, logs):
        out = h2_handoff(logs)
        if not out["pass"]:
            conditions = [
                out["hab_onset_episode"] is not None,
                out["wgd_drop_onset_episode"] is not None,
                (
                    out["hab_onset_episode"] is not None
                    and out["wgd_drop_onset_episode"] is not None
                    and out["wgd_drop_onset_episode"] >= out["hab_onset_episode"]
                ),
                out["wgd_final"] <= 0.5 * out["wgd_peak"] + 1e-9,
                out["combined_final"] >= 0.8,
            ]
            assert not all(conditions)
