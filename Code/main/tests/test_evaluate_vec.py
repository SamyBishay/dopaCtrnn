"""
Tests for evaluate_vec() in analysis.py.

Written from spec only — does not read the implementation.
"""

import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from environment import TMazeFreeNav
from model import DualSystemModel
from analysis import evaluate_vec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_objects():
    """Shared cfg / env / model to avoid rebuilding for every test."""
    cfg = Config()
    rng = np.random.default_rng(0)
    env = TMazeFreeNav(cfg, rng)
    model = DualSystemModel(cfg)
    return cfg, env, model


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cfg_with_batch(batch_size: int) -> Config:
    cfg = Config()
    cfg.batch_size = batch_size
    return cfg


# ---------------------------------------------------------------------------
# Return-type / key tests
# ---------------------------------------------------------------------------


class TestReturnShape:
    def test_returns_dict(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5)
        assert isinstance(out, dict)

    def test_required_keys_present(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5)
        for key in ("acc", "k", "n", "da_mean", "w_mean"):
            assert key in out, f"Missing key: {key}"

    def test_no_extra_keys(self, base_objects):
        """Result should contain exactly the five specified keys."""
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5)
        assert set(out.keys()) == {"acc", "k", "n", "da_mean", "w_mean"}


# ---------------------------------------------------------------------------
# Value-range / type tests
# ---------------------------------------------------------------------------


class TestValues:
    def test_n_exact(self, base_objects):
        cfg, env, model = base_objects
        for n in (5, 10):
            out = evaluate_vec(model, env, cfg, n=n)
            assert out["n"] == n, f"Expected n={n}, got {out['n']}"

    def test_k_integer(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=10)
        assert isinstance(out["k"], (int, np.integer))

    def test_k_in_range(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=10)
        assert 0 <= out["k"] <= 10

    def test_acc_float(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=10)
        assert isinstance(out["acc"], (float, np.floating))

    def test_acc_in_unit_interval(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=10)
        assert 0.0 <= out["acc"] <= 1.0

    def test_acc_equals_k_over_n(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=10)
        assert math.isclose(out["acc"], out["k"] / out["n"], rel_tol=1e-6)

    def test_all_values_finite(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=10)
        for key, val in out.items():
            assert math.isfinite(float(val)), f"Key '{key}' is not finite: {val}"


# ---------------------------------------------------------------------------
# Batch-size boundary tests
# ---------------------------------------------------------------------------


class TestBatchBoundaries:
    def test_n_less_than_default_batch(self):
        """n < cfg.batch_size (default 128) must still produce n trials."""
        cfg = Config()  # batch_size defaults to 128
        rng = np.random.default_rng(1)
        env = TMazeFreeNav(cfg, rng)
        model = DualSystemModel(cfg)
        out = evaluate_vec(model, env, cfg, n=5)
        assert out["n"] == 5

    def test_n_equals_batch_size(self):
        """n == cfg.batch_size: single full batch."""
        cfg = _cfg_with_batch(8)
        rng = np.random.default_rng(2)
        env = TMazeFreeNav(cfg, rng)
        model = DualSystemModel(cfg)
        out = evaluate_vec(model, env, cfg, n=8)
        assert out["n"] == 8
        assert 0 <= out["k"] <= 8

    def test_n_greater_than_batch_size(self):
        """n > cfg.batch_size: multiple batches must be combined correctly."""
        cfg = _cfg_with_batch(3)
        rng = np.random.default_rng(3)
        env = TMazeFreeNav(cfg, rng)
        model = DualSystemModel(cfg)
        out = evaluate_vec(model, env, cfg, n=10)
        assert out["n"] == 10
        assert 0 <= out["k"] <= 10
        assert math.isclose(out["acc"], out["k"] / out["n"], rel_tol=1e-6)

    def test_n_not_divisible_by_batch(self):
        """n=5, batch_size=3 → two batches (3 + 2) → must count exactly 5."""
        cfg = _cfg_with_batch(3)
        rng = np.random.default_rng(4)
        env = TMazeFreeNav(cfg, rng)
        model = DualSystemModel(cfg)
        out = evaluate_vec(model, env, cfg, n=5)
        assert out["n"] == 5

    def test_n_equals_one(self):
        """Edge-case: single trial."""
        cfg = Config()
        rng = np.random.default_rng(5)
        env = TMazeFreeNav(cfg, rng)
        model = DualSystemModel(cfg)
        out = evaluate_vec(model, env, cfg, n=1)
        assert out["n"] == 1
        assert out["k"] in (0, 1)
        assert out["acc"] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# force_w parameter tests
# ---------------------------------------------------------------------------


class TestForceW:
    def test_force_w_none_runs(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5, force_w=None)
        assert out["n"] == 5

    def test_force_w_zero_habitual_only(self, base_objects):
        """force_w=0.0 → pure habitual; must still return valid dict."""
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5, force_w=0.0)
        assert out["n"] == 5
        assert 0.0 <= out["acc"] <= 1.0
        # w_mean should be 0 (or very close) when weight is forced to 0
        assert math.isclose(float(out["w_mean"]), 0.0, abs_tol=1e-5)

    def test_force_w_one_goal_directed_only(self, base_objects):
        """force_w=1.0 → pure goal-directed; must still return valid dict."""
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5, force_w=1.0)
        assert out["n"] == 5
        assert 0.0 <= out["acc"] <= 1.0
        assert math.isclose(float(out["w_mean"]), 1.0, abs_tol=1e-5)

    def test_force_w_midpoint(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5, force_w=0.5)
        assert out["n"] == 5
        assert math.isfinite(float(out["acc"]))


# ---------------------------------------------------------------------------
# lesion parameter tests
# ---------------------------------------------------------------------------


class TestLesion:
    @pytest.mark.parametrize("lesion", ["gd", "hab", "both", None])
    def test_lesion_variants_run(self, base_objects, lesion):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5, lesion=lesion)
        assert out["n"] == 5
        assert 0.0 <= out["acc"] <= 1.0

    @pytest.mark.parametrize("lesion", ["gd", "hab", "both", None])
    def test_lesion_all_values_finite(self, base_objects, lesion):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5, lesion=lesion)
        for key, val in out.items():
            assert math.isfinite(float(val)), (
                f"lesion={lesion!r}: key '{key}' not finite: {val}"
            )


# ---------------------------------------------------------------------------
# env.current_delay propagation test
# ---------------------------------------------------------------------------


class TestDelayPropagation:
    def test_delay_zero_runs(self, base_objects):
        cfg, env, model = base_objects
        env.current_delay = 0
        out = evaluate_vec(model, env, cfg, n=5)
        assert out["n"] == 5
        assert math.isfinite(float(out["acc"]))

    def test_delay_nonzero_runs(self, base_objects):
        cfg, env, model = base_objects
        env.current_delay = 3
        out = evaluate_vec(model, env, cfg, n=5)
        assert out["n"] == 5
        assert math.isfinite(float(out["acc"]))

    def test_delay_change_does_not_corrupt_output(self, base_objects):
        """Switching delay between calls should not raise or corrupt keys."""
        cfg, env, model = base_objects
        env.current_delay = 0
        out0 = evaluate_vec(model, env, cfg, n=5)
        env.current_delay = 3
        out3 = evaluate_vec(model, env, cfg, n=5)
        for key in ("acc", "k", "n", "da_mean", "w_mean"):
            assert key in out0 and key in out3


# ---------------------------------------------------------------------------
# mot parameter test
# ---------------------------------------------------------------------------


class TestMot:
    def test_mot_default(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5)
        assert out["n"] == 5

    def test_mot_zero(self, base_objects):
        """mot=0.0 is an extreme but should not crash."""
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5, mot=0.0)
        assert out["n"] == 5

    def test_mot_explicit_one(self, base_objects):
        cfg, env, model = base_objects
        out = evaluate_vec(model, env, cfg, n=5, mot=1.0)
        assert out["n"] == 5
        assert 0.0 <= out["acc"] <= 1.0


# ---------------------------------------------------------------------------
# Determinism smoke-test
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self):
        """Two calls with the same seed should give identical results."""
        cfg = Config()

        rng1 = np.random.default_rng(42)
        env1 = TMazeFreeNav(cfg, rng1)
        import torch
        torch.manual_seed(0)
        model1 = DualSystemModel(cfg)
        out1 = evaluate_vec(model1, env1, cfg, n=10, force_w=1.0)

        rng2 = np.random.default_rng(42)
        env2 = TMazeFreeNav(cfg, rng2)
        torch.manual_seed(0)
        model2 = DualSystemModel(cfg)
        out2 = evaluate_vec(model2, env2, cfg, n=10, force_w=1.0)

        assert out1["n"] == out2["n"]
        assert out1["k"] == out2["k"]
