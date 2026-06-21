"""
Tests for h3_devaluation, h4_lesions, h5_reactivation, h6_attractor.

Focuses on output structure (keys, types, value ranges) rather than specific
accuracy values, since the model is untrained.
"""

import sys
import os
import copy

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from environment import TMazeFreeNav
from model import DualSystemModel
from analysis import h3_devaluation, h4_lesions, h5_reactivation, h6_attractor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup():
    cfg = Config()
    cfg.eval_trials = 5
    cfg.attractor_trials = 10
    rng = np.random.default_rng(0)
    env = TMazeFreeNav(cfg, rng)
    model = DualSystemModel(cfg)
    state = copy.deepcopy(model.state_dict())
    return cfg, env, model, state


# ---------------------------------------------------------------------------
# h3_devaluation
# ---------------------------------------------------------------------------

class TestH3Devaluation:
    def test_returns_dict(self, setup):
        cfg, env, model, state = setup
        out = h3_devaluation(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        assert isinstance(out, dict)

    def test_top_level_keys(self, setup):
        cfg, env, model, state = setup
        out = h3_devaluation(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        assert "learning" in out
        assert "maintenance" in out
        assert "dissociation_pass" in out

    def test_learning_subdict_keys(self, setup):
        cfg, env, model, state = setup
        out = h3_devaluation(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        for key in ("before", "after", "drop"):
            assert key in out["learning"], f"missing key '{key}' in learning sub-dict"

    def test_maintenance_subdict_keys(self, setup):
        cfg, env, model, state = setup
        out = h3_devaluation(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        for key in ("before", "after", "drop"):
            assert key in out["maintenance"], f"missing key '{key}' in maintenance sub-dict"

    def test_floats(self, setup):
        cfg, env, model, state = setup
        out = h3_devaluation(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        for phase in ("learning", "maintenance"):
            for key in ("before", "after", "drop"):
                val = out[phase][key]
                assert isinstance(val, (float, int, np.floating)), (
                    f"{phase}[{key!r}] should be a float, got {type(val)}"
                )

    def test_dissociation_pass_is_bool(self, setup):
        cfg, env, model, state = setup
        out = h3_devaluation(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        assert isinstance(out["dissociation_pass"], (bool, np.bool_))

    def test_drop_equals_before_minus_after(self, setup):
        cfg, env, model, state = setup
        out = h3_devaluation(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        for phase in ("learning", "maintenance"):
            before = out[phase]["before"]
            after = out[phase]["after"]
            drop = out[phase]["drop"]
            assert abs(drop - (before - after)) < 1e-6, (
                f"{phase}: drop ({drop}) != before ({before}) - after ({after})"
            )

    def test_before_after_in_unit_interval(self, setup):
        cfg, env, model, state = setup
        out = h3_devaluation(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        for phase in ("learning", "maintenance"):
            for key in ("before", "after"):
                val = float(out[phase][key])
                assert 0.0 <= val <= 1.0, (
                    f"{phase}[{key!r}] = {val} is outside [0, 1]"
                )

    def test_dissociation_pass_logic(self, setup):
        """dissociation_pass must match the stated rule."""
        cfg, env, model, state = setup
        out = h3_devaluation(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        expected = (
            float(out["learning"]["drop"]) > 0.15
            and float(out["maintenance"]["drop"]) < 0.10
        )
        assert bool(out["dissociation_pass"]) == expected


# ---------------------------------------------------------------------------
# h4_lesions
# ---------------------------------------------------------------------------

class TestH4Lesions:
    def test_returns_dict(self, setup):
        cfg, env, model, state = setup
        out = h4_lesions(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        assert isinstance(out, dict)

    def test_top_level_keys(self, setup):
        cfg, env, model, state = setup
        out = h4_lesions(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        assert "learning" in out
        assert "maintenance" in out

    def test_lesion_subdict_keys(self, setup):
        cfg, env, model, state = setup
        out = h4_lesions(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        for phase in ("learning", "maintenance"):
            for key in ("gd", "hab", "both", "intact"):
                assert key in out[phase], (
                    f"missing key '{key}' in {phase} sub-dict"
                )

    def test_values_in_unit_interval(self, setup):
        cfg, env, model, state = setup
        out = h4_lesions(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        for phase in ("learning", "maintenance"):
            for key in ("gd", "hab", "both", "intact"):
                val = float(out[phase][key])
                assert 0.0 <= val <= 1.0, (
                    f"{phase}[{key!r}] = {val} is outside [0, 1]"
                )

    def test_values_are_floats(self, setup):
        cfg, env, model, state = setup
        out = h4_lesions(model, state_learn=state, state_maint=state, env=env, cfg=cfg)
        for phase in ("learning", "maintenance"):
            for key in ("gd", "hab", "both", "intact"):
                val = out[phase][key]
                assert isinstance(val, (float, int, np.floating)), (
                    f"{phase}[{key!r}] should be a float, got {type(val)}"
                )


# ---------------------------------------------------------------------------
# h5_reactivation
# ---------------------------------------------------------------------------

class TestH5Reactivation:
    def test_returns_dict(self, setup):
        cfg, env, model, state = setup
        out = h5_reactivation(model, state_maint=state, env=env, cfg=cfg)
        assert isinstance(out, dict)

    def test_required_keys_present(self, setup):
        cfg, env, model, state = setup
        out = h5_reactivation(model, state_maint=state, env=env, cfg=cfg)
        required = (
            "da_request_intact",
            "da_request_hab_silenced",
            "da_request_delta",
            "acc_intact",
            "acc_hab_silenced",
            "deval_sensitivity_silenced",
            "pass",
            "note",
        )
        for key in required:
            assert key in out, f"missing key '{key}'"

    def test_da_request_values_are_floats(self, setup):
        cfg, env, model, state = setup
        out = h5_reactivation(model, state_maint=state, env=env, cfg=cfg)
        for key in ("da_request_intact", "da_request_hab_silenced", "da_request_delta"):
            val = out[key]
            assert isinstance(val, (float, int, np.floating)), (
                f"'{key}' should be a float, got {type(val)}"
            )

    def test_delta_equals_silenced_minus_intact(self, setup):
        cfg, env, model, state = setup
        out = h5_reactivation(model, state_maint=state, env=env, cfg=cfg)
        expected_delta = float(out["da_request_hab_silenced"]) - float(out["da_request_intact"])
        assert abs(float(out["da_request_delta"]) - expected_delta) < 1e-6

    def test_acc_values_in_unit_interval(self, setup):
        cfg, env, model, state = setup
        out = h5_reactivation(model, state_maint=state, env=env, cfg=cfg)
        for key in ("acc_intact", "acc_hab_silenced"):
            val = float(out[key])
            assert 0.0 <= val <= 1.0, f"'{key}' = {val} is outside [0, 1]"

    def test_deval_sensitivity_is_float(self, setup):
        cfg, env, model, state = setup
        out = h5_reactivation(model, state_maint=state, env=env, cfg=cfg)
        val = out["deval_sensitivity_silenced"]
        assert isinstance(val, (float, int, np.floating))

    def test_pass_is_bool(self, setup):
        cfg, env, model, state = setup
        out = h5_reactivation(model, state_maint=state, env=env, cfg=cfg)
        assert isinstance(out["pass"], (bool, np.bool_))

    def test_note_is_nonempty_string(self, setup):
        cfg, env, model, state = setup
        out = h5_reactivation(model, state_maint=state, env=env, cfg=cfg)
        assert isinstance(out["note"], str) and len(out["note"]) > 0

    def test_pass_logic(self, setup):
        """pass must match the documented rule."""
        cfg, env, model, state = setup
        out = h5_reactivation(model, state_maint=state, env=env, cfg=cfg)
        expected = (
            float(out["da_request_delta"]) > 0
            and float(out["acc_hab_silenced"]) >= 0.65
            and float(out["deval_sensitivity_silenced"]) > 0.15
        )
        assert bool(out["pass"]) == expected


# ---------------------------------------------------------------------------
# h6_attractor
# ---------------------------------------------------------------------------

class TestH6Attractor:
    def test_requires_sklearn(self):
        pytest.importorskip("sklearn", reason="sklearn not installed — skipping h6 tests")

    def test_returns_dict(self, setup):
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        assert isinstance(out, dict)

    def test_required_keys_present(self, setup):
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        for key in ("pca_coords", "labels", "decoder_acc", "decoder_sd",
                    "participation_ratio", "pass"):
            assert key in out, f"missing key '{key}'"

    def test_pca_coords_is_list(self, setup):
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        assert isinstance(out["pca_coords"], list)

    def test_labels_is_list_of_ints(self, setup):
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        labels = out["labels"]
        assert isinstance(labels, list)
        for lbl in labels:
            assert lbl in (0, 1), f"label {lbl} is not 0 or 1"

    def test_decoder_acc_in_unit_interval(self, setup):
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        val = float(out["decoder_acc"])
        assert 0.0 <= val <= 1.0, f"decoder_acc = {val} is outside [0, 1]"

    def test_decoder_sd_nonnegative(self, setup):
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        assert float(out["decoder_sd"]) >= 0.0

    def test_participation_ratio_is_float(self, setup):
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        val = out["participation_ratio"]
        assert isinstance(val, (float, int, np.floating))

    def test_pass_is_bool(self, setup):
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        assert isinstance(out["pass"], (bool, np.bool_))

    def test_pass_logic(self, setup):
        """pass must be True iff decoder_acc >= 0.90."""
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        expected = float(out["decoder_acc"]) >= 0.90
        assert bool(out["pass"]) == expected

    def test_too_few_states_branch(self, setup):
        """When too few delay states are collected, pca_coords==[], pass==False,
        and note contains 'Too few'. Only verifiable if cfg produces < 10 states."""
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        if out["pca_coords"] == []:
            assert out["pass"] is False or out["pass"] == False
            assert "note" in out
            assert "Too few" in out["note"]

    def test_note_when_present_is_string(self, setup):
        pytest.importorskip("sklearn")
        cfg, env, model, state = setup
        out = h6_attractor(model, state_maint=state, env=env, cfg=cfg)
        if "note" in out:
            assert isinstance(out["note"], str)
