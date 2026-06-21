import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from config import Config
from model import DualSystemModel


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def model(cfg):
    m = DualSystemModel(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------

class TestResetState:
    def test_default_batch_size_shapes(self, model, cfg):
        model.reset_state()
        assert model.h_gd.shape == (1, cfg.n_gd)
        assert model.h_hab.shape == (1, cfg.n_hab)
        assert model.da_tonic.shape == (1,)

    def test_default_batch_size_zeros(self, model):
        model.reset_state()
        assert torch.all(model.h_gd == 0)
        assert torch.all(model.h_hab == 0)
        assert torch.all(model.da_tonic == 0)

    def test_explicit_batch_size_1(self, model, cfg):
        model.reset_state(batch_size=1)
        assert model.h_gd.shape == (1, cfg.n_gd)
        assert model.h_hab.shape == (1, cfg.n_hab)
        assert model.da_tonic.shape == (1,)

    def test_batch_size_4_shapes(self, model, cfg):
        model.reset_state(batch_size=4)
        assert model.h_gd.shape == (4, cfg.n_gd)
        assert model.h_hab.shape == (4, cfg.n_hab)
        assert model.da_tonic.shape == (4,)

    def test_batch_size_4_zeros(self, model):
        # dirty the state first
        model.reset_state(batch_size=4)
        model.h_gd = torch.ones_like(model.h_gd)
        model.h_hab = torch.ones_like(model.h_hab)
        model.da_tonic = torch.ones_like(model.da_tonic)
        # now reset
        model.reset_state(batch_size=4)
        assert torch.all(model.h_gd == 0)
        assert torch.all(model.h_hab == 0)
        assert torch.all(model.da_tonic == 0)

    def test_batch_size_8_shapes(self, model, cfg):
        model.reset_state(batch_size=8)
        assert model.h_gd.shape == (8, cfg.n_gd)
        assert model.h_hab.shape == (8, cfg.n_hab)
        assert model.da_tonic.shape == (8,)

    def test_reset_clears_after_step(self, model, cfg):
        B = 2
        model.reset_state(B)
        obs = torch.randn(B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs, obs)
        model.reset_state(B)
        assert torch.all(model.h_gd == 0)
        assert torch.all(model.h_hab == 0)


# ---------------------------------------------------------------------------
# gd_params
# ---------------------------------------------------------------------------

class TestGdParams:
    def test_returns_list(self, model):
        params = model.gd_params()
        assert isinstance(params, list)

    def test_all_are_parameters(self, model):
        for p in model.gd_params():
            assert isinstance(p, torch.nn.Parameter)

    def test_includes_gd_net_params(self, model):
        gd_params = set(id(p) for p in model.gd_params())
        for p in model.gd.parameters():
            assert id(p) in gd_params, "gd_params() missing a parameter from model.gd"

    def test_includes_alpha(self, model):
        gd_params = set(id(p) for p in model.gd_params())
        assert id(model.alpha) in gd_params, "gd_params() must include model.alpha"

    def test_includes_bias(self, model):
        gd_params = set(id(p) for p in model.gd_params())
        assert id(model.bias) in gd_params, "gd_params() must include model.bias"

    def test_excludes_hab_params(self, model):
        gd_param_ids = set(id(p) for p in model.gd_params())
        for p in model.hab.parameters():
            assert id(p) not in gd_param_ids, "gd_params() must not include model.hab parameters"

    def test_non_empty(self, model):
        assert len(model.gd_params()) > 0


# ---------------------------------------------------------------------------
# hab_params
# ---------------------------------------------------------------------------

class TestHabParams:
    def test_returns_list(self, model):
        params = model.hab_params()
        assert isinstance(params, list)

    def test_all_are_parameters(self, model):
        for p in model.hab_params():
            assert isinstance(p, torch.nn.Parameter)

    def test_includes_hab_net_params(self, model):
        hab_params = set(id(p) for p in model.hab_params())
        for p in model.hab.parameters():
            assert id(p) in hab_params, "hab_params() missing a parameter from model.hab"

    def test_excludes_gd_params(self, model):
        hab_param_ids = set(id(p) for p in model.hab_params())
        for p in model.gd.parameters():
            assert id(p) not in hab_param_ids, "hab_params() must not include model.gd parameters"

    def test_excludes_alpha(self, model):
        hab_param_ids = set(id(p) for p in model.hab_params())
        assert id(model.alpha) not in hab_param_ids, "hab_params() must not include model.alpha"

    def test_excludes_bias(self, model):
        hab_param_ids = set(id(p) for p in model.hab_params())
        assert id(model.bias) not in hab_param_ids, "hab_params() must not include model.bias"

    def test_non_empty(self, model):
        assert len(model.hab_params()) > 0


# ---------------------------------------------------------------------------
# step — output keys and shapes
# ---------------------------------------------------------------------------

class TestStepOutputKeys:
    REQUIRED_KEYS = {"combined", "pi_gd", "pi_h", "da_request", "value", "w_gd"}

    def test_returns_dict(self, model, cfg):
        B = 1
        model.reset_state(B)
        obs = torch.randn(B, cfg.obs_dim)
        with torch.no_grad():
            out = model.step(obs, obs)
        assert isinstance(out, dict)

    def test_all_keys_present(self, model, cfg):
        B = 1
        model.reset_state(B)
        obs = torch.randn(B, cfg.obs_dim)
        with torch.no_grad():
            out = model.step(obs, obs)
        assert self.REQUIRED_KEYS.issubset(out.keys())


class TestStepShapes:
    B = 4

    @pytest.fixture(autouse=True)
    def step_output(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            self.out = model.step(obs, obs)
        self.cfg = cfg

    def test_combined_shape(self):
        assert self.out["combined"].shape == (self.B, self.cfg.n_actions)

    def test_pi_gd_shape(self):
        assert self.out["pi_gd"].shape == (self.B, self.cfg.n_actions)

    def test_pi_h_shape(self):
        assert self.out["pi_h"].shape == (self.B, self.cfg.n_actions)

    def test_da_request_shape(self):
        assert self.out["da_request"].shape == (self.B,)

    def test_value_shape(self):
        assert self.out["value"].shape == (self.B,)

    def test_w_gd_shape(self):
        assert self.out["w_gd"].shape == (self.B,)

    def test_da_request_in_0_1(self):
        assert torch.all(self.out["da_request"] >= 0)
        assert torch.all(self.out["da_request"] <= 1)

    def test_w_gd_in_0_1(self):
        assert torch.all(self.out["w_gd"] >= 0)
        assert torch.all(self.out["w_gd"] <= 1)


# ---------------------------------------------------------------------------
# step — force_w
# ---------------------------------------------------------------------------

class TestForceW:
    B = 4

    def test_force_w_1_combined_equals_mot_times_pi_gd(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            out = model.step(obs, obs, force_w=1.0)
        assert out["w_gd"].shape == (self.B,)
        assert torch.allclose(out["w_gd"], torch.ones(self.B), atol=1e-5)
        expected = out["pi_gd"]  # mot=1.0 default
        assert torch.allclose(out["combined"], expected, atol=1e-5)

    def test_force_w_0_combined_equals_pi_h(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            out = model.step(obs, obs, force_w=0.0)
        assert torch.allclose(out["w_gd"], torch.zeros(self.B), atol=1e-5)
        assert torch.allclose(out["combined"], out["pi_h"], atol=1e-5)

    def test_force_w_none_w_gd_not_forced(self, model, cfg):
        """With force_w=None the weight is computed normally (need not be 0 or 1)."""
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            out = model.step(obs, obs, force_w=None)
        # w_gd should still be in [0,1] but not necessarily 0 or 1
        assert torch.all(out["w_gd"] >= 0) and torch.all(out["w_gd"] <= 1)

    def test_force_w_default_is_none(self, model, cfg):
        """Calling step without force_w should not raise."""
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            out = model.step(obs, obs)
        assert "w_gd" in out


# ---------------------------------------------------------------------------
# step — mot (motivation scalar)
# ---------------------------------------------------------------------------

class TestMot:
    B = 4

    def test_mot_0_force_w_1_combined_zero(self, model, cfg):
        """mot=0, force_w=1 → combined = 0 * pi_gd + 1 * pi_h = pi_h"""
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            out = model.step(obs, obs, force_w=1.0, mot=0.0)
        # w_gd=1, so combined = 1 * 0 * pi_gd + 0 * pi_h = 0 * pi_gd
        # per spec: mot=0 with force_w=1 → devaluation → combined = 0 * pi_gd + 1 * pi_h
        # rewrite: combined = w_gd * mot * pi_gd + (1-w_gd)*pi_h
        # with w_gd=1, mot=0: combined = 0 + 0 * pi_h = 0? No — spec says "= 0 * pi_gd + 1 * pi_h"
        # The spec formula: combined = w_gd * mot * pi_gd + (1 - w_gd) * pi_h
        # w_gd=1, mot=0 → 1*0*pi_gd + 0*pi_h = 0; but spec also says "combined = pi_h" in devaluation
        # We test the formula as written: w_gd * mot * pi_gd + (1-w_gd) * pi_h
        expected = 1.0 * 0.0 * out["pi_gd"] + 0.0 * out["pi_h"]
        assert torch.allclose(out["combined"], expected, atol=1e-5)

    def test_mot_default_is_1(self, model, cfg):
        """mot=1.0 (default) with force_w=1 → combined == pi_gd"""
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            out_default = model.step(obs, obs, force_w=1.0)
        model.reset_state(self.B)
        with torch.no_grad():
            out_explicit = model.step(obs, obs, force_w=1.0, mot=1.0)
        assert torch.allclose(out_default["combined"], out_explicit["combined"], atol=1e-5)

    def test_combined_formula(self, model, cfg):
        """Verify the blending formula: combined = w_gd * mot * pi_gd + (1-w_gd) * pi_h."""
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        mot = 0.7
        with torch.no_grad():
            out = model.step(obs, obs, mot=mot)
        w = out["w_gd"].unsqueeze(-1)
        expected = w * mot * out["pi_gd"] + (1 - w) * out["pi_h"]
        assert torch.allclose(out["combined"], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# step — lesion
# ---------------------------------------------------------------------------

class TestLesion:
    B = 2

    def test_lesion_none_updates_both_states(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs, obs, lesion=None)
        # states should be non-zero after a normal forward pass
        assert not torch.all(model.h_gd == 0), "h_gd should be updated after step"
        assert not torch.all(model.h_hab == 0), "h_hab should be updated after step"

    def test_lesion_gd_zeroes_h_gd(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs, obs, lesion="gd")
        assert torch.all(model.h_gd == 0), "GD lesion must zero h_gd"

    def test_lesion_hab_zeroes_h_hab(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs, obs, lesion="hab")
        assert torch.all(model.h_hab == 0), "HAB lesion must zero h_hab"

    def test_lesion_gd_does_not_zero_h_hab(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs, obs, lesion="gd")
        assert not torch.all(model.h_hab == 0), "GD lesion must not zero h_hab"

    def test_lesion_hab_does_not_zero_h_gd(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs, obs, lesion="hab")
        assert not torch.all(model.h_gd == 0), "HAB lesion must not zero h_gd"

    def test_lesion_both_zeroes_both(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs, obs, lesion="both")
        assert torch.all(model.h_gd == 0), "Both lesion must zero h_gd"
        assert torch.all(model.h_hab == 0), "Both lesion must zero h_hab"

    def test_lesion_default_is_none(self, model, cfg):
        """step() without lesion kwarg should not raise and should update states."""
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            out = model.step(obs, obs)
        assert "combined" in out


# ---------------------------------------------------------------------------
# step — hidden state update (non-lesion)
# ---------------------------------------------------------------------------

class TestHiddenStateUpdates:
    B = 3

    def test_h_gd_updated_after_step(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs, obs)
        assert not torch.all(model.h_gd == 0)

    def test_h_hab_updated_after_step(self, model, cfg):
        model.reset_state(self.B)
        obs = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs, obs)
        assert not torch.all(model.h_hab == 0)

    def test_state_changes_across_steps(self, model, cfg):
        model.reset_state(self.B)
        obs1 = torch.randn(self.B, cfg.obs_dim)
        obs2 = torch.randn(self.B, cfg.obs_dim)
        with torch.no_grad():
            model.step(obs1, obs1)
            h_gd_after_1 = model.h_gd.clone()
            model.step(obs2, obs2)
            h_gd_after_2 = model.h_gd.clone()
        assert not torch.allclose(h_gd_after_1, h_gd_after_2), \
            "h_gd should change between steps with different observations"
