import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from config import Config
from model import GDNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def net(cfg):
    return GDNet(cfg)


@pytest.fixture
def batch_inputs(cfg):
    B = 4
    x = torch.randn(B, cfg.obs_dim)
    h = torch.zeros(B, cfg.n_gd)
    da_tonic = torch.zeros(B)
    return B, x, h, da_tonic


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_pi_shape(self, net, cfg, batch_inputs):
        B, x, h, da_tonic = batch_inputs
        with torch.no_grad():
            pi, da_request, value, h_new, da_tonic_new = net.step(x, h, da_tonic)
        assert pi.shape == (B, cfg.n_actions), (
            f"Expected pi.shape == ({B}, {cfg.n_actions}), got {pi.shape}"
        )

    def test_da_request_shape(self, net, cfg, batch_inputs):
        B, x, h, da_tonic = batch_inputs
        with torch.no_grad():
            pi, da_request, value, h_new, da_tonic_new = net.step(x, h, da_tonic)
        assert da_request.shape == (B,), (
            f"Expected da_request.shape == ({B},), got {da_request.shape}"
        )

    def test_value_shape(self, net, cfg, batch_inputs):
        B, x, h, da_tonic = batch_inputs
        with torch.no_grad():
            pi, da_request, value, h_new, da_tonic_new = net.step(x, h, da_tonic)
        assert value.shape == (B,), (
            f"Expected value.shape == ({B},), got {value.shape}"
        )

    def test_h_new_shape(self, net, cfg, batch_inputs):
        B, x, h, da_tonic = batch_inputs
        with torch.no_grad():
            pi, da_request, value, h_new, da_tonic_new = net.step(x, h, da_tonic)
        assert h_new.shape == (B, cfg.n_gd), (
            f"Expected h_new.shape == ({B}, {cfg.n_gd}), got {h_new.shape}"
        )

    def test_da_tonic_new_shape(self, net, cfg, batch_inputs):
        B, x, h, da_tonic = batch_inputs
        with torch.no_grad():
            pi, da_request, value, h_new, da_tonic_new = net.step(x, h, da_tonic)
        assert da_tonic_new.shape == (B,), (
            f"Expected da_tonic_new.shape == ({B},), got {da_tonic_new.shape}"
        )

    def test_returns_five_outputs(self, net, batch_inputs):
        B, x, h, da_tonic = batch_inputs
        with torch.no_grad():
            outputs = net.step(x, h, da_tonic)
        assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"


# ---------------------------------------------------------------------------
# DA request range test
# ---------------------------------------------------------------------------

class TestDARequestRange:
    def test_da_request_in_zero_one(self, net, batch_inputs):
        B, x, h, da_tonic = batch_inputs
        with torch.no_grad():
            _, da_request, _, _, _ = net.step(x, h, da_tonic)
        assert (da_request > 0).all(), "da_request should be > 0 (sigmoid output)"
        assert (da_request < 1).all(), "da_request should be < 1 (sigmoid output)"

    def test_da_request_bounded_random_inputs(self, net, cfg):
        """Check range over multiple random batches."""
        torch.manual_seed(42)
        for _ in range(5):
            B = 8
            x = torch.randn(B, cfg.obs_dim)
            h = torch.randn(B, cfg.n_gd)
            da_tonic = torch.rand(B)
            with torch.no_grad():
                _, da_request, _, _, _ = net.step(x, h, da_tonic)
            assert (da_request > 0).all() and (da_request < 1).all(), (
                f"da_request out of (0,1): min={da_request.min():.4f}, max={da_request.max():.4f}"
            )


# ---------------------------------------------------------------------------
# Lesion mode tests
# ---------------------------------------------------------------------------

class TestLesionMode:
    def test_lesion_true_h_new_is_zeros(self, net, batch_inputs):
        B, x, h, da_tonic = batch_inputs
        with torch.no_grad():
            _, _, _, h_new, _ = net.step(x, h, da_tonic, lesion=True)
        assert torch.all(h_new == 0), "With lesion=True, h_new must be all zeros"

    def test_lesion_false_h_new_not_all_zeros(self, net, cfg):
        """After a forward pass from non-trivial input, h_new should not be all zeros."""
        torch.manual_seed(0)
        B = 4
        x = torch.randn(B, cfg.obs_dim)
        h = torch.zeros(B, cfg.n_gd)
        da_tonic = torch.zeros(B)
        with torch.no_grad():
            _, _, _, h_new, _ = net.step(x, h, da_tonic, lesion=False)
        assert not torch.all(h_new == 0), (
            "With lesion=False and random input, h_new should not be all zeros"
        )

    def test_lesion_other_outputs_still_computed(self, net, cfg, batch_inputs):
        """pi, da_request, value, da_tonic_new should be finite even with lesion=True."""
        B, x, h, da_tonic = batch_inputs
        with torch.no_grad():
            pi, da_request, value, h_new, da_tonic_new = net.step(x, h, da_tonic, lesion=True)
        assert torch.isfinite(pi).all(), "pi should be finite with lesion=True"
        assert torch.isfinite(da_request).all(), "da_request should be finite with lesion=True"
        assert torch.isfinite(value).all(), "value should be finite with lesion=True"
        assert torch.isfinite(da_tonic_new).all(), "da_tonic_new should be finite with lesion=True"
        assert pi.shape == (B, cfg.n_actions)
        assert da_request.shape == (B,)
        assert value.shape == (B,)


# ---------------------------------------------------------------------------
# Tonic DA update rule
# ---------------------------------------------------------------------------

class TestTonicDAUpdate:
    def test_tonic_update_formula(self, net, cfg, batch_inputs):
        """da_tonic_new = (1 - kappa) * da_tonic + kappa * da_request"""
        B, x, h, da_tonic = batch_inputs
        kappa = cfg.tonic_kappa  # 0.10
        with torch.no_grad():
            _, da_request, _, _, da_tonic_new = net.step(x, h, da_tonic)
        expected = (1 - kappa) * da_tonic + kappa * da_request
        assert torch.allclose(da_tonic_new, expected, atol=1e-6), (
            f"da_tonic_new does not follow expected formula.\n"
            f"  da_tonic_new={da_tonic_new}\n  expected={expected}"
        )

    def test_tonic_update_nonzero_da_tonic(self, net, cfg):
        """Same formula holds when da_tonic starts non-zero."""
        torch.manual_seed(7)
        B = 4
        x = torch.randn(B, cfg.obs_dim)
        h = torch.zeros(B, cfg.n_gd)
        da_tonic = torch.rand(B)  # non-zero starting tonic
        kappa = cfg.tonic_kappa
        with torch.no_grad():
            _, da_request, _, _, da_tonic_new = net.step(x, h, da_tonic)
        expected = (1 - kappa) * da_tonic + kappa * da_request
        assert torch.allclose(da_tonic_new, expected, atol=1e-6)

    def test_tonic_kappa_value(self, cfg):
        assert cfg.tonic_kappa == pytest.approx(0.10), (
            f"Expected tonic_kappa=0.10, got {cfg.tonic_kappa}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_outputs(self, net, cfg):
        torch.manual_seed(1)
        B = 4
        x = torch.randn(B, cfg.obs_dim)
        h = torch.zeros(B, cfg.n_gd)
        da_tonic = torch.zeros(B)

        with torch.no_grad():
            pi1, da1, v1, h1, dt1 = net.step(x, h, da_tonic)
            pi2, da2, v2, h2, dt2 = net.step(x, h, da_tonic)

        assert torch.allclose(pi1, pi2), "pi not deterministic"
        assert torch.allclose(da1, da2), "da_request not deterministic"
        assert torch.allclose(v1, v2), "value not deterministic"
        assert torch.allclose(h1, h2), "h_new not deterministic"
        assert torch.allclose(dt1, dt2), "da_tonic_new not deterministic"


# ---------------------------------------------------------------------------
# Batch consistency
# ---------------------------------------------------------------------------

class TestBatchConsistency:
    def test_single_vs_batch(self, net, cfg):
        """Processing one sample alone vs. in a batch should give the same result."""
        torch.manual_seed(99)
        B = 4
        x = torch.randn(B, cfg.obs_dim)
        h = torch.zeros(B, cfg.n_gd)
        da_tonic = torch.zeros(B)

        with torch.no_grad():
            pi_batch, da_batch, v_batch, h_batch, dt_batch = net.step(x, h, da_tonic)

        # Check each sample individually
        for i in range(B):
            x_i = x[i].unsqueeze(0)        # [1, obs_dim]
            h_i = h[i].unsqueeze(0)        # [1, n_gd]
            dt_i = da_tonic[i].unsqueeze(0)  # [1]
            with torch.no_grad():
                pi_i, da_i, v_i, h_i_new, dt_i_new = net.step(x_i, h_i, dt_i)

            assert torch.allclose(pi_batch[i], pi_i.squeeze(0), atol=1e-5), (
                f"pi mismatch at batch index {i}"
            )
            assert torch.allclose(da_batch[i], da_i.squeeze(0), atol=1e-5), (
                f"da_request mismatch at batch index {i}"
            )
            assert torch.allclose(v_batch[i], v_i.squeeze(0), atol=1e-5), (
                f"value mismatch at batch index {i}"
            )
            assert torch.allclose(h_batch[i], h_i_new.squeeze(0), atol=1e-5), (
                f"h_new mismatch at batch index {i}"
            )
            assert torch.allclose(dt_batch[i], dt_i_new.squeeze(0), atol=1e-5), (
                f"da_tonic_new mismatch at batch index {i}"
            )
