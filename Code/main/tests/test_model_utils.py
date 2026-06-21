import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import pytest
from model import _tau, _inv_softplus, _mixed_tau_init
from config import Config


# ---------------------------------------------------------------------------
# _tau tests
# ---------------------------------------------------------------------------

class TestTau:
    def test_always_greater_than_one_positive(self):
        param = torch.tensor(5.0)
        result = _tau(param)
        assert result.item() > 1.0

    def test_always_greater_than_one_zero(self):
        param = torch.tensor(0.0)
        result = _tau(param)
        assert result.item() > 1.0

    def test_always_greater_than_one_negative(self):
        param = torch.tensor(-10.0)
        result = _tau(param)
        assert result.item() > 1.0

    def test_always_greater_than_one_very_negative(self):
        # At param=-100, float32 softplus underflows to 0 → tau==1.0 exactly
        param = torch.tensor(-100.0)
        result = _tau(param)
        assert result.item() >= 1.0

    def test_approaches_one_from_above_as_param_very_negative(self):
        # softplus(-100) ≈ 0, so tau ≈ 1.0
        param = torch.tensor(-100.0)
        result = _tau(param)
        assert result.item() == pytest.approx(1.0, abs=1e-4)

    def test_increases_with_large_param(self):
        # As param → +∞, tau → +∞
        param_small = torch.tensor(0.0)
        param_large = torch.tensor(100.0)
        assert _tau(param_large).item() > _tau(param_small).item()

    def test_large_param_gives_large_tau(self):
        param = torch.tensor(100.0)
        result = _tau(param)
        # softplus(100) ≈ 100, so tau ≈ 101
        assert result.item() > 50.0

    def test_formula_matches_manual_computation(self):
        param = torch.tensor(1.0)
        expected = 1.0 + F.softplus(param).item()
        result = _tau(param)
        assert result.item() == pytest.approx(expected, rel=1e-6)

    def test_scalar_input_gives_scalar_output(self):
        param = torch.tensor(0.5)
        result = _tau(param)
        assert result.dim() == 0  # 0-dimensional = scalar tensor

    def test_vector_input(self):
        params = torch.tensor([-5.0, 0.0, 5.0])
        result = _tau(params)
        assert result.shape == (3,)
        assert (result > 1.0).all()


# ---------------------------------------------------------------------------
# _inv_softplus tests
# ---------------------------------------------------------------------------

class TestInvSoftplus:
    def test_returns_float(self):
        result = _inv_softplus(1.0)
        assert isinstance(result, float)

    def test_known_value(self):
        # _inv_softplus(1.0) = log(e - 1)
        import math
        expected = math.log(math.e - 1)
        result = _inv_softplus(1.0)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_known_value_approx(self):
        # log(e-1) ≈ 0.54132...; test_known_value already checks the exact formula
        import math
        result = _inv_softplus(1.0)
        assert result == pytest.approx(math.log(math.e - 1), rel=1e-5)

    def test_round_trip_y1(self):
        # softplus(inv_softplus(y)) ≈ y
        y = 1.0
        x = _inv_softplus(y)
        recovered = F.softplus(torch.tensor(x)).item()
        assert recovered == pytest.approx(y, rel=1e-5)

    def test_round_trip_y2(self):
        y = 2.0
        x = _inv_softplus(y)
        recovered = F.softplus(torch.tensor(x)).item()
        assert recovered == pytest.approx(y, rel=1e-5)

    def test_round_trip_y_large(self):
        y = 24.0
        x = _inv_softplus(y)
        recovered = F.softplus(torch.tensor(x)).item()
        assert recovered == pytest.approx(y, rel=1e-4)

    def test_round_trip_y_small(self):
        y = 0.1
        x = _inv_softplus(y)
        recovered = F.softplus(torch.tensor(x)).item()
        assert recovered == pytest.approx(y, rel=1e-4)

    def test_monotone(self):
        # inv_softplus should be monotonically increasing
        assert _inv_softplus(1.0) < _inv_softplus(2.0) < _inv_softplus(10.0)


# ---------------------------------------------------------------------------
# _mixed_tau_init tests
# ---------------------------------------------------------------------------

class TestMixedTauInit:
    def setup_method(self):
        self.cfg = Config()  # tau_fast=2.0, tau_slow=25.0

    def test_returns_tensor(self):
        result = _mixed_tau_init(4, self.cfg)
        assert isinstance(result, torch.Tensor)

    def test_shape_even(self):
        n = 8
        result = _mixed_tau_init(n, self.cfg)
        assert result.shape == (n,)

    def test_shape_small(self):
        result = _mixed_tau_init(2, self.cfg)
        assert result.shape == (2,)

    def test_1d(self):
        result = _mixed_tau_init(6, self.cfg)
        assert result.dim() == 1

    def test_first_half_tau_equals_tau_fast(self):
        n = 8
        result = _mixed_tau_init(n, self.cfg)
        half = n // 2
        first_half = result[:half]
        tau_values = _tau(first_half)
        expected = torch.full((half,), self.cfg.tau_fast)
        assert torch.allclose(tau_values, expected, atol=1e-5)

    def test_second_half_tau_equals_tau_slow(self):
        n = 8
        result = _mixed_tau_init(n, self.cfg)
        half = n // 2
        second_half = result[half:]
        tau_values = _tau(second_half)
        expected = torch.full((n - half,), self.cfg.tau_slow)
        assert torch.allclose(tau_values, expected, atol=1e-4)

    def test_first_half_raw_value(self):
        # First half values should be _inv_softplus(tau_fast - 1.0) = _inv_softplus(1.0)
        n = 6
        result = _mixed_tau_init(n, self.cfg)
        half = n // 2
        expected_raw = _inv_softplus(self.cfg.tau_fast - 1.0)
        for i in range(half):
            assert result[i].item() == pytest.approx(expected_raw, rel=1e-5)

    def test_second_half_raw_value(self):
        # Second half values should be _inv_softplus(tau_slow - 1.0) = _inv_softplus(24.0)
        n = 6
        result = _mixed_tau_init(n, self.cfg)
        half = n // 2
        expected_raw = _inv_softplus(self.cfg.tau_slow - 1.0)
        for i in range(half, n):
            assert result[i].item() == pytest.approx(expected_raw, rel=1e-5)

    def test_n_equals_2(self):
        result = _mixed_tau_init(2, self.cfg)
        assert result.shape == (2,)
        assert _tau(result[0]).item() == pytest.approx(self.cfg.tau_fast, rel=1e-5)
        assert _tau(result[1]).item() == pytest.approx(self.cfg.tau_slow, rel=1e-4)

    def test_n_equals_10(self):
        n = 10
        result = _mixed_tau_init(n, self.cfg)
        assert result.shape == (n,)
        half = n // 2
        # All fast-half taus ≈ tau_fast
        fast_taus = _tau(result[:half])
        slow_taus = _tau(result[half:])
        assert torch.allclose(fast_taus, torch.full((half,), self.cfg.tau_fast), atol=1e-5)
        assert torch.allclose(slow_taus, torch.full((n - half,), self.cfg.tau_slow), atol=1e-4)
