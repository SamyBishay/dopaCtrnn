"""Tests for _t() and participation_ratio() utility functions in analysis.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from analysis import _t, participation_ratio


# ---------------------------------------------------------------------------
# _t() tests
# ---------------------------------------------------------------------------

class TestT:
    """Tests for the _t(x, dev) -> Tensor helper."""

    def test_numpy_array_values(self):
        result = _t(np.array([1.0, 2.0]), "cpu")
        expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_numpy_array_dtype_float32(self):
        result = _t(np.array([1.0, 2.0]), "cpu")
        assert result.dtype == torch.float32

    def test_python_list_int(self):
        result = _t([1, 2, 3], "cpu")
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_python_list_dtype_float32(self):
        result = _t([1, 2, 3], "cpu")
        assert result.dtype == torch.float32

    def test_scalar_float(self):
        result = _t(0.5, "cpu")
        assert torch.allclose(result, torch.tensor(0.5, dtype=torch.float32))

    def test_scalar_dtype_float32(self):
        result = _t(0.5, "cpu")
        assert result.dtype == torch.float32

    def test_dtype_forced_float32_from_float64(self):
        """Even a float64 numpy array should come out as float32."""
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = _t(arr, "cpu")
        assert result.dtype == torch.float32

    def test_dtype_forced_float32_from_int_array(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = _t(arr, "cpu")
        assert result.dtype == torch.float32

    def test_shape_preserved_1d(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _t(arr, "cpu")
        assert result.shape == torch.Size([3])

    def test_shape_preserved_2d(self):
        arr = np.zeros((4, 5), dtype=np.float32)
        result = _t(arr, "cpu")
        assert result.shape == torch.Size([4, 5])

    def test_device_is_cpu(self):
        result = _t(np.array([1.0]), "cpu")
        assert result.device.type == "cpu"

    def test_returns_tensor(self):
        result = _t([1.0], "cpu")
        assert isinstance(result, torch.Tensor)

    def test_matches_torch_as_tensor(self):
        """Should be equivalent to torch.as_tensor(x, dtype=float32, device=dev)."""
        x = np.array([3.0, 4.0, 5.0])
        result = _t(x, "cpu")
        expected = torch.as_tensor(x, dtype=torch.float32, device="cpu")
        assert torch.allclose(result, expected)
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape


# ---------------------------------------------------------------------------
# participation_ratio() tests
# ---------------------------------------------------------------------------

class TestParticipationRatio:
    """Tests for participation_ratio(X: np.ndarray) -> float."""

    def test_returns_float(self):
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
        result = participation_ratio(X)
        assert isinstance(result, float)

    def test_all_variance_in_one_dim(self):
        """All points on a single axis → PR ≈ 1.0."""
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
        result = participation_ratio(X)
        assert abs(result - 1.0) < 0.1, f"Expected PR ≈ 1.0, got {result}"

    def test_correlated_two_columns_gives_one(self):
        """Two columns that are identical (same axis) → PR ≈ 1.0."""
        vals = np.linspace(0, 1, 100)
        X = np.column_stack([vals, vals])
        result = participation_ratio(X)
        assert abs(result - 1.0) < 0.15, f"Expected PR ≈ 1.0, got {result}"

    def test_independent_dims_gives_two(self):
        """Two independent Gaussian dims → PR ≈ 2.0."""
        rng = np.random.default_rng(0)
        X = np.column_stack([rng.standard_normal(200), rng.standard_normal(200)])
        result = participation_ratio(X)
        assert abs(result - 2.0) < 0.3, f"Expected PR ≈ 2.0, got {result}"

    def test_degenerate_all_same_point(self):
        """All identical points → zero/near-zero PR (denominator guard 1e-12)."""
        X = np.ones((10, 3), dtype=float)
        result = participation_ratio(X)
        # With 1e-12 guard in denominator and 0 eigenvalues, numerator = 0
        assert result < 1.0, f"Expected near-zero PR for degenerate data, got {result}"

    def test_pr_at_least_one_for_nontrivial_data(self):
        """PR should be >= 1.0 for any non-degenerate dataset."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        result = participation_ratio(X)
        assert result >= 1.0, f"Expected PR >= 1.0, got {result}"

    def test_pr_bounded_by_dimensionality(self):
        """PR should not exceed the number of dimensions."""
        rng = np.random.default_rng(7)
        d = 4
        X = rng.standard_normal((100, d))
        result = participation_ratio(X)
        assert result <= d + 0.5, f"PR {result} exceeded d={d}"

    def test_three_independent_dims(self):
        """Three independent Gaussian dims → PR ≈ 3.0."""
        rng = np.random.default_rng(1)
        X = np.column_stack([rng.standard_normal(500) for _ in range(3)])
        result = participation_ratio(X)
        assert abs(result - 3.0) < 0.4, f"Expected PR ≈ 3.0, got {result}"

    def test_formula_manually(self):
        """Verify the formula directly on a known simple case."""
        # 4 points with all variance in dim 0
        X = np.array([[1.0, 0.0],
                      [2.0, 0.0],
                      [3.0, 0.0],
                      [4.0, 0.0]])
        N = len(X)
        X_c = X - X.mean(axis=0)
        cov = X_c.T @ X_c / (N - 1)
        evs = np.linalg.eigvalsh(cov)
        evs = np.clip(evs, 0, None)
        expected_pr = (evs.sum()) ** 2 / (evs @ evs + 1e-12)
        result = participation_ratio(X)
        assert abs(result - expected_pr) < 1e-6, (
            f"Expected {expected_pr}, got {result}"
        )
