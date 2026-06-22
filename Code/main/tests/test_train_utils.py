"""
Tests for discounted_batch() and _physical_cores() in train.py.

discounted_batch() replaced the old per-episode discounted() helper in the
vectorisation rewrite: it operates on [T, B] reward/valid tensors instead of
a flat per-episode rewards list, so every B>1 episodes are discounted in one
shot instead of with a Python loop per episode. `discounted()` itself is
gone — there is no longer a single-episode entry point, just the B=1 case of
the batched one. The wrapper below recovers the old (rewards_list, gamma,
dev) -> Tensor[T] signature exactly as a thin adapter over the batched
function (reshape to [T, 1], an all-True valid mask, squeeze back to [T]),
so the original test cases (written against the per-episode semantics) still
exercise the same recursion with no behavioural change.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from train import discounted_batch, _physical_cores


def discounted(rewards_list, gamma, dev):
    """B=1 adapter over discounted_batch(), matching the old discounted()
    signature: a flat per-episode rewards list -> a [T] tensor of returns."""
    T = len(rewards_list)
    r = torch.tensor(rewards_list, dtype=torch.float32, device=dev).reshape(T, 1)
    valid = torch.ones(T, 1, dtype=torch.bool, device=dev)
    out = discounted_batch(r, gamma, valid)
    return out.reshape(T)


# ---------------------------------------------------------------------------
# discounted_batch() (via the discounted() B=1 adapter above)
# ---------------------------------------------------------------------------

class TestDiscounted:
    """Tests for discounted(rewards_list, gamma, dev) -> Tensor, the B=1 case
    of train.discounted_batch()."""

    # --- basic correctness --------------------------------------------------

    def test_single_reward(self):
        result = discounted([1.0], 0.95, "cpu")
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_two_rewards_first_nonzero(self):
        # R[0] = 1.0 + 0.95*0.0 = 1.0, R[1] = 0.0
        result = discounted([1.0, 0.0], 0.95, "cpu")
        expected = torch.tensor([1.0, 0.0])
        assert torch.allclose(result, expected)

    def test_two_rewards_last_nonzero(self):
        # R[0] = 0.0 + 0.95*1.0 = 0.95, R[1] = 1.0
        result = discounted([0.0, 1.0], 0.95, "cpu")
        expected = torch.tensor([0.95, 1.0])
        assert torch.allclose(result, expected)

    def test_three_uniform_rewards(self):
        # R[2]=1.0, R[1]=1.9, R[0]=2.71 (gamma=0.9)
        result = discounted([1.0, 1.0, 1.0], 0.9, "cpu")
        expected = torch.tensor([2.71, 1.9, 1.0])
        assert torch.allclose(result, expected, atol=1e-5)

    # --- edge cases ---------------------------------------------------------

    def test_empty_list(self):
        result = discounted([], 0.95, "cpu")
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 0

    def test_all_zero_rewards(self):
        result = discounted([0.0, 0.0, 0.0], 0.9, "cpu")
        expected = torch.zeros(3)
        assert torch.allclose(result, expected)

    def test_gamma_zero(self):
        # With gamma=0, R[t] = r[t] for all t (no future propagation)
        rewards = [1.0, 2.0, 3.0]
        result = discounted(rewards, 0.0, "cpu")
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(result, expected)

    def test_gamma_one_undiscounted_sum(self):
        # With gamma=1, R[t] = sum(r[t:])
        rewards = [1.0, 2.0, 3.0]
        result = discounted(rewards, 1.0, "cpu")
        expected = torch.tensor([6.0, 5.0, 3.0])
        assert torch.allclose(result, expected)

    # --- output properties --------------------------------------------------

    def test_output_length_matches_input(self):
        for n in [1, 5, 10]:
            rewards = [float(i) for i in range(n)]
            result = discounted(rewards, 0.9, "cpu")
            assert result.shape[0] == n, f"Expected length {n}, got {result.shape[0]}"

    def test_output_dtype_float32(self):
        result = discounted([1.0, 2.0], 0.9, "cpu")
        assert result.dtype == torch.float32

    def test_output_device_cpu(self):
        result = discounted([1.0, 2.0], 0.9, "cpu")
        assert result.device.type == "cpu"

    def test_output_is_1d(self):
        result = discounted([1.0, 2.0, 3.0], 0.9, "cpu")
        assert result.dim() == 1

    # --- backward recursion sanity check ------------------------------------

    def test_backward_recursion_invariant(self):
        """R[t] == r[t] + gamma * R[t+1] for all valid t."""
        rewards = [0.5, 1.0, -0.5, 2.0, 0.0]
        gamma = 0.95
        result = discounted(rewards, gamma, "cpu")
        for t in range(len(rewards) - 1):
            expected_rt = rewards[t] + gamma * result[t + 1].item()
            assert abs(result[t].item() - expected_rt) < 1e-5, (
                f"Recursion violated at t={t}: "
                f"got {result[t].item():.6f}, expected {expected_rt:.6f}"
            )


# ---------------------------------------------------------------------------
# _physical_cores()
# ---------------------------------------------------------------------------

class TestPhysicalCores:
    """Tests for _physical_cores() -> int."""

    def test_returns_integer(self):
        result = _physical_cores()
        assert isinstance(result, int)

    def test_returns_at_least_one(self):
        result = _physical_cores()
        assert result >= 1

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("DOPA_NUM_THREADS", "4")
        # Re-import is not needed; the function should read env at call time
        result = _physical_cores()
        assert result == 4

    def test_env_var_string_parsed_as_int(self, monkeypatch):
        monkeypatch.setenv("DOPA_NUM_THREADS", "8")
        result = _physical_cores()
        assert result == 8

    def test_env_var_one(self, monkeypatch):
        monkeypatch.setenv("DOPA_NUM_THREADS", "1")
        result = _physical_cores()
        assert result == 1

    def test_no_env_var_result_at_least_one(self, monkeypatch):
        monkeypatch.delenv("DOPA_NUM_THREADS", raising=False)
        result = _physical_cores()
        assert result >= 1

    def test_no_env_var_result_is_int(self, monkeypatch):
        monkeypatch.delenv("DOPA_NUM_THREADS", raising=False)
        result = _physical_cores()
        assert isinstance(result, int)

    def test_no_env_var_logical_even_gt1(self, monkeypatch):
        """When logical_cores > 1 and even, result should be logical_cores // 2."""
        monkeypatch.delenv("DOPA_NUM_THREADS", raising=False)
        import multiprocessing
        logical = os.cpu_count() or 1
        result = _physical_cores()
        if logical > 1 and logical % 2 == 0:
            assert result == logical // 2
        else:
            # Not in the even-gt-1 regime; just check >= 1
            assert result >= 1

    def test_no_env_var_odd_or_one_logical(self, monkeypatch):
        """When logical_cores is 1 or odd, result should equal logical_cores."""
        monkeypatch.delenv("DOPA_NUM_THREADS", raising=False)
        logical = os.cpu_count() or 1
        result = _physical_cores()
        if logical == 1 or logical % 2 != 0:
            assert result == logical
        else:
            # Even > 1 regime; just check >= 1
            assert result >= 1
