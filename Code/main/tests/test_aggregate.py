import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")

from aggregate import _ms, _fmt, _ms_raw, load
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tests for _ms
# ---------------------------------------------------------------------------

class TestMs:
    def test_empty_list_returns_none_none(self):
        result = _ms([])
        assert result == (None, None)

    def test_all_none_returns_none_none(self):
        result = _ms([None])
        assert result == (None, None)

    def test_all_none_multiple_returns_none_none(self):
        result = _ms([None, None, None])
        assert result == (None, None)

    def test_single_value_returns_value_and_zero_std(self):
        mean, std = _ms([5.0])
        assert mean == pytest.approx(5.0)
        assert std == pytest.approx(0.0)

    def test_two_values_returns_correct_mean_and_std(self):
        mean, std = _ms([1.0, 3.0])
        assert mean == pytest.approx(2.0)
        # numpy population std (ddof=0): sqrt(((1-2)^2 + (3-2)^2)/2) = 1.0
        assert std == pytest.approx(1.0)

    def test_filters_none_values(self):
        mean, std = _ms([1.0, None, 3.0])
        expected_mean = np.mean([1.0, 3.0])
        expected_std = np.std([1.0, 3.0])
        assert mean == pytest.approx(expected_mean)
        assert std == pytest.approx(expected_std)

    def test_none_at_start_filtered(self):
        mean, std = _ms([None, 2.0, 4.0])
        expected_mean = np.mean([2.0, 4.0])
        expected_std = np.std([2.0, 4.0])
        assert mean == pytest.approx(expected_mean)
        assert std == pytest.approx(expected_std)

    def test_none_at_end_filtered(self):
        mean, std = _ms([2.0, 4.0, None])
        expected_mean = np.mean([2.0, 4.0])
        expected_std = np.std([2.0, 4.0])
        assert mean == pytest.approx(expected_mean)
        assert std == pytest.approx(expected_std)

    def test_returns_float_mean(self):
        mean, std = _ms([1.0, 3.0])
        assert isinstance(mean, float)

    def test_returns_float_std(self):
        mean, std = _ms([1.0, 3.0])
        assert isinstance(std, float)

    def test_three_values(self):
        xs = [1.0, 2.0, 3.0]
        mean, std = _ms(xs)
        assert mean == pytest.approx(np.mean(xs))
        assert std == pytest.approx(np.std(xs))

    def test_mixed_none_and_values_multiple(self):
        mean, std = _ms([None, 1.0, None, 3.0, None])
        expected_mean = np.mean([1.0, 3.0])
        expected_std = np.std([1.0, 3.0])
        assert mean == pytest.approx(expected_mean)
        assert std == pytest.approx(expected_std)


# ---------------------------------------------------------------------------
# Tests for _fmt
# ---------------------------------------------------------------------------

class TestFmt:
    def test_none_none_returns_na(self):
        assert _fmt((None, None)) == "n/a"

    def test_contains_separator(self):
        result = _fmt((0.5, 0.1))
        assert " +/- " in result

    def test_simple_values(self):
        result = _fmt((0.5, 0.1))
        assert result == "0.5 +/- 0.1"

    def test_three_sig_fig_mean(self):
        result = _fmt((1.234, 0.056))
        assert result == "1.23 +/- 0.056"

    def test_integer_like_values(self):
        result = _fmt((100.0, 5.0))
        assert result == "100 +/- 5"

    def test_format_structure(self):
        m, s = 0.5, 0.1
        expected = f"{m:.3g} +/- {s:.2g}"
        assert _fmt((m, s)) == expected

    def test_format_structure_various(self):
        m, s = 1.234, 0.056
        expected = f"{m:.3g} +/- {s:.2g}"
        assert _fmt((m, s)) == expected

    def test_format_structure_large(self):
        m, s = 100.0, 5.0
        expected = f"{m:.3g} +/- {s:.2g}"
        assert _fmt((m, s)) == expected

    def test_zero_mean_and_std(self):
        result = _fmt((0.0, 0.0))
        expected = f"{0.0:.3g} +/- {0.0:.2g}"
        assert result == expected

    def test_small_values(self):
        m, s = 0.001, 0.0001
        result = _fmt((m, s))
        expected = f"{m:.3g} +/- {s:.2g}"
        assert result == expected


# ---------------------------------------------------------------------------
# Tests for _ms_raw
# ---------------------------------------------------------------------------

class TestMsRaw:
    def test_na_returns_empty_strings(self):
        assert _ms_raw("n/a") == ("", "")

    def test_simple_float_values(self):
        assert _ms_raw("0.5 +/- 0.1") == ("0.5", "0.1")

    def test_integer_like_values(self):
        assert _ms_raw("100 +/- 5") == ("100", "5")

    def test_returns_strings_not_numbers(self):
        mean_str, std_str = _ms_raw("0.5 +/- 0.1")
        assert isinstance(mean_str, str)
        assert isinstance(std_str, str)

    def test_returns_strings_not_numbers_integers(self):
        mean_str, std_str = _ms_raw("100 +/- 5")
        assert isinstance(mean_str, str)
        assert isinstance(std_str, str)

    def test_roundtrip_with_fmt(self):
        formatted = _fmt((1.234, 0.056))
        mean_str, std_str = _ms_raw(formatted)
        assert mean_str != ""
        assert std_str != ""

    def test_na_roundtrip(self):
        formatted = _fmt((None, None))
        assert formatted == "n/a"
        assert _ms_raw(formatted) == ("", "")

    def test_decimal_values(self):
        assert _ms_raw("1.23 +/- 0.056") == ("1.23", "0.056")


# ---------------------------------------------------------------------------
# Tests for load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_empty_root_returns_empty_lists(self):
        with tempfile.TemporaryDirectory() as root:
            trained, untrained = load(root)
            assert trained == []
            assert untrained == []

    def test_no_seed_dirs_returns_empty(self):
        with tempfile.TemporaryDirectory() as root:
            # create non-seed directories
            os.makedirs(os.path.join(root, "other_dir"))
            trained, untrained = load(root)
            assert trained == []
            assert untrained == []

    def test_trained_and_untrained_split(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "seed0"))
            with open(os.path.join(root, "seed0", "results.json"), "w") as f:
                json.dump({"h1": {"accuracy": 0.9}, "untrained": False}, f)
            os.makedirs(os.path.join(root, "seed1"))
            with open(os.path.join(root, "seed1", "results.json"), "w") as f:
                json.dump({"h1": {"accuracy": 0.5}, "untrained": True}, f)
            trained, untrained = load(root)
            assert len(trained) == 1
            assert len(untrained) == 1

    def test_trained_result_has_correct_data(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "seed0"))
            data = {"h1": {"accuracy": 0.9}, "untrained": False}
            with open(os.path.join(root, "seed0", "results.json"), "w") as f:
                json.dump(data, f)
            trained, untrained = load(root)
            assert trained[0]["h1"]["accuracy"] == 0.9

    def test_untrained_result_has_correct_data(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "seed1"))
            data = {"h1": {"accuracy": 0.5}, "untrained": True}
            with open(os.path.join(root, "seed1", "results.json"), "w") as f:
                json.dump(data, f)
            trained, untrained = load(root)
            assert untrained[0]["h1"]["accuracy"] == 0.5

    def test_multiple_trained_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            for i in range(3):
                os.makedirs(os.path.join(root, f"seed{i}"))
                with open(os.path.join(root, f"seed{i}", "results.json"), "w") as f:
                    json.dump({"accuracy": 0.9 - i * 0.1, "untrained": False}, f)
            trained, untrained = load(root)
            assert len(trained) == 3
            assert len(untrained) == 0

    def test_multiple_untrained_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            for i in range(2):
                os.makedirs(os.path.join(root, f"seed{i}"))
                with open(os.path.join(root, f"seed{i}", "results.json"), "w") as f:
                    json.dump({"accuracy": 0.5, "untrained": True}, f)
            trained, untrained = load(root)
            assert len(trained) == 0
            assert len(untrained) == 2

    def test_seed_without_results_json_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            # seed0 has results.json, seed1 does not
            os.makedirs(os.path.join(root, "seed0"))
            with open(os.path.join(root, "seed0", "results.json"), "w") as f:
                json.dump({"untrained": False}, f)
            os.makedirs(os.path.join(root, "seed1"))
            # no results.json in seed1
            trained, untrained = load(root)
            assert len(trained) == 1
            assert len(untrained) == 0

    def test_absent_untrained_key_goes_to_trained(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "seed0"))
            # no "untrained" key at all → falsy/absent → trained
            with open(os.path.join(root, "seed0", "results.json"), "w") as f:
                json.dump({"accuracy": 0.8}, f)
            trained, untrained = load(root)
            assert len(trained) == 1
            assert len(untrained) == 0

    def test_returns_tuple_of_two_lists(self):
        with tempfile.TemporaryDirectory() as root:
            result = load(root)
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], list)
            assert isinstance(result[1], list)
