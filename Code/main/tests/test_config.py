import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dataclasses
import pytest
from config import Config


class TestConfigToDict:
    """Tests for Config.to_dict()"""

    def test_returns_dict(self):
        cfg = Config()
        result = cfg.to_dict()
        assert isinstance(result, dict)

    def test_not_a_config(self):
        cfg = Config()
        result = cfg.to_dict()
        assert not isinstance(result, Config)

    def test_all_fields_present(self):
        cfg = Config()
        result = cfg.to_dict()
        expected_keys = {f.name for f in dataclasses.fields(cfg)}
        assert expected_keys == set(result.keys())

    def test_values_match_config(self):
        cfg = Config()
        result = cfg.to_dict()
        for field in dataclasses.fields(cfg):
            assert result[field.name] == getattr(cfg, field.name), (
                f"Mismatch on field '{field.name}': "
                f"dict has {result[field.name]!r}, config has {getattr(cfg, field.name)!r}"
            )

    # --- Known default field values ---

    def test_default_delay_start(self):
        d = Config().to_dict()
        assert d["delay_start"] == 15

    def test_default_delay_max(self):
        d = Config().to_dict()
        assert d["delay_max"] == 40

    def test_default_n_actions(self):
        d = Config().to_dict()
        assert d["n_actions"] == 5

    def test_default_obs_dim(self):
        d = Config().to_dict()
        assert d["obs_dim"] == 6

    def test_default_n_gd(self):
        d = Config().to_dict()
        assert d["n_gd"] == 512

    def test_default_n_hab(self):
        d = Config().to_dict()
        assert d["n_hab"] == 512

    def test_default_batch_size(self):
        d = Config().to_dict()
        assert d["batch_size"] == 128

    def test_default_device(self):
        d = Config().to_dict()
        assert d["device"] == "cpu"

    def test_default_gamma(self):
        d = Config().to_dict()
        assert d["gamma"] == pytest.approx(0.99)

    # --- Mutation isolation ---

    def test_modifying_dict_does_not_affect_config(self):
        cfg = Config()
        d = cfg.to_dict()
        # Mutate every key in the returned dict
        for key in list(d.keys()):
            original = d[key]
            d[key] = None
            assert getattr(cfg, key) == original, (
                f"Config.{key} was affected by mutating the returned dict"
            )

    def test_returns_new_dict_each_call(self):
        cfg = Config()
        d1 = cfg.to_dict()
        d2 = cfg.to_dict()
        assert d1 is not d2

    def test_two_dicts_are_equal_in_value(self):
        cfg = Config()
        d1 = cfg.to_dict()
        d2 = cfg.to_dict()
        assert d1 == d2

    def test_mutating_first_dict_does_not_affect_second(self):
        cfg = Config()
        d1 = cfg.to_dict()
        d2 = cfg.to_dict()
        for key in list(d1.keys()):
            d1[key] = None
        # d2 should be untouched
        assert d2["delay_start"] == 15

    # --- Custom field values ---

    def test_custom_seed(self):
        cfg = Config(seed=99)
        assert cfg.to_dict()["seed"] == 99

    def test_custom_field_reflected_in_dict(self):
        """Any field passed at construction should appear in the dict."""
        cfg = Config(seed=42)
        d = cfg.to_dict()
        assert d["seed"] == 42
        assert d["seed"] == cfg.seed
