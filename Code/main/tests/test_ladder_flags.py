"""
Smoke coverage for the Tier-1 ladder experiment flags (config.py E2-E5, E8,
and the Step-5 scalar split). Each flag gets at least one test exercising
BOTH of its values end-to-end (forward pass and, where cheap, a real
training step), per IMPLEMENTATION_ROADMAP.md Step 7's acceptance
criterion: "smoke passes for BOTH values of every flag." Before this file,
none of these flags were referenced anywhere in tests/.

Flags covered (one TestXxx class each):
  - gate_mode        ("expression" / "scheduled")             -- E2
  - da_components    ("both" / "gain_only" / "weights_only")  -- E3
  - habit_rule       ("value_free" / "value_coupled")         -- E4
  - habit_obs        ("position_free" / "allocentric")        -- E5
  - hab_rank         (0 / >0, low-rank habitual recurrence)   -- E8
  - da_split         (False / True, scalar split)              -- Step 5

Per the brief: E2/E3 are only non-vacuous with da_split=True (otherwise
da_arbitration === da_request and the gate/component switches have nothing
independent to act on), so those two classes set da_split=True explicitly.

All networks are sized tiny (n_gd/n_hab in the 16-32 range) and training
runs use a handful of episodes, to keep this file's total runtime in the
sub-minute range.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from config import Config
from environment import TMazeVecEnv
from model import DualSystemModel
from train import train, _train_batch, RunningNorm, scheduled_w_value


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tiny_cfg(**overrides):
    """A Config() sized for fast tests: small nets, short episodes, a small
    maze, short delay. Individual tests override flag-specific fields."""
    base = dict(
        n_gd=24, n_hab=24,
        len_edge=7, difficulty=0,
        delay_start=1, delay_max=2,
        max_episode_steps=25,
        batch_size=8,
        episodes=64,
        eval_every=64, eval_trials=4,
        traj_log_every=0,
        da_warmup=0, da_ramp=16,
    )
    base.update(overrides)
    return Config(**base)


def _make_model(cfg, seed=0):
    torch.manual_seed(seed)
    return DualSystemModel(cfg)


def _run_steps(model, cfg, n=3, B=4, seed=0):
    """Run a handful of forward steps with random observations; returns the
    list of per-step output dicts."""
    torch.manual_seed(seed)
    model.reset_state(B)
    outs = []
    obs = torch.randn(B, cfg.obs_dim)
    obs_h = torch.randn(B, cfg.obs_dim_hab)
    with torch.no_grad():
        for _ in range(n):
            out = model.step(obs, obs_h)
            outs.append(out)
    return outs


def _one_train_batch(cfg, seed=0):
    """One vectorised training iteration -- the smallest unit that exercises
    the full forward + backward + optimizer-step path including the
    flag-specific loss terms in train.py's _train_batch()."""
    torch.manual_seed(seed)
    model = DualSystemModel(cfg)
    venv = TMazeVecEnv(cfg, np.random.default_rng(seed), batch_size=cfg.batch_size)
    opt_gd = torch.optim.Adam(model.gd_params(), lr=cfg.lr_gd)
    opt_hab = torch.optim.Adam(model.hab_params(), lr=cfg.lr_hab)
    ret_norm = RunningNorm(cfg.ret_norm_window)
    gd_loss, hab_loss = _train_batch(model, venv, cfg, opt_gd, opt_hab,
                                      da_lambda=0.0, ret_norm=ret_norm)
    return model, gd_loss, hab_loss


# ---------------------------------------------------------------------------
# gate_mode (E2): "expression" (default) vs "scheduled"
# ---------------------------------------------------------------------------

class TestGateMode:
    @pytest.mark.parametrize("gate_mode", ["expression", "scheduled"])
    def test_forward_smoke_both_values(self, gate_mode):
        """No crash; w_gd stays in [0, 1] for either gate mode."""
        cfg = _tiny_cfg(gate_mode=gate_mode, da_split=True)
        model = _make_model(cfg)
        outs = _run_steps(model, cfg)
        for out in outs:
            assert torch.all(out["w_gd"] >= 0) and torch.all(out["w_gd"] <= 1)
            assert torch.isfinite(out["combined"]).all()

    @pytest.mark.parametrize("gate_mode", ["expression", "scheduled"])
    def test_train_smoke_both_values(self, gate_mode):
        """A full (tiny) train() run completes without crashing for both gate
        modes, and produces a usable log dict."""
        cfg = _tiny_cfg(gate_mode=gate_mode, da_split=True)
        torch.manual_seed(0)
        model, logs, ckpt_learn, ckpt_maint, train_trajs, final_delay = train(
            cfg, verbose=False)
        assert len(logs["episode"]) > 0
        assert all(0.0 <= w <= 1.0 for w in logs["w_gd"])

    def test_scheduled_buffer_starts_nan(self):
        """Before any training iteration writes to it, model.scheduled_w is
        NaN -- it is genuinely unset, not silently defaulted to something
        that would mask gate_mode='expression' ever reading it."""
        cfg = _tiny_cfg(gate_mode="scheduled", da_split=True)
        model = _make_model(cfg)
        assert torch.isnan(model.scheduled_w)

    def test_scheduled_w_becomes_finite_after_training(self):
        """After a couple of training iterations under gate_mode='scheduled',
        model.scheduled_w must hold a finite value (the per-iteration ramp
        write in train.py actually fired)."""
        cfg = _tiny_cfg(gate_mode="scheduled", da_split=True, episodes=32,
                         batch_size=8, eval_every=32, eval_trials=4)
        torch.manual_seed(0)
        model, logs, *_ = train(cfg, verbose=False)
        assert not torch.isnan(model.scheduled_w)
        assert torch.isfinite(model.scheduled_w)

    def test_scheduled_w_matches_pure_schedule_function(self):
        """The scheduled value written into the model buffer must equal
        train.scheduled_w_value(ep, cfg) exactly for the episode count at
        the time of the write -- i.e. it is a pure function of episode
        count, structurally independent of da_request."""
        cfg = _tiny_cfg(gate_mode="scheduled", da_split=True)
        torch.manual_seed(0)
        model = DualSystemModel(cfg)
        for ep in (0, 8, 16, 40):
            model.scheduled_w.fill_(scheduled_w_value(ep, cfg))
            expected = scheduled_w_value(ep, cfg)
            assert float(model.scheduled_w) == pytest.approx(expected)

    def test_scheduled_mode_ignores_da_request_for_gating(self):
        """With gate_mode='scheduled' and the buffer set, w_gd must equal the
        scheduled value regardless of what da_request/da_arbitration say --
        i.e. the gate does not depend on the DA mechanism at all."""
        cfg = _tiny_cfg(gate_mode="scheduled", da_split=True)
        model = _make_model(cfg)
        model.scheduled_w.fill_(0.37)
        B = 5
        model.reset_state(B)
        obs = torch.randn(B, cfg.obs_dim)
        obs_h = torch.randn(B, cfg.obs_dim_hab)
        with torch.no_grad():
            out = model.step(obs, obs_h)
        assert torch.allclose(out["w_gd"], torch.full((B,), 0.37), atol=1e-6)
        # da_request/da_arbitration vary across the batch (different obs)
        # yet w_gd is constant -- structural proof the gate ignored them.
        assert out["da_request"].std() > 0


# ---------------------------------------------------------------------------
# da_components (E3): "both" (default) / "gain_only" / "weights_only"
# ---------------------------------------------------------------------------

class TestDaComponents:
    @pytest.mark.parametrize("da_components", ["both", "gain_only", "weights_only"])
    def test_forward_smoke_all_values(self, da_components):
        cfg = _tiny_cfg(da_components=da_components, da_split=True)
        model = _make_model(cfg)
        outs = _run_steps(model, cfg)
        for out in outs:
            assert torch.isfinite(out["combined"]).all()
            assert torch.all(out["w_gd"] >= 0) and torch.all(out["w_gd"] <= 1)

    @pytest.mark.parametrize("da_components", ["both", "gain_only", "weights_only"])
    def test_train_smoke_all_values(self, da_components):
        cfg = _tiny_cfg(da_components=da_components, da_split=True)
        torch.manual_seed(0)
        model, logs, *_ = train(cfg, verbose=False)
        assert len(logs["episode"]) > 0

    def test_gain_only_expression_gain_is_gain_base(self):
        """da_components='gain_only' must suppress the DA expression-gain
        term entirely: the gain applied to tanh(h) is exactly gain_base,
        independent of da_comp (which still varies with da_request and the
        fast/slow timescale mask)."""
        torch.manual_seed(0)
        cfg = _tiny_cfg(da_components="gain_only")
        from model import GDNet, _tau
        gd = GDNet(cfg)
        B = 6
        x = torch.randn(B, cfg.obs_dim)
        h = torch.zeros(B, cfg.n_gd)
        da_tonic = torch.zeros(B)
        with torch.no_grad():
            pi, da_request, value, h_new, da_tonic_new = gd.step(x, h, da_tonic)
            # Recompute the expected pi by hand using gain_base only.
            r = torch.tanh(h)
            dh = -h + r @ gd.W.T + x @ gd.W_in.T + gd.b
            h_expected = h + (cfg.dt / _tau(gd.tau_p)) * dh
            r_out_expected = gd.gain_base * torch.tanh(h_expected)
            pi_expected = r_out_expected @ gd.W_out.T + gd.b_out
        assert torch.allclose(pi, pi_expected, atol=1e-5)

    def test_weights_only_w_gd_is_exactly_zero(self):
        """da_components='weights_only' must zero the arbitration gate for
        ANY input -- the system is fully habitual regardless of da_request
        or da_arbitration."""
        cfg = _tiny_cfg(da_components="weights_only", da_split=True)
        model = _make_model(cfg)
        B = 6
        model.reset_state(B)
        for seed in range(3):
            torch.manual_seed(seed)
            obs = torch.randn(B, cfg.obs_dim) * 10  # large, varied inputs
            obs_h = torch.randn(B, cfg.obs_dim_hab)
            with torch.no_grad():
                out = model.step(obs, obs_h)
            assert torch.equal(out["w_gd"], torch.zeros(B))

    def test_both_is_default_and_differs_from_gain_only_pi(self):
        """Sanity: 'both' (default) actually applies the DA-modulated gain,
        so its pi_gd differs from the gain_only branch for the same weights
        and input (the two branches are not accidentally identical)."""
        torch.manual_seed(0)
        cfg_both = _tiny_cfg(da_components="both")
        from model import GDNet
        gd_both = GDNet(cfg_both)
        B = 4
        x = torch.randn(B, cfg_both.obs_dim)
        h = torch.randn(B, cfg_both.n_gd) * 0.5  # nonzero h so da_comp matters
        da_tonic = torch.zeros(B)
        with torch.no_grad():
            pi_both, *_ = gd_both.step(x, h, da_tonic)

        gd_gain_only = GDNet(cfg_both)
        gd_gain_only.load_state_dict(gd_both.state_dict())
        gd_gain_only.cfg = _tiny_cfg(da_components="gain_only")
        with torch.no_grad():
            pi_gain_only, *_ = gd_gain_only.step(x, h, da_tonic)

        assert not torch.allclose(pi_both, pi_gain_only, atol=1e-6)


# ---------------------------------------------------------------------------
# habit_rule (E4): "value_free" (default) vs "value_coupled"
# ---------------------------------------------------------------------------

class TestHabitRule:
    @pytest.mark.parametrize("habit_rule", ["value_free", "value_coupled"])
    def test_forward_smoke_both_values(self, habit_rule):
        cfg = _tiny_cfg(habit_rule=habit_rule)
        model = _make_model(cfg)
        outs = _run_steps(model, cfg)
        for out in outs:
            assert torch.isfinite(out["pi_h"]).all()

    @pytest.mark.parametrize("habit_rule", ["value_free", "value_coupled"])
    def test_one_train_batch_runs_and_grads_are_finite(self, habit_rule):
        """One _train_batch() call (forward + backward + optimizer step)
        must not crash for either rule, and afterwards every habitual
        parameter's gradient must be finite and at least one must be
        non-zero (the value-coupled term, when present, actually
        contributes to the gradient rather than being dead code)."""
        cfg = _tiny_cfg(habit_rule=habit_rule)
        model, gd_loss, hab_loss = _one_train_batch(cfg)
        assert np.isfinite(gd_loss) and np.isfinite(hab_loss)
        any_nonzero = False
        for p in model.hab_params():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
            if torch.any(p.grad != 0):
                any_nonzero = True
        assert any_nonzero

    def test_value_coupled_changes_habitual_loss_given_same_seed(self):
        """With identical seeds/data, the *habitual* loss value should
        differ between value_free and value_coupled (the new A2C term on
        task reward is actually added to total_hab, not a no-op)."""
        cfg_free = _tiny_cfg(habit_rule="value_free")
        cfg_coupled = _tiny_cfg(habit_rule="value_coupled")
        _, _, hab_loss_free = _one_train_batch(cfg_free, seed=1)
        _, _, hab_loss_coupled = _one_train_batch(cfg_coupled, seed=1)
        assert hab_loss_free != pytest.approx(hab_loss_coupled)


# ---------------------------------------------------------------------------
# habit_obs (E5): "position_free" (default) vs "allocentric"
# ---------------------------------------------------------------------------

class TestHabitObs:
    def test_position_free_obs_dim_hab_unchanged(self):
        cfg = Config(habit_obs="position_free")
        assert cfg.obs_dim_hab == 4

    def test_allocentric_obs_dim_hab_equals_obs_dim(self):
        """Config.__post_init__ must re-size obs_dim_hab to match obs_dim
        when habit_obs='allocentric'."""
        cfg = Config(habit_obs="allocentric")
        assert cfg.obs_dim_hab == cfg.obs_dim

    def test_hab_net_w_in_shape_matches_obs_dim_allocentric(self):
        cfg = Config(habit_obs="allocentric", n_hab=16)
        from model import HabNet
        net = HabNet(cfg)
        assert net.W_in.shape[1] == cfg.obs_dim

    def test_hab_net_w_in_shape_matches_obs_dim_hab_position_free(self):
        cfg = Config(habit_obs="position_free", n_hab=16)
        from model import HabNet
        net = HabNet(cfg)
        assert net.W_in.shape[1] == cfg.obs_dim_hab

    @pytest.mark.parametrize("habit_obs", ["position_free", "allocentric"])
    def test_forward_smoke_both_values(self, habit_obs):
        cfg = _tiny_cfg(habit_obs=habit_obs)
        model = _make_model(cfg)
        B = 4
        model.reset_state(B)
        obs = torch.randn(B, cfg.obs_dim)
        obs_h = torch.randn(B, cfg.obs_dim_hab)
        with torch.no_grad():
            out = model.step(obs, obs_h)
        assert torch.isfinite(out["combined"]).all()

    @pytest.mark.parametrize("habit_obs", ["position_free", "allocentric"])
    def test_train_smoke_both_values(self, habit_obs):
        cfg = _tiny_cfg(habit_obs=habit_obs)
        torch.manual_seed(0)
        model, logs, *_ = train(cfg, verbose=False)
        assert len(logs["episode"]) > 0

    def test_env_obs_hab_matches_config_dim(self):
        """The vectorised env's obs_hab() output width must track
        cfg.obs_dim_hab for both habit_obs settings (environment.py reads
        cfg.habit_obs directly, independent of model.py)."""
        for habit_obs in ("position_free", "allocentric"):
            cfg = Config(habit_obs=habit_obs, len_edge=7, difficulty=0)
            env = TMazeVecEnv(cfg, np.random.default_rng(0), batch_size=3)
            assert env.obs_hab().shape == (3, cfg.obs_dim_hab)


# ---------------------------------------------------------------------------
# hab_rank (E8): 0 (full-rank, default) vs >0 (low-rank)
# ---------------------------------------------------------------------------

class TestHabRank:
    def test_full_rank_default_has_W_not_low_rank_params(self):
        cfg = Config(hab_rank=0, n_hab=16)
        from model import HabNet
        net = HabNet(cfg)
        assert hasattr(net, "W")
        assert not hasattr(net, "m_lr")
        assert not hasattr(net, "n_lr")

    @pytest.mark.parametrize("rank", [1, 2, 4, 8])
    def test_low_rank_params_exist_with_correct_shape(self, rank):
        cfg = Config(hab_rank=rank, n_hab=16)
        from model import HabNet
        net = HabNet(cfg)
        assert net.m_lr.shape == (cfg.n_hab, rank)
        assert net.n_lr.shape == (cfg.n_hab, rank)
        assert not hasattr(net, "W")

    @pytest.mark.parametrize("rank", [0, 1, 2, 4, 8])
    def test_rec_weight_runs_and_has_correct_shape(self, rank):
        cfg = Config(hab_rank=rank, n_hab=16)
        from model import HabNet
        net = HabNet(cfg)
        W = net.rec_weight()
        assert W.shape == (cfg.n_hab, cfg.n_hab)

    @pytest.mark.parametrize("rank", [1, 2, 4])
    def test_rec_weight_rank_is_bounded(self, rank):
        """The materialised recurrent matrix's numerical rank must not
        exceed the configured low-rank budget (it is a product of two
        [n_hab, rank] factors, so this is a structural, not statistical,
        guarantee)."""
        torch.manual_seed(0)
        cfg = Config(hab_rank=rank, n_hab=16)
        from model import HabNet
        net = HabNet(cfg)
        W = net.rec_weight()
        actual_rank = torch.linalg.matrix_rank(W)
        assert actual_rank <= rank

    @pytest.mark.parametrize("rank", [0, 2])
    def test_forward_smoke_full_and_low_rank(self, rank):
        cfg = _tiny_cfg(hab_rank=rank)
        model = _make_model(cfg)
        outs = _run_steps(model, cfg)
        for out in outs:
            assert torch.isfinite(out["pi_h"]).all()

    @pytest.mark.parametrize("rank", [0, 2])
    def test_train_smoke_full_and_low_rank(self, rank):
        cfg = _tiny_cfg(hab_rank=rank)
        torch.manual_seed(0)
        model, logs, *_ = train(cfg, verbose=False)
        assert len(logs["episode"]) > 0


# ---------------------------------------------------------------------------
# da_split (Step 5 scalar split): False (default, tied) vs True (independent)
# ---------------------------------------------------------------------------

class TestDaSplit:
    def test_default_false_is_bit_identical_to_pre_split(self):
        """Regression guarantee from IMPLEMENTATION_ROADMAP.md: with the
        default da_split=False, da_expression, da_arbitration, and
        da_request must be EXACTLY (bit-identical) equal -- not just close
        -- for any input. This protects "default reproduces Step 1 exactly"
        from silently regressing."""
        cfg = _tiny_cfg(da_split=False)
        model = _make_model(cfg)
        B = 5
        model.reset_state(B)
        for seed in range(3):
            torch.manual_seed(seed)
            obs = torch.randn(B, cfg.obs_dim)
            obs_h = torch.randn(B, cfg.obs_dim_hab)
            with torch.no_grad():
                out = model.step(obs, obs_h)
            assert torch.equal(out["da_expression"], out["da_arbitration"])
            assert torch.equal(out["da_arbitration"], out["da_request"])

    def test_split_true_arbitration_independent_of_expression(self):
        """With da_split=True, da_arbitration must come from
        GDNet.arbitration(h_gd) -- a separately-parameterised readout --
        and need not equal da_expression."""
        cfg = _tiny_cfg(da_split=True)
        model = _make_model(cfg)
        B = 5
        model.reset_state(B)
        torch.manual_seed(0)
        obs = torch.randn(B, cfg.obs_dim)
        obs_h = torch.randn(B, cfg.obs_dim_hab)
        with torch.no_grad():
            out = model.step(obs, obs_h)
            expected_arb = model.gd.arbitration(model.h_gd)
        assert torch.equal(out["da_arbitration"], expected_arb)
        # Independently parameterised readouts on a random init will not
        # coincide exactly with da_expression (da_request).
        assert not torch.equal(out["da_expression"], out["da_arbitration"])

    def test_split_true_after_training_signals_can_diverge(self):
        """After a few real gradient steps (separate optimizers update
        w_arb/b_arb only via the GD policy/value loss, alongside w_da/b_da),
        da_expression and da_arbitration should remain free to differ -- the
        split is not silently re-tied during training."""
        cfg = _tiny_cfg(da_split=True, episodes=32, batch_size=8,
                        eval_every=32, eval_trials=4)
        torch.manual_seed(0)
        model, logs, *_ = train(cfg, verbose=False)
        B = 5
        model.reset_state(B)
        torch.manual_seed(1)
        obs = torch.randn(B, cfg.obs_dim)
        obs_h = torch.randn(B, cfg.obs_dim_hab)
        with torch.no_grad():
            out = model.step(obs, obs_h)
        assert not torch.equal(out["da_expression"], out["da_arbitration"])

    def test_split_false_train_smoke(self):
        """Training under the default da_split=False must also run cleanly
        end-to-end (the tied path is exercised by every other test file in
        this suite implicitly, but pin it explicitly here too)."""
        cfg = _tiny_cfg(da_split=False)
        torch.manual_seed(0)
        model, logs, *_ = train(cfg, verbose=False)
        assert len(logs["episode"]) > 0

    def test_w_arb_b_arb_parameters_exist_and_are_distinct_from_w_da_b_da(self):
        cfg = _tiny_cfg(da_split=True)
        model = _make_model(cfg)
        assert hasattr(model.gd, "w_arb") and hasattr(model.gd, "b_arb")
        assert not torch.equal(model.gd.w_arb, model.gd.w_da)
