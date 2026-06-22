"""
Tests for maze_layout() in environment.py.

`_blocked_cells()` (a standalone pure function returning the set of cells
sealed off when one arm is blocked) was deleted in the vectorisation
rewrite — that logic is now inlined as a boolean mask (`seal`) inside
TMazeVecEnv._passable_target, computed jointly over the whole batch and not
exposed as a per-cell function. There is no longer a public symbol with
that name or shape to test directly, so this file instead:
  1. Tests maze_layout() against the current parametric grid (it now takes
     an optional cfg and derives geometry from len_edge/difficulty; calling
     it with no cfg defaults to len_edge=5/difficulty=0, which is a
     deliberately-disallowed degenerate grid post-rewrite -- see
     environment.py's grid_dims()).
  2. Replaces the _blocked_cells() unit tests with a behavioural check of
     the same "blocked arm is impassable during sample" property, driven
     through the public TMazeVecEnv API (the closest current equivalent).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config import Config
from environment import maze_layout, TMazeVecEnv, SAMPLE


# ---------------------------------------------------------------------------
# maze_layout() tests — default Config()
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def layout(cfg):
    return maze_layout(cfg)


class TestMazeLayoutDefaultConfig:
    def test_returns_dict(self, layout):
        assert isinstance(layout, dict)

    def test_rows_matches_grid_dims(self, layout, cfg):
        from environment import grid_dims
        h, w = grid_dims(cfg.len_edge, cfg.difficulty)
        assert layout["rows"] == h

    def test_cols_matches_grid_dims(self, layout, cfg):
        from environment import grid_dims
        h, w = grid_dims(cfg.len_edge, cfg.difficulty)
        assert layout["cols"] == w

    def test_exact_keys(self, layout):
        expected_keys = {"rows", "cols", "passable", "start", "junction", "l_end", "r_end"}
        assert set(layout.keys()) == expected_keys

    def test_passable_is_sorted(self, layout):
        passable = layout["passable"]
        assert passable == sorted(passable, key=lambda rc: (rc[0], rc[1]))

    def test_passable_cells_are_two_element_lists(self, layout):
        for cell in layout["passable"]:
            assert isinstance(cell, list) and len(cell) == 2

    def test_start_is_junction_column(self, layout):
        assert layout["start"][1] == layout["junction"][1]

    def test_junction_between_l_end_and_r_end(self, layout):
        assert layout["l_end"][1] < layout["junction"][1] < layout["r_end"][1]

    def test_l_end_and_r_end_same_row_as_junction(self, layout):
        assert layout["l_end"][0] == layout["junction"][0]
        assert layout["r_end"][0] == layout["junction"][0]

    def test_start_junction_l_end_r_end_are_passable(self, layout):
        passable_set = {tuple(c) for c in layout["passable"]}
        for key in ("start", "junction", "l_end", "r_end"):
            assert tuple(layout[key]) in passable_set, f"{key} should be passable"


# ---------------------------------------------------------------------------
# maze_layout() — no cfg / default args
# ---------------------------------------------------------------------------

class TestMazeLayoutNoCfg:
    def test_default_args_raise_on_degenerate_grid(self):
        """maze_layout(None) falls back to len_edge=5/difficulty=0, which
        grid_dims() now rejects as degenerate (no stem) — this is the
        intentional post-rewrite behaviour, not a bug to work around."""
        with pytest.raises(ValueError):
            maze_layout()


# ---------------------------------------------------------------------------
# maze_layout() — varies with len_edge / difficulty
# ---------------------------------------------------------------------------

class TestMazeLayoutParametric:
    @pytest.mark.parametrize("len_edge,difficulty", [(7, 0), (7, 1), (7, 2), (9, 2)])
    def test_layout_matches_requested_geometry(self, len_edge, difficulty):
        cfg = Config(len_edge=len_edge, difficulty=difficulty)
        layout = maze_layout(cfg)
        from environment import grid_dims
        h, w = grid_dims(len_edge, difficulty)
        assert layout["rows"] == h
        assert layout["cols"] == w

    def test_larger_len_edge_gives_wider_grid(self):
        layout_small = maze_layout(Config(len_edge=7, difficulty=2))
        layout_large = maze_layout(Config(len_edge=9, difficulty=2))
        assert layout_large["cols"] > layout_small["cols"]

    def test_higher_difficulty_gives_taller_or_equal_grid(self):
        layout_easy = maze_layout(Config(len_edge=7, difficulty=0))
        layout_hard = maze_layout(Config(len_edge=7, difficulty=2))
        assert layout_hard["rows"] >= layout_easy["rows"]


# ---------------------------------------------------------------------------
# Behavioural replacement for the deleted _blocked_cells() unit tests:
# the blocked arm must be impassable during 'sample', the open arm passable.
# ---------------------------------------------------------------------------

class TestBlockedArmSealing:
    def _make_env_at_junction(self, blocked_side):
        cfg = Config()
        rng = np.random.default_rng(0)
        env = TMazeVecEnv(cfg, rng, batch_size=1)
        env.phase[:] = SAMPLE
        env.blocked[:] = blocked_side
        env.open_side[:] = "R" if blocked_side == "L" else "L"
        env.pos[:] = env.g["junction"]
        return env

    def test_blocked_L_arm_impassable(self):
        env = self._make_env_at_junction("L")
        ny = env.pos[:, 0].copy()
        nx = env.pos[:, 1] - 1   # one step toward the L arm
        ok = env._passable_target(ny, nx, env.phase, env.blocked)
        assert not ok[0]

    def test_blocked_R_arm_impassable(self):
        env = self._make_env_at_junction("R")
        ny = env.pos[:, 0].copy()
        nx = env.pos[:, 1] + 1   # one step toward the R arm
        ok = env._passable_target(ny, nx, env.phase, env.blocked)
        assert not ok[0]

    def test_open_L_arm_passable_when_R_blocked(self):
        env = self._make_env_at_junction("R")
        ny = env.pos[:, 0].copy()
        nx = env.pos[:, 1] - 1   # one step toward the (open) L arm
        ok = env._passable_target(ny, nx, env.phase, env.blocked)
        assert ok[0]

    def test_open_R_arm_passable_when_L_blocked(self):
        env = self._make_env_at_junction("L")
        ny = env.pos[:, 0].copy()
        nx = env.pos[:, 1] + 1   # one step toward the (open) R arm
        ok = env._passable_target(ny, nx, env.phase, env.blocked)
        assert ok[0]

    def test_junction_itself_always_passable_in_sample(self):
        for side in ("L", "R"):
            env = self._make_env_at_junction(side)
            ny = env.pos[:, 0].copy()
            nx = env.pos[:, 1].copy()
            ok = env._passable_target(ny, nx, env.phase, env.blocked)
            assert ok[0]
