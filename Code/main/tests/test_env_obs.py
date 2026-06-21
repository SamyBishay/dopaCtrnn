import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from config import Config
from environment import TMazeFreeNav


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    cfg = Config()
    rng = np.random.default_rng(42)
    e = TMazeFreeNav(cfg, rng)
    e.reset()
    return e


# ---------------------------------------------------------------------------
# Constants (from spec)
# ---------------------------------------------------------------------------

START    = (2, 2)
STEM     = (1, 2)
JUNCTION = (0, 2)
L_END    = (0, 0)
L_ARM1   = (0, 1)
R_END    = (0, 4)
R_ARM1   = (0, 3)

PASSABLE_ALL = {(0,0),(0,1),(0,2),(0,3),(0,4),(1,2),(2,2)}

OBS_DIM  = 6
COLS_1   = 4   # COLS - 1
ROWS_1   = 2   # ROWS - 1


# ===========================================================================
# obs() tests
# ===========================================================================

class TestObs:

    def test_shape(self, env):
        obs = env.obs()
        assert obs.shape == (OBS_DIM,)

    def test_dtype(self, env):
        obs = env.obs()
        assert obs.dtype == np.float32

    def test_obs2_always_zero(self, env):
        """obs[2] must always be 0.0 regardless of position or phase."""
        for pos in PASSABLE_ALL:
            env.pos = pos
            assert env.obs()[2] == 0.0, f"obs[2] != 0.0 at pos={pos}"

    # --- position encoding --------------------------------------------------

    def test_start_position(self, env):
        env.pos = START          # (2, 2)
        env.phase = "pre_sample"
        obs = env.obs()
        # col=2 → 2/4=0.5, row=2 → 2/2=1.0
        assert obs[0] == pytest.approx(0.5, abs=1e-6)
        assert obs[1] == pytest.approx(1.0, abs=1e-6)
        assert obs[2] == pytest.approx(0.0, abs=1e-6)

    def test_junction_position(self, env):
        env.pos = JUNCTION       # (0, 2)
        obs = env.obs()
        # col=2 → 2/4=0.5, row=0 → 0/2=0.0
        assert obs[0] == pytest.approx(0.5, abs=1e-6)
        assert obs[1] == pytest.approx(0.0, abs=1e-6)

    def test_l_end_position(self, env):
        env.pos = L_END          # (0, 0)
        obs = env.obs()
        # col=0 → 0/4=0.0, row=0 → 0/2=0.0
        assert obs[0] == pytest.approx(0.0, abs=1e-6)
        assert obs[1] == pytest.approx(0.0, abs=1e-6)

    def test_r_end_position(self, env):
        env.pos = R_END          # (0, 4)
        obs = env.obs()
        # col=4 → 4/4=1.0, row=0 → 0/2=0.0
        assert obs[0] == pytest.approx(1.0, abs=1e-6)
        assert obs[1] == pytest.approx(0.0, abs=1e-6)

    def test_stem_position(self, env):
        env.pos = STEM           # (1, 2)
        obs = env.obs()
        # col=2 → 2/4=0.5, row=1 → 1/2=0.5
        assert obs[0] == pytest.approx(0.5, abs=1e-6)
        assert obs[1] == pytest.approx(0.5, abs=1e-6)

    def test_l_arm1_position(self, env):
        env.pos = L_ARM1         # (0, 1)
        obs = env.obs()
        # col=1 → 1/4=0.25, row=0 → 0/2=0.0
        assert obs[0] == pytest.approx(0.25, abs=1e-6)
        assert obs[1] == pytest.approx(0.0, abs=1e-6)

    def test_r_arm1_position(self, env):
        env.pos = R_ARM1         # (0, 3)
        obs = env.obs()
        # col=3 → 3/4=0.75, row=0 → 0/2=0.0
        assert obs[0] == pytest.approx(0.75, abs=1e-6)
        assert obs[1] == pytest.approx(0.0, abs=1e-6)

    # --- phase signal -------------------------------------------------------

    def test_no_phase_signal_at_start(self, env):
        """At START with pre_sample phase (no cue), signal slots 3-5 should be 0."""
        env.pos = START
        env.phase = "pre_sample"
        obs = env.obs()
        # pre_sample should have no cue signal
        assert obs[3] == pytest.approx(0.0, abs=1e-6)
        assert obs[4] == pytest.approx(0.0, abs=1e-6)
        assert obs[5] == pytest.approx(0.0, abs=1e-6)

    def test_phase_signal_slots_exist(self, env):
        """obs[3:6] slice must always be length 3."""
        obs = env.obs()
        assert len(obs[3:6]) == 3

    def test_obs_values_normalized(self, env):
        """obs[0] and obs[1] must be in [0, 1] for all passable cells."""
        for pos in PASSABLE_ALL:
            env.pos = pos
            obs = env.obs()
            assert 0.0 <= obs[0] <= 1.0, f"obs[0]={obs[0]} out of range at {pos}"
            assert 0.0 <= obs[1] <= 1.0, f"obs[1]={obs[1]} out of range at {pos}"


# ===========================================================================
# _passable() tests
# ===========================================================================

class TestPassable:

    # --- out-of-bounds ------------------------------------------------------

    def test_oob_row_too_large(self, env):
        assert env._passable((3, 2)) is False

    def test_oob_row_negative(self, env):
        assert env._passable((-1, 2)) is False

    def test_oob_col_too_large(self, env):
        assert env._passable((0, 5)) is False

    def test_oob_col_negative(self, env):
        assert env._passable((0, -1)) is False

    def test_oob_both(self, env):
        assert env._passable((-1, -1)) is False

    # --- cells not in PASSABLE_ALL ------------------------------------------

    def test_wall_row0_col_in_gap(self, env):
        # (1,0), (1,1), (1,3), (1,4) are wall cells (not in PASSABLE_ALL)
        for pos in [(1,0),(1,1),(1,3),(1,4)]:
            assert env._passable(pos) is False, f"Expected False for wall cell {pos}"

    def test_wall_row2_non_stem(self, env):
        # (2,0),(2,1),(2,3),(2,4) are wall cells
        for pos in [(2,0),(2,1),(2,3),(2,4)]:
            assert env._passable(pos) is False, f"Expected False for wall cell {pos}"

    # --- pre_sample phase: all PASSABLE_ALL cells open ----------------------

    def test_pre_sample_all_passable(self, env):
        env.phase = "pre_sample"
        for cell in PASSABLE_ALL:
            assert env._passable(cell) is True, \
                f"Expected True for {cell} in pre_sample phase"

    # --- delay phase: all PASSABLE_ALL cells open ---------------------------

    def test_delay_all_passable(self, env):
        env.phase = "delay"
        for cell in PASSABLE_ALL:
            assert env._passable(cell) is True, \
                f"Expected True for {cell} in delay phase"

    # --- choice phase: all PASSABLE_ALL cells open --------------------------

    def test_choice_all_passable(self, env):
        env.phase = "choice"
        for cell in PASSABLE_ALL:
            assert env._passable(cell) is True, \
                f"Expected True for {cell} in choice phase"

    # --- sample phase with blocked=="L": L arm blocked ----------------------

    def test_sample_blocked_L_arm_impassable(self, env):
        env.phase   = "sample"
        env.blocked = "L"
        # L_ARM1=(0,1) and L_END=(0,0) should be impassable
        assert env._passable(L_ARM1) is False, "L_ARM1 should be blocked when blocked=='L'"
        assert env._passable(L_END)  is False, "L_END should be blocked when blocked=='L'"

    def test_sample_blocked_L_right_arm_still_passable(self, env):
        env.phase   = "sample"
        env.blocked = "L"
        # Right arm cells should still be passable
        assert env._passable(R_ARM1) is True,  "R_ARM1 should be open when blocked=='L'"
        assert env._passable(R_END)  is True,  "R_END should be open when blocked=='L'"

    def test_sample_blocked_L_stem_passable(self, env):
        env.phase   = "sample"
        env.blocked = "L"
        # Stem, junction, start still passable
        for cell in [START, STEM, JUNCTION]:
            assert env._passable(cell) is True, \
                f"{cell} should be passable in sample/blocked==L"

    # --- sample phase with blocked=="R": R arm blocked ----------------------

    def test_sample_blocked_R_arm_impassable(self, env):
        env.phase   = "sample"
        env.blocked = "R"
        # R_ARM1=(0,3) and R_END=(0,4) should be impassable
        assert env._passable(R_ARM1) is False, "R_ARM1 should be blocked when blocked=='R'"
        assert env._passable(R_END)  is False, "R_END should be blocked when blocked=='R'"

    def test_sample_blocked_R_left_arm_still_passable(self, env):
        env.phase   = "sample"
        env.blocked = "R"
        # Left arm cells should still be passable
        assert env._passable(L_ARM1) is True,  "L_ARM1 should be open when blocked=='R'"
        assert env._passable(L_END)  is True,  "L_END should be open when blocked=='R'"

    def test_sample_blocked_R_stem_passable(self, env):
        env.phase   = "sample"
        env.blocked = "R"
        for cell in [START, STEM, JUNCTION]:
            assert env._passable(cell) is True, \
                f"{cell} should be passable in sample/blocked==R"

    # --- symmetry sanity check ----------------------------------------------

    def test_sample_phase_junction_always_passable(self, env):
        """Junction must be passable in sample phase regardless of which arm is blocked."""
        for blk in ("L", "R"):
            env.phase   = "sample"
            env.blocked = blk
            assert env._passable(JUNCTION) is True, \
                f"Junction should be passable with blocked=='{blk}'"
