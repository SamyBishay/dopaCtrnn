import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment import maze_layout, _blocked_cells


# ---------------------------------------------------------------------------
# maze_layout() tests
# ---------------------------------------------------------------------------

def test_maze_layout_returns_dict():
    result = maze_layout()
    assert isinstance(result, dict)


def test_maze_layout_rows():
    assert maze_layout()["rows"] == 3


def test_maze_layout_cols():
    assert maze_layout()["cols"] == 5


def test_maze_layout_passable_content():
    expected = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 2]]
    assert maze_layout()["passable"] == expected


def test_maze_layout_passable_is_sorted():
    passable = maze_layout()["passable"]
    assert passable == sorted(passable, key=lambda rc: (rc[0], rc[1]))


def test_maze_layout_passable_length():
    assert len(maze_layout()["passable"]) == 7


def test_maze_layout_start():
    assert maze_layout()["start"] == [2, 2]


def test_maze_layout_junction():
    assert maze_layout()["junction"] == [0, 2]


def test_maze_layout_l_end():
    assert maze_layout()["l_end"] == [0, 0]


def test_maze_layout_r_end():
    assert maze_layout()["r_end"] == [0, 4]


def test_maze_layout_exact_keys():
    keys = set(maze_layout().keys())
    expected_keys = {"rows", "cols", "passable", "start", "junction", "l_end", "r_end"}
    assert expected_keys.issubset(keys)


# ---------------------------------------------------------------------------
# _blocked_cells() tests
# ---------------------------------------------------------------------------

def test_blocked_cells_L_returns_set():
    result = _blocked_cells("L")
    assert isinstance(result, set)


def test_blocked_cells_R_returns_set():
    result = _blocked_cells("R")
    assert isinstance(result, set)


def test_blocked_cells_L_content():
    expected = {(0, 1), (0, 0)}
    assert _blocked_cells("L") == expected


def test_blocked_cells_L_length():
    assert len(_blocked_cells("L")) == 2


def test_blocked_cells_L_contains_l_arm1():
    assert (0, 1) in _blocked_cells("L")


def test_blocked_cells_L_contains_l_end():
    assert (0, 0) in _blocked_cells("L")


def test_blocked_cells_R_content():
    expected = {(0, 3), (0, 4)}
    assert _blocked_cells("R") == expected


def test_blocked_cells_R_length():
    assert len(_blocked_cells("R")) == 2


def test_blocked_cells_R_contains_r_arm1():
    assert (0, 3) in _blocked_cells("R")


def test_blocked_cells_R_contains_r_end():
    assert (0, 4) in _blocked_cells("R")


def test_blocked_cells_non_L_falls_through_to_R():
    """Any input other than 'L' is treated as 'R' (else branch)."""
    expected = {(0, 3), (0, 4)}
    assert _blocked_cells("X") == expected
    assert _blocked_cells("") == expected
    assert _blocked_cells("l") == expected   # lowercase l is not "L"
    assert _blocked_cells("r") == expected   # lowercase r is not "L"


def test_blocked_cells_L_does_not_contain_R_cells():
    blocked_l = _blocked_cells("L")
    assert (0, 3) not in blocked_l
    assert (0, 4) not in blocked_l


def test_blocked_cells_R_does_not_contain_L_cells():
    blocked_r = _blocked_cells("R")
    assert (0, 0) not in blocked_r
    assert (0, 1) not in blocked_r
