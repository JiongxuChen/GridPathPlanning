"""The new verifier input must encode raw geometry, not execution results."""
from analysis.local_occupancy_v1 import representation
from gridpath.planning.protocol import ACTION_MARKER, END_MARKER
from gridpath.types import Grid


def test_center_and_boundaries_from_raw_grid():
    grid = Grid(6, 6, frozenset({(1, 4)}), (3, 4), (0, 5))
    assert "CENTER blocked" in representation.occupancy(grid, (1, 4))
    assert "N outside" in representation.occupancy(grid, (0, 5))
    assert "CENTER free" in representation.occupancy(grid, (2, 4))


def test_render_preserves_markers_and_unchecked_positions():
    grid = Grid(6, 6, frozenset({(1, 4)}), (3, 4), (0, 5))
    text = representation.render(grid, "up up up right")
    assert text.count(ACTION_MARKER) == 4
    assert text.count(END_MARKER) == 1
    assert "step 2 from row 2 col 4 action up to row 1 col 4" in text
    assert "step 3 from row 1 col 4 action up to row 0 col 4" in text
    assert "CENTER blocked" in text
    assert "goal minus end dr 0 dc 0" in text
    assert "reason=" not in text and "invalid |" not in text
