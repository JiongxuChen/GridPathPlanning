"""Controlled masks of the nine local occupancy values, not the grid header."""
from gridpath.planning.protocol import ACTION_MARKER, END_MARKER, actions_of, virtual_states

from representation import NEIGHBORS

MODES = ("center_only", "no_center")


def occupancy(grid, proposed, mode):
    if mode not in MODES:
        raise ValueError("Unknown occupancy mask")
    parts = []
    for name, dr, dc in NEIGHBORS:
        coord = (proposed[0] + dr, proposed[1] + dc)
        visible = (name == "CENTER") == (mode == "center_only")
        value = ("redacted" if not visible else
                 "outside" if not grid.in_bounds(coord) else
                 "blocked" if coord in grid.obstacles else "free")
        parts.append(f"{name} {value}")
    return "; ".join(parts)


def render(grid, candidate, mode):
    actions = actions_of(candidate)
    states = virtual_states(grid.start, actions)
    obstacles = "; ".join(f"row {r} col {c}" for r, c in sorted(grid.obstacles)) or "none"
    lines = [f"Verify all actions. Grid rows {grid.rows} cols {grid.cols}. "
             f"Start row {grid.start[0]} col {grid.start[1]}. "
             f"Goal row {grid.goal[0]} col {grid.goal[1]}. Obstacles: {obstacles}.",
             "Coordinates below are unchecked displacement sums, not legal execution."]
    for i, action in enumerate(actions):
        before, after = states[i:i + 2]
        lines.append(f"{ACTION_MARKER} step {i + 1} from row {before[0]} col {before[1]} "
                     f"action {action} to row {after[0]} col {after[1]}. "
                     f"Local occupancy around proposed: {occupancy(grid, after, mode)}.")
    lines.append(f"{END_MARKER} END at row {states[-1][0]} col {states[-1][1]}; "
                 f"goal row {grid.goal[0]} col {grid.goal[1]}; "
                 f"goal minus end dr {grid.goal[0] - states[-1][0]} "
                 f"dc {grid.goal[1] - states[-1][1]}.")
    return "\n".join(lines)
