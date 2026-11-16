"""Raw local occupancy at each proposed position; no executor or label access."""
from gridpath.planning.protocol import ACTION_MARKER, END_MARKER, actions_of, virtual_states

NEIGHBORS = (
    ("NW", -1, -1), ("N", -1, 0), ("NE", -1, 1),
    ("W", 0, -1), ("CENTER", 0, 0), ("E", 0, 1),
    ("SW", 1, -1), ("S", 1, 0), ("SE", 1, 1),
)


def occupancy(grid, proposed):
    """A fixed 3x3 window from the input grid, even after an invalid action."""
    result = []
    for name, dr, dc in NEIGHBORS:
        coord = (proposed[0] + dr, proposed[1] + dc)
        value = ("outside" if not grid.in_bounds(coord)
                 else "blocked" if coord in grid.obstacles else "free")
        result.append(f"{name} {value}")
    return "; ".join(result)


def render(grid, candidate):
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
                     f"Local occupancy around proposed: {occupancy(grid, after)}.")
    lines.append(f"{END_MARKER} END at row {states[-1][0]} col {states[-1][1]}; "
                 f"goal row {grid.goal[0]} col {grid.goal[1]}; "
                 f"goal minus end dr {grid.goal[0] - states[-1][0]} "
                 f"dc {grid.goal[1] - states[-1][1]}.")
    return "\n".join(lines)
