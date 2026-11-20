"""Full nine-cell occupancy, with the current center next to its step marker."""
from gridpath.planning.protocol import ACTION_MARKER, END_MARKER, actions_of, virtual_states
from representation import NEIGHBORS


def cell_status(grid, coord):
    return ("outside" if not grid.in_bounds(coord)
            else "blocked" if coord in grid.obstacles else "free")


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
        neighbors = "; ".join(
            f"{name} {cell_status(grid, (after[0] + dr, after[1] + dc))}"
            for name, dr, dc in NEIGHBORS if name != "CENTER")
        lines.append(f"{ACTION_MARKER} CENTER {cell_status(grid, after)}; "
                     f"step {i + 1} from row {before[0]} col {before[1]} "
                     f"action {action} to row {after[0]} col {after[1]}. "
                     f"Local occupancy around proposed: {neighbors}.")
    lines.append(f"{END_MARKER} END at row {states[-1][0]} col {states[-1][1]}; "
                 f"goal row {grid.goal[0]} col {grid.goal[1]}; "
                 f"goal minus end dr {grid.goal[0] - states[-1][0]} "
                 f"dc {grid.goal[1] - states[-1][1]}.")
    return "\n".join(lines)
