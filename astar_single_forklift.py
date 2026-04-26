"""A* pathfinding for one forklift on the generated warehouse map.

Depends on generate_map.py for map generation.
"""

from __future__ import annotations

import argparse
import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from generate_map import MapConfig, generate_warehouse_map


GridPos = Tuple[int, int]  # (row, col)
ANIMATION_CACHE: List[object] = []


def heuristic(a: GridPos, b: GridPos) -> int:
    """Manhattan distance for 4-neighbor movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(grid: np.ndarray, node: GridPos) -> List[GridPos]:
    """Return valid 4-direction neighbors that are not racks."""
    rows, cols = grid.shape
    r, c = node
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    neighbors: List[GridPos] = []

    for nr, nc in candidates:
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            neighbors.append((nr, nc))

    return neighbors


def reconstruct_path(came_from: Dict[GridPos, GridPos], current: GridPos) -> List[GridPos]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_pathfinding(grid: np.ndarray, start: GridPos, goal: GridPos) -> Tuple[List[GridPos], Set[GridPos]]:
    """Run A* and return (path, explored_nodes)."""
    if grid[start] != 0:
        raise ValueError(f"Start position {start} is blocked by a rack.")
    if grid[goal] != 0:
        raise ValueError(f"Goal position {goal} is blocked by a rack.")

    open_heap: List[Tuple[int, int, GridPos]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))

    came_from: Dict[GridPos, GridPos] = {}
    g_score: Dict[GridPos, int] = {start: 0}
    explored: Set[GridPos] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in explored:
            continue

        explored.add(current)

        if current == goal:
            return reconstruct_path(came_from, current), explored

        for neighbor in get_neighbors(grid, current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, 10**12):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbor))

    return [], explored


def parse_coord(value: str) -> GridPos:
    """Parse coordinate from 'row,col'."""
    try:
        r_str, c_str = value.split(",")
        return int(r_str.strip()), int(c_str.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Coordinate must use format row,col. Example: 1,2"
        ) from exc


def random_free_cell(grid: np.ndarray, rng: np.random.Generator) -> GridPos:
    free_cells = np.argwhere(grid == 0)
    if len(free_cells) == 0:
        raise ValueError("No free cells available on the map.")
    idx = int(rng.integers(0, len(free_cells)))
    return int(free_cells[idx][0]), int(free_cells[idx][1])


def plot_result(
    grid: np.ndarray,
    start: GridPos,
    goal: GridPos,
    path: List[GridPos],
    explored: Set[GridPos],
    animate: bool,
    interval_ms: int,
    save_path: Optional[str],
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.animation import FuncAnimation
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plotting. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    cmap = ListedColormap(["#f5f5f5", "#2f4f4f"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(grid, cmap=cmap, interpolation="none", origin="upper")

    if explored:
        ex_r = [p[0] for p in explored]
        ex_c = [p[1] for p in explored]
        ax.scatter(ex_c, ex_r, s=14, c="#87ceeb", alpha=0.35, label="Explored")

    ax.scatter(start[1], start[0], c="#1f77b4", s=80, marker="o", label="Start")
    ax.scatter(goal[1], goal[0], c="#d62728", s=80, marker="*", label="Goal")

    path_line, = ax.plot([], [], color="#2ca02c", linewidth=2.5, label="A* Path")
    forklift_dot, = ax.plot([], [], marker="s", color="#ff7f0e", markersize=8, label="Forklift")

    if path:
        path_r = [p[0] for p in path]
        path_c = [p[1] for p in path]

        if animate:
            def update(frame: int):
                path_line.set_data(path_c[: frame + 1], path_r[: frame + 1])
                forklift_dot.set_data([path_c[frame]], [path_r[frame]])
                return path_line, forklift_dot

            # Keep a strong reference to animation object to prevent garbage collection.
            anim = FuncAnimation(
                fig,
                update,
                frames=len(path),
                interval=interval_ms,
                blit=True,
                repeat=False,
            )
            ANIMATION_CACHE.append(anim)
        else:
            path_line.set_data(path_c, path_r)
            forklift_dot.set_data([path_c[-1]], [path_r[-1]])

    ax.set_title("Single Forklift A* Pathfinding")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#c8c8c8", linestyle="-", linewidth=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180)

    if show:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A* pathfinding for a single forklift in warehouse map.")

    parser.add_argument("--width", type=int, default=30, help="Map width (columns).")
    parser.add_argument("--height", type=int, default=20, help="Map height (rows).")
    parser.add_argument("--rack-width", type=int, default=3, help="Rack width.")
    parser.add_argument("--rack-height", type=int, default=2, help="Rack height.")
    parser.add_argument("--random", action="store_true", help="Use random rack layout.")
    parser.add_argument("--rack-count", type=int, default=24, help="Rack count for random map.")
    parser.add_argument("--aisle", type=int, default=1, help="Minimum spacing between racks.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible map/random positions.")

    parser.add_argument("--start", type=parse_coord, default=None, help="Start coordinate as row,col.")
    parser.add_argument("--goal", type=parse_coord, default=None, help="Goal coordinate as row,col.")

    parser.add_argument("--animate", action="store_true", help="Animate forklift path drawing.")
    parser.add_argument("--interval", type=int, default=150, help="Animation interval in milliseconds.")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path.")

    parser.add_argument("--show", dest="show", action="store_true", help="Show matplotlib window (default: enabled).")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable plot window display.")
    parser.set_defaults(show=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = MapConfig(
        width=args.width,
        height=args.height,
        rack_width=args.rack_width,
        rack_height=args.rack_height,
        random_layout=args.random,
        rack_count=args.rack_count,
        aisle=args.aisle,
        seed=args.seed,
    )

    grid, rack_placed = generate_warehouse_map(config)
    rng = np.random.default_rng(args.seed)

    start = args.start if args.start is not None else random_free_cell(grid, rng)
    goal = args.goal if args.goal is not None else random_free_cell(grid, rng)

    attempts = 0
    while goal == start and attempts < 20:
        goal = random_free_cell(grid, rng)
        attempts += 1

    path, explored = astar_pathfinding(grid, start, goal)

    print(f"Map: {args.width}x{args.height}, racks placed: {rack_placed}")
    print(f"Start: {start}")
    print(f"Goal : {goal}")
    print(f"Explored nodes: {len(explored)}")

    if path:
        print(f"Path found with {len(path) - 1} steps.")
    else:
        print("No path found.")

    plot_result(
        grid=grid,
        start=start,
        goal=goal,
        path=path,
        explored=explored,
        animate=args.animate,
        interval_ms=args.interval,
        save_path=args.save,
        show=args.show,
    )


if __name__ == "__main__":
    main()
