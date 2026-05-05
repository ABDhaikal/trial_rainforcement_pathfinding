"""A* pathfinding for multiple forklifts using Hungarian Method or DQN.

Depends on generate_map.py for map generation.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Adjust import to work with local generate_map
try:
    from generate_map import MapConfig, generate_warehouse_map
except ImportError:
    pass

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


GridPos = Tuple[int, int]
ANIMATION_CACHE: List[object] = []
MAX_TIME_LIMIT = 500


# =====================================================================
# DQN Network (Must match the one in hungarian_rainfocement/v2)
# =====================================================================
class DQNetwork(nn.Module):
    def __init__(self, state_dim=9, action_dim=6, hidden_size=64):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        return self.net(x)

ALL_PERMUTATIONS = list(itertools.permutations(range(3)))

def load_dqn_model(model_path: str, hidden_size: int = 64) -> tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQNetwork(state_dim=9, action_dim=6, hidden_size=hidden_size).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "policy_net_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["policy_net_state_dict"])
    else:
        model.load_state_dict(checkpoint) # fallback
    model.eval()
    return model, device

def assign_goals_rl(cost_matrix: np.ndarray, model_path: str) -> Tuple[Tuple[int, ...], float]:
    model, device = load_dqn_model(model_path)
    state_t = torch.FloatTensor(cost_matrix.flatten()).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_t).squeeze().cpu().numpy()
    action = int(np.argmax(q_values))
    assignment = ALL_PERMUTATIONS[action]
    total_cost = sum(cost_matrix[a][t] for a, t in enumerate(assignment))
    return assignment, float(total_cost)

def assign_goals_hungarian(cost_matrix: np.ndarray) -> Tuple[Tuple[int, ...], float]:
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = tuple(col_ind)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return assignment, float(total_cost)


# =====================================================================
# Basic A* for Cost Matrix Generation
# =====================================================================
def heuristic(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(grid: np.ndarray, node: GridPos) -> List[GridPos]:
    rows, cols = grid.shape
    r, c = node
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    neighbors: List[GridPos] = []
    for nr, nc in candidates:
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def astar_cost(grid: np.ndarray, start: GridPos, goal: GridPos) -> float:
    """Returns the distance from start to goal. Returns float('inf') if no path."""
    if grid[start] != 0 or grid[goal] != 0:
        return float('inf')
    
    if start == goal:
        return 0.0

    open_heap: List[Tuple[int, int, GridPos]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))

    g_score: Dict[GridPos, int] = {start: 0}
    explored: Set[GridPos] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in explored:
            continue
        explored.add(current)

        if current == goal:
            return float(g_score[current])

        for neighbor in get_neighbors(grid, current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, 10**12):
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbor))

    return float('inf')

# =====================================================================
# Space-Time A* for Prioritized Planning
# =====================================================================
def st_get_neighbors(
    grid: np.ndarray,
    r: int,
    c: int,
    t: int,
    reservation_table: Set[Tuple[int, int, int]],
    edge_reservations: Set[Tuple[int, int, int, int, int]]
) -> List[Tuple[int, int, int]]:
    rows, cols = grid.shape
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1), (r, c)] # 4 dir + wait
    valid = []
    for nr, nc in candidates:
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            nt = t + 1
            if nt > MAX_TIME_LIMIT:
                continue
            # Check vertex collision
            if (nr, nc, nt) in reservation_table:
                continue
            # Check edge collision: agent moving from (r,c) to (nr,nc) at time t
            # Another agent cannot move from (nr,nc) to (r,c) at time t
            if (nr, nc, r, c, t) in edge_reservations:
                continue
            valid.append((nr, nc, nt))
    return valid

def is_goal_safe(goal: GridPos, t: int, reservation_table: Set[Tuple[int, int, int]]) -> bool:
    """Check if the goal is completely free from time t up to MAX_TIME_LIMIT."""
    for k in range(t, MAX_TIME_LIMIT + 1):
        if (goal[0], goal[1], k) in reservation_table:
            return False
    return True

def space_time_astar(
    grid: np.ndarray,
    start: GridPos,
    goal: GridPos,
    reservation_table: Set[Tuple[int, int, int]],
    edge_reservations: Set[Tuple[int, int, int, int, int]]
) -> List[Tuple[GridPos, int]]:
    """Returns path as list of ((r, c), t)."""
    if grid[start] != 0 or grid[goal] != 0:
        return []
    
    # State: (r, c, t)
    open_heap: List[Tuple[int, int, int, int, int]] = []
    counter = 0
    # f_score, counter, r, c, t
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start[0], start[1], 0))

    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_score: Dict[Tuple[int, int, int], int] = {(start[0], start[1], 0): 0}
    explored: Set[Tuple[int, int, int]] = set()

    while open_heap:
        _, _, r, c, t = heapq.heappop(open_heap)
        current = (r, c, t)
        
        if current in explored:
            continue
        explored.add(current)

        # Check goal condition
        if (r, c) == goal and is_goal_safe(goal, t, reservation_table):
            path = [current]
            curr = current
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()
            return [((pr, pc), pt) for pr, pc, pt in path]

        for nr, nc, nt in st_get_neighbors(grid, r, c, t, reservation_table, edge_reservations):
            neighbor = (nr, nc, nt)
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, 10**12):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic((nr, nc), goal)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, nr, nc, nt))

    return []

# =====================================================================
# Utilities
# =====================================================================
def random_free_cell(grid: np.ndarray, rng: np.random.Generator, exclude: Set[GridPos]) -> GridPos:
    free_cells = np.argwhere(grid == 0)
    rng.shuffle(free_cells)
    for cell in free_cells:
        pos = (int(cell[0]), int(cell[1]))
        if pos not in exclude:
            return pos
    raise ValueError("Not enough free cells available.")

def plot_multi_agent_result(
    grid: np.ndarray,
    starts: List[GridPos],
    goals: List[GridPos],
    paths: List[List[Tuple[GridPos, int]]],
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
        raise ImportError("Matplotlib is required for plotting.") from exc

    cmap = ListedColormap(["#f5f5f5", "#2f4f4f"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(grid, cmap=cmap, interpolation="none", origin="upper")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    path_lines = []
    forklift_dots = []

    for i in range(len(starts)):
        color = colors[i % len(colors)]
        ax.scatter(starts[i][1], starts[i][0], c=color, s=80, marker="o", label=f"Start {i}")
        ax.scatter(goals[i][1], goals[i][0], c=color, s=120, marker="*", label=f"Goal {i}")
        
        pline, = ax.plot([], [], color=color, linewidth=2.5, alpha=0.7)
        fdot, = ax.plot([], [], marker="s", color=color, markersize=10, markeredgecolor='black')
        
        path_lines.append(pline)
        forklift_dots.append(fdot)

    if paths and animate:
        max_len = max(len(p) for p in paths) if paths else 0

        def update(frame: int):
            for i, path in enumerate(paths):
                if not path:
                    continue
                # If frame exceeds path length, stay at the last position
                f_idx = min(frame, len(path) - 1)
                
                path_r = [p[0][0] for p in path[:f_idx + 1]]
                path_c = [p[0][1] for p in path[:f_idx + 1]]
                
                path_lines[i].set_data(path_c, path_r)
                forklift_dots[i].set_data([path_c[-1]], [path_r[-1]])
                
            return path_lines + forklift_dots

        anim = FuncAnimation(
            fig,
            update,
            frames=max_len,
            interval=interval_ms,
            blit=True,
            repeat=False,
        )
        ANIMATION_CACHE.append(anim)
    elif paths and not animate:
        for i, path in enumerate(paths):
            if not path:
                continue
            path_r = [p[0][0] for p in path]
            path_c = [p[0][1] for p in path]
            path_lines[i].set_data(path_c, path_r)
            forklift_dots[i].set_data([path_c[-1]], [path_r[-1]])

    ax.set_title("Multi-Agent Pathfinding (Prioritized Space-Time A*)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#c8c8c8", linestyle="-", linewidth=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180)
    if show:
        plt.show()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-agent pathfinding for forklifts.")

    parser.add_argument("--width", type=int, default=10, help="Map width (columns).")
    parser.add_argument("--height", type=int, default=10, help="Map height (rows).")
    parser.add_argument("--rack-width", type=int, default=5, help="Rack width.")
    parser.add_argument("--rack-height", type=int, default=2, help="Rack height.")
    parser.add_argument("--random", action="store_true", help="Use random rack layout.")
    parser.add_argument("--rack-count", type=int, default=24, help="Rack count for random map.")
    parser.add_argument("--aisle", type=int, default=1, help="Minimum spacing between racks.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible map/random positions.")
    
    parser.add_argument("--forklifts", type=int, default=3, help="Number of forklifts (must be 3 for DQN).")
    parser.add_argument("--assign-method", choices=["hungarian", "rl"], default="hungarian", help="Method for assignment.")
    parser.add_argument("--dqn-model", type=str, default="hungarian_rainfocement/v2/dqn_model.pth", help="Path to DQN model if rl is chosen.")

    parser.add_argument("--animate", action="store_true", help="Animate forklift path drawing.")
    parser.add_argument("--interval", type=int, default=150, help="Animation interval in milliseconds.")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path.")
    parser.add_argument("--show", dest="show", action="store_true", help="Show matplotlib window.")
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

    grid, _ = generate_warehouse_map(config)
    rng = np.random.default_rng(args.seed)

    N = args.forklifts
    
    # 1. Generate unique starts and goals
    starts: List[GridPos] = []
    goals: List[GridPos] = []
    used_cells: Set[GridPos] = set()
    
    for _ in range(N):
        s = random_free_cell(grid, rng, used_cells)
        starts.append(s)
        used_cells.add(s)
        
    for _ in range(N):
        g = random_free_cell(grid, rng, used_cells)
        goals.append(g)
        used_cells.add(g)
        
    print(f"Generated {N} starts and {N} goals.")
    
    # 2. Compute Cost Matrix
    cost_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            cost = astar_cost(grid, starts[i], goals[j])
            cost_matrix[i, j] = cost
            
    print("Cost Matrix:")
    print(cost_matrix)
    
    if np.isinf(cost_matrix).any():
        print("Warning: Some paths are blocked statically (cost is inf).")

    # 3. Assignment
    if args.assign_method == "rl" and N == 3:
        if not os.path.isfile(args.dqn_model):
            print(f"Error: DQN model not found at {args.dqn_model}")
            sys.exit(1)
        assignment, total_cost = assign_goals_rl(cost_matrix, args.dqn_model)
        print(f"RL Assignment: Forklift i -> Goal j: {assignment}")
    else:
        assignment, total_cost = assign_goals_hungarian(cost_matrix)
        print(f"Hungarian Assignment: Forklift i -> Goal j: {assignment}")
        
    print(f"Total static cost: {total_cost}")

    # 4. Space-Time A* for Collision Avoidance
    reservation_table: Set[Tuple[int, int, int]] = set()
    edge_reservations: Set[Tuple[int, int, int, int, int]] = set()
    paths: List[List[Tuple[GridPos, int]]] = [[] for _ in range(N)]
    
    # To prioritize, we can sort agents. Here we just plan in order 0 to N-1
    for agent_id in range(N):
        start = starts[agent_id]
        goal_idx = assignment[agent_id]
        goal = goals[goal_idx]
        
        path = space_time_astar(grid, start, goal, reservation_table, edge_reservations)
        
        if not path:
            print(f"Agent {agent_id} could not find a path to its goal!")
        else:
            paths[agent_id] = path
            # Add to reservation table
            for i, (pos, t) in enumerate(path):
                r, c = pos
                reservation_table.add((r, c, t))
                # Add edge reservation
                if i > 0:
                    prev_pos, prev_t = path[i-1]
                    pr, pc = prev_pos
                    # We moved from prev_pos to pos at time prev_t
                    # Another agent cannot move from pos to prev_pos at time prev_t
                    edge_reservations.add((pr, pc, r, c, prev_t))
            
            # The agent stays at the goal forever
            final_pos, final_t = path[-1]
            for k in range(final_t + 1, MAX_TIME_LIMIT + 1):
                reservation_table.add((final_pos[0], final_pos[1], k))

    # 5. Plot
    plot_multi_agent_result(
        grid=grid,
        starts=starts,
        goals=[goals[assignment[i]] for i in range(N)],
        paths=paths,
        animate=args.animate,
        interval_ms=args.interval,
        save_path=args.save,
        show=args.show,
    )

if __name__ == "__main__":
    main()
