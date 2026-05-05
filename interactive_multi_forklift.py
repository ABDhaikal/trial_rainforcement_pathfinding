"""Interactive A* pathfinding for multiple forklifts using Hungarian Method or DQN.

Allows the user to manually click to place forklifts, goals, and obstacles, 
or randomize the layout before calculating and animating the optimal paths.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons

# Adjust import to work with local generate_map
try:
    from generate_map import MapConfig, generate_warehouse_map
except ImportError:
    pass

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


GridPos = Tuple[int, int]
MAX_TIME_LIMIT = 500
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# =====================================================================
# DQN Network
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
        model.load_state_dict(checkpoint)
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
    grid: np.ndarray, r: int, c: int, t: int,
    reservation_table: Set[Tuple[int, int, int]],
    edge_reservations: Set[Tuple[int, int, int, int, int]]
) -> List[Tuple[int, int, int]]:
    rows, cols = grid.shape
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1), (r, c)]
    valid = []
    for nr, nc in candidates:
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            nt = t + 1
            if nt > MAX_TIME_LIMIT:
                continue
            if (nr, nc, nt) in reservation_table:
                continue
            if (nr, nc, r, c, t) in edge_reservations:
                continue
            valid.append((nr, nc, nt))
    return valid

def is_goal_safe(goal: GridPos, t: int, reservation_table: Set[Tuple[int, int, int]]) -> bool:
    for k in range(t, MAX_TIME_LIMIT + 1):
        if (goal[0], goal[1], k) in reservation_table:
            return False
    return True

def space_time_astar(
    grid: np.ndarray, start: GridPos, goal: GridPos,
    reservation_table: Set[Tuple[int, int, int]],
    edge_reservations: Set[Tuple[int, int, int, int, int]]
) -> List[Tuple[GridPos, int]]:
    if grid[start] != 0 or grid[goal] != 0:
        return []
    
    open_heap: List[Tuple[int, int, int, int, int]] = []
    counter = 0
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
# Interactive GUI
# =====================================================================
class InteractivePathfinder:
    def __init__(self, grid: np.ndarray, assign_method: str, dqn_model: str, max_agents: int = 3, interval_ms: int = 150):
        self.grid = grid
        self.assign_method = assign_method
        self.dqn_model = dqn_model
        self.max_agents = max_agents
        self.interval_ms = interval_ms
        
        self.starts: List[GridPos] = []
        self.goals: List[GridPos] = []
        self.placement_mode = 'Forklift'
        self.state = 'EDITING'
        
        self.cmap = ListedColormap(["#f5f5f5", "#2f4f4f"])
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        # Make room for controls on the left and bottom
        plt.subplots_adjust(left=0.25, bottom=0.2)
        
        self.img = self.ax.imshow(self.grid, cmap=self.cmap, interpolation="none", origin="upper", vmin=0, vmax=1)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        self.ax.grid(which="minor", color="#c8c8c8", linestyle="-", linewidth=0.4)
        
        # Scatters for manual placement
        self.start_scatter = self.ax.scatter([], [], s=80, marker="o", edgecolors='black', zorder=5)
        self.goal_scatter = self.ax.scatter([], [], s=120, marker="*", edgecolors='black', zorder=5)
        
        # Lines and dots for animation
        self.path_lines = []
        self.forklift_dots = []
        for i in range(self.max_agents):
            color = COLORS[i % len(COLORS)]
            pline, = self.ax.plot([], [], color=color, linewidth=2.5, alpha=0.7, zorder=3)
            fdot, = self.ax.plot([], [], marker="s", color=color, markersize=10, markeredgecolor='black', zorder=6)
            self.path_lines.append(pline)
            self.forklift_dots.append(fdot)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Controls
        ax_radio = plt.axes([0.05, 0.4, 0.15, 0.15], facecolor='lightgoldenrodyellow')
        self.radio = RadioButtons(ax_radio, ('Forklift', 'Goal', 'Obstacle'))
        self.radio.on_clicked(self.on_mode_change)
        
        ax_start = plt.axes([0.3, 0.05, 0.15, 0.075])
        self.btn_start = Button(ax_start, 'Start')
        self.btn_start.on_clicked(self.on_start)
        
        ax_random = plt.axes([0.5, 0.05, 0.15, 0.075])
        self.btn_random = Button(ax_random, 'Random')
        self.btn_random.on_clicked(self.on_random)
        
        ax_reset = plt.axes([0.7, 0.05, 0.15, 0.075])
        self.btn_reset = Button(ax_reset, 'Clear All')
        self.btn_reset.on_clicked(self.on_reset)
        
        self.anim = None
        self.update_title()

    def on_mode_change(self, label):
        self.placement_mode = label
        self.update_title()

    def update_title(self):
        s_count = len(self.starts)
        g_count = len(self.goals)
        
        if self.state == 'COMPUTING':
            self.ax.set_title("Computing Paths...")
        elif self.state == 'ANIMATING':
            pass # Keep animation title
        else:
            self.ax.set_title(f"Mode: {self.placement_mode} | Forklifts: {s_count}/{self.max_agents} | Goals: {g_count}/{self.max_agents}")
        self.fig.canvas.draw_idle()

    def _update_scatter(self):
        # Update starts
        if self.starts:
            sr, sc = zip(*self.starts)
            colors = [COLORS[i % len(COLORS)] for i in range(len(self.starts))]
            self.start_scatter.set_offsets(np.column_stack((sc, sr)))
            self.start_scatter.set_facecolors(colors)
        else:
            self.start_scatter.set_offsets(np.empty((0, 2)))
            
        # Update goals
        if self.goals:
            gr, gc = zip(*self.goals)
            self.goal_scatter.set_offsets(np.column_stack((gc, gr)))
            self.goal_scatter.set_facecolors('gray') # Will be colored after assignment
        else:
            self.goal_scatter.set_offsets(np.empty((0, 2)))
            
        self.fig.canvas.draw_idle()
        self.update_title()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        c = int(round(event.xdata))
        r = int(round(event.ydata))
        
        rows, cols = self.grid.shape
        if not (0 <= r < rows and 0 <= c < cols):
            return
            
        pos = (r, c)
        
        # Stop animation if we click the grid
        if self.anim:
            if self.anim.event_source:
                self.anim.event_source.stop()
            self.anim = None
            self.state = 'EDITING'
            for pline in self.path_lines: pline.set_data([], [])
            for fdot in self.forklift_dots: fdot.set_data([], [])
        
        if self.placement_mode == 'Obstacle':
            if pos in self.starts or pos in self.goals:
                return # Cannot place obstacle over start/goal
            if self.grid[r, c] == 1:
                self.grid[r, c] = 0 # Remove
            else:
                self.grid[r, c] = 1 # Add
            self.img.set_data(self.grid)
            self.fig.canvas.draw_idle()
            
        elif self.placement_mode == 'Forklift':
            if self.grid[r, c] == 1:
                return # Cannot place over obstacle
            if pos in self.starts:
                self.starts.remove(pos)
            else:
                if len(self.starts) < self.max_agents and pos not in self.goals:
                    self.starts.append(pos)
            self._update_scatter()
            
        elif self.placement_mode == 'Goal':
            if self.grid[r, c] == 1:
                return # Cannot place over obstacle
            if pos in self.goals:
                self.goals.remove(pos)
            else:
                if len(self.goals) < self.max_agents and pos not in self.starts:
                    self.goals.append(pos)
            self._update_scatter()

    def on_reset(self, event):
        if self.anim:
            if self.anim.event_source:
                self.anim.event_source.stop()
            self.anim = None
            
        self.starts.clear()
        self.goals.clear()
        self.grid.fill(0)
        self.img.set_data(self.grid)
        self.state = 'EDITING'
        
        for pline in self.path_lines:
            pline.set_data([], [])
        for fdot in self.forklift_dots:
            fdot.set_data([], [])
            
        self._update_scatter()

    def on_random(self, event):
        if self.anim:
            if self.anim.event_source:
                self.anim.event_source.stop()
            self.anim = None
            
        self.starts.clear()
        self.goals.clear()
        
        for pline in self.path_lines:
            pline.set_data([], [])
        for fdot in self.forklift_dots:
            fdot.set_data([], [])
            
        # Clear grid
        self.grid.fill(0)
        rows, cols = self.grid.shape
        
        # Randomize obstacles (~15% of the board)
        num_obstacles = int(rows * cols * 0.15)
        for _ in range(num_obstacles):
            r = int(np.random.randint(0, rows))
            c = int(np.random.randint(0, cols))
            self.grid[r, c] = 1
            
        # Helper to get random free spot
        def get_random_free():
            for _ in range(100):
                r = int(np.random.randint(0, rows))
                c = int(np.random.randint(0, cols))
                if self.grid[r, c] == 0 and (r, c) not in self.starts and (r, c) not in self.goals:
                    return (r, c)
            return None # Should rarely happen on empty board
            
        # Place 3 starts
        for _ in range(self.max_agents):
            pos = get_random_free()
            if pos: self.starts.append(pos)
            
        # Place 3 goals
        for _ in range(self.max_agents):
            pos = get_random_free()
            if pos: self.goals.append(pos)
            
        self.state = 'EDITING'
        self.img.set_data(self.grid)
        self._update_scatter()

    def on_start(self, event):
        if len(self.starts) != self.max_agents or len(self.goals) != self.max_agents:
            print("Please place exactly 3 starts and 3 goals first.")
            return
            
        self.state = 'COMPUTING'
        self.update_title()
        
        N = self.max_agents
        
        # 1. Cost Matrix
        cost_matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                cost = astar_cost(self.grid, self.starts[i], self.goals[j])
                cost_matrix[i, j] = cost
                
        print("Cost Matrix:\n", cost_matrix)
        
        # 2. Assignment
        if self.assign_method == "rl" and N == 3:
            if not os.path.isfile(self.dqn_model):
                print(f"Error: DQN model not found at {self.dqn_model}")
                self.ax.set_title("Error: DQN Model not found.")
                self.state = 'EDITING'
                return
            assignment, total_cost = assign_goals_rl(cost_matrix, self.dqn_model)
            print(f"RL Assignment: {assignment}")
        else:
            assignment, total_cost = assign_goals_hungarian(cost_matrix)
            print(f"Hungarian Assignment: {assignment}")
            
        # Color goals based on assignment
        gr, gc = zip(*self.goals)
        goal_colors = [COLORS[list(assignment).index(i) % len(COLORS)] for i in range(N)]
        self.goal_scatter.set_facecolors(goal_colors)
        self.fig.canvas.draw_idle()

        # 3. Space-Time A*
        reservation_table: Set[Tuple[int, int, int]] = set()
        edge_reservations: Set[Tuple[int, int, int, int, int]] = set()
        paths: List[List[Tuple[GridPos, int]]] = [[] for _ in range(N)]
        
        success = True
        for agent_id in range(N):
            start = self.starts[agent_id]
            goal_idx = assignment[agent_id]
            goal = self.goals[goal_idx]
            
            path = space_time_astar(self.grid, start, goal, reservation_table, edge_reservations)
            
            if not path:
                print(f"Agent {agent_id} could not find a path to its goal!")
                success = False
            else:
                paths[agent_id] = path
                for i, (pos, t) in enumerate(path):
                    r, c = pos
                    reservation_table.add((r, c, t))
                    if i > 0:
                        prev_pos, prev_t = path[i-1]
                        pr, pc = prev_pos
                        edge_reservations.add((pr, pc, r, c, prev_t))
                
                final_pos, final_t = path[-1]
                for k in range(final_t + 1, MAX_TIME_LIMIT + 1):
                    reservation_table.add((final_pos[0], final_pos[1], k))

        if not success:
            self.ax.set_title(f"Failed to find paths. Some goals might be blocked.")
            self.state = 'EDITING'
            return

        # 4. Animation
        self.state = 'ANIMATING'
        self.ax.set_title(f"Animating ({self.assign_method.upper()}) | Cost: {total_cost:.1f}")
        
        max_len = max(len(p) for p in paths) if any(paths) else 0

        def update(frame: int):
            for i, path in enumerate(paths):
                if not path:
                    continue
                f_idx = min(frame, len(path) - 1)
                
                path_r = [p[0][0] for p in path[:f_idx + 1]]
                path_c = [p[0][1] for p in path[:f_idx + 1]]
                
                self.path_lines[i].set_data(path_c, path_r)
                self.forklift_dots[i].set_data([path_c[-1]], [path_r[-1]])
                
            return self.path_lines + self.forklift_dots

        if max_len > 0:
            if self.anim:
                if self.anim.event_source:
                    self.anim.event_source.stop()
            self.anim = FuncAnimation(
                self.fig, update, frames=max_len,
                interval=self.interval_ms, blit=True, repeat=False
            )
            self.fig.canvas.draw_idle()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Multi-agent pathfinding.")

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
    parser.add_argument("--dqn-model", type=str, default="hungarian_rainfocement/v2/dqn_model.pth", help="Path to DQN model.")
    parser.add_argument("--interval", type=int, default=150, help="Animation interval in milliseconds.")

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Use generate_map just for empty grid
    config = MapConfig(width=args.width, height=args.height, rack_count=0)
    grid, _ = generate_warehouse_map(config)
    print(f"Map size: {args.width}x{args.height}. Click to edit.")

    app = InteractivePathfinder(
        grid=grid,
        assign_method=args.assign_method,
        dqn_model=args.dqn_model,
        max_agents=args.forklifts,
        interval_ms=args.interval
    )
    
    plt.show()

if __name__ == "__main__":
    main()
