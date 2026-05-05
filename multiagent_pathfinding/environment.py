"""
environment.py — Grid 10x10 Multi-Agent Path Planning Environment
CTDE (Centralized Training, Decentralized Execution)

Grid layout:
  0 = free cell
  1 = obstacle
  Agent positions tracked separately (not encoded in grid)

Observations (local, per agent):
  - 5x5 FOV around agent (flattened, 25 values): 0=free, 1=obstacle, 2=other agent
  - own position normalized (2 values)
  - goal position normalized (2 values)
  Total: 29 values per agent

Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
"""

import sys
import os
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate_map import generate_warehouse_map, MapConfig

FOV_RADIUS   = 2          # 5x5 window around agent
OBS_SIZE     = (2*FOV_RADIUS+1)**2 + 4   # 25 + 4 = 29

ACTION_DELTAS = {
    0: (0,  0),   # stay
    1: (-1, 0),   # up
    2: ( 1, 0),   # down
    3: (0, -1),   # left
    4: (0,  1),   # right
}
N_ACTIONS = len(ACTION_DELTAS)

# Reward shaping
R_GOAL        =  10.0
R_COLLISION   =  -5.0
R_STEP        =  -0.1
R_APPROACH    =   0.2   # per unit closer to goal


class MultiAgentGridEnv:
    def __init__(self, n_agents: int = 4, seed: int | None = None, map_config: MapConfig = None):
        self.n_agents   = n_agents
        self.obs_size   = OBS_SIZE
        self.n_actions  = N_ACTIONS
        self.rng        = np.random.default_rng(seed)
        self._base_seed = seed

        if map_config is None:
            map_config = MapConfig()  # Default warehouse map

        self.grid, _ = generate_warehouse_map(map_config)
        self.grid_height, self.grid_width = self.grid.shape

        self.agent_pos   = []
        self.agent_goals = []
        self.done_flags  = []
        self.steps       = 0
        self.max_steps   = 200

        self._set_fixed_goals()

    # ------------------------------------------------------------------ #
    #  Setup helpers
    # ------------------------------------------------------------------ #

    def _set_fixed_goals(self):
        free = self._free_cells()
        self.rng.shuffle(free)
        self.agent_goals = list(free[:self.n_agents])

    def _free_cells(self):
        return [(r, c) for r in range(self.grid_height)
                        for c in range(self.grid_width)
                        if self.grid[r, c] == 0]

    def reset(self):
        self.steps = 0
        free = self._free_cells()
        # Avoid spawning on goals
        free = [c for c in free if c not in self.agent_goals]
        self.rng.shuffle(free)

        self.agent_pos   = list(free[:self.n_agents])
        self.done_flags  = [False] * self.n_agents

        return self._get_observations()

    # ------------------------------------------------------------------ #
    #  Core step
    # ------------------------------------------------------------------ #

    def step(self, actions):
        """
        actions: list of ints, length n_agents
        Returns:
            obs       : list of np.ndarray (29,)
            rewards   : list of float
            dones     : list of bool
            info      : dict
        """
        assert len(actions) == self.n_agents
        rewards   = [0.0] * self.n_agents
        new_pos   = list(self.agent_pos)

        # Compute intended new positions
        for i, a in enumerate(actions):
            if self.done_flags[i]:
                rewards[i] = 0.0
                continue
            dr, dc = ACTION_DELTAS[a]
            nr = self.agent_pos[i][0] + dr
            nc = self.agent_pos[i][1] + dc

            # Wall / obstacle collision
            if not (0 <= nr < self.grid_height and 0 <= nc < self.grid_width):
                rewards[i] += R_COLLISION
                nr, nc = self.agent_pos[i]   # stay
            elif self.grid[nr, nc] == 1:
                rewards[i] += R_COLLISION
                nr, nc = self.agent_pos[i]   # stay

            new_pos[i] = (nr, nc)

        # Agent-agent collision: if two agents want same cell → both stay
        for i in range(self.n_agents):
            if self.done_flags[i]:
                continue
            for j in range(i+1, self.n_agents):
                if self.done_flags[j]:
                    continue
                if new_pos[i] == new_pos[j]:
                    rewards[i] += R_COLLISION
                    rewards[j] += R_COLLISION
                    new_pos[i] = self.agent_pos[i]
                    new_pos[j] = self.agent_pos[j]

        # Apply moves, compute per-step rewards
        for i in range(self.n_agents):
            if self.done_flags[i]:
                continue

            prev_dist = self._dist(self.agent_pos[i], self.agent_goals[i])
            self.agent_pos[i] = new_pos[i]
            new_dist  = self._dist(self.agent_pos[i], self.agent_goals[i])

            rewards[i] += R_STEP
            rewards[i] += R_APPROACH * (prev_dist - new_dist)

            if self.agent_pos[i] == self.agent_goals[i]:
                rewards[i] += R_GOAL
                self.done_flags[i] = True

        self.steps += 1
        timeout = self.steps >= self.max_steps
        global_done = all(self.done_flags) or timeout

        dones = [d or timeout for d in self.done_flags]
        info  = {
            "timeout"     : timeout,
            "reached_goal": sum(self.done_flags),
            "steps"       : self.steps,
        }
        return self._get_observations(), rewards, dones, global_done, info

    # ------------------------------------------------------------------ #
    #  Observation builder
    # ------------------------------------------------------------------ #

    def _get_observations(self):
        return [self._obs_for(i) for i in range(self.n_agents)]

    def _obs_for(self, agent_idx):
        r, c = self.agent_pos[agent_idx]
        fov = np.zeros((2*FOV_RADIUS+1, 2*FOV_RADIUS+1), dtype=np.float32)

        for dr in range(-FOV_RADIUS, FOV_RADIUS+1):
            for dc in range(-FOV_RADIUS, FOV_RADIUS+1):
                nr, nc = r+dr, c+dc
                fr, fc = dr+FOV_RADIUS, dc+FOV_RADIUS
                if not (0 <= nr < self.grid_height and 0 <= nc < self.grid_width):
                    fov[fr, fc] = 1.0   # treat out-of-bounds as obstacle
                elif self.grid[nr, nc] == 1:
                    fov[fr, fc] = 1.0
                elif any((nr, nc) == self.agent_pos[j]
                         for j in range(self.n_agents) if j != agent_idx):
                    fov[fr, fc] = 2.0

        gr, gc = self.agent_goals[agent_idx]
        own_norm  = np.array([r  / (self.grid_height-1), c  / (self.grid_width-1)], dtype=np.float32)
        goal_norm = np.array([gr / (self.grid_height-1), gc / (self.grid_width-1)], dtype=np.float32)

        return np.concatenate([fov.flatten(), own_norm, goal_norm])

    # ------------------------------------------------------------------ #
    #  Global state (for central critic)
    # ------------------------------------------------------------------ #

    def get_global_state(self):
        """
        Returns a flat vector of shape:
          grid_flat + agent_positions_norm (2*n) + agent_goals_norm (2*n)
        """
        grid_flat = self.grid.flatten().astype(np.float32)
        pos_flat  = np.array(
            [val for pos in self.agent_pos for val in (pos[0] / (self.grid_height-1), pos[1] / (self.grid_width-1))],
            dtype=np.float32)
        goal_flat = np.array(
            [val for g in self.agent_goals for val in (g[0] / (self.grid_height-1), g[1] / (self.grid_width-1))],
            dtype=np.float32)
        return np.concatenate([grid_flat, pos_flat, goal_flat])

    @property
    def global_state_size(self):
        return self.grid_height * self.grid_width + 4 * self.n_agents

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dist(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])   # Manhattan

    def render_ascii(self):
        """Quick text render for debugging."""
        grid = [list("." * self.grid_width) for _ in range(self.grid_height)]
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if self.grid[r, c] == 1:
                    grid[r][c] = "#"
        for i, (r, c) in enumerate(self.agent_goals):
            grid[r][c] = str(i).lower()
        for i, (r, c) in enumerate(self.agent_pos):
            grid[r][c] = str(i).upper()
        print(f"Step {self.steps}")
        for row in grid:
            print(" ".join(row))
        print()
