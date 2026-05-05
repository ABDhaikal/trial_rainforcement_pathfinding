"""
environment.py
==============
Environment untuk Assignment Problem 3x3 menggunakan DQN.
Tujuan: Minimasi total cost dari matriks 3x3 yang berubah per episode.

Action space : 6 permutasi assignment (3! = 6)
State space  : 9 nilai flatten dari matriks cost 3x3
Reward       : -total_cost (negatif agar minimize = maximize reward)
"""

import numpy as np
import itertools


# Semua permutasi assignment untuk 3 agent → 3 task
ALL_PERMUTATIONS = list(itertools.permutations(range(3)))  # 6 permutasi


class AssignmentEnv:
    """
    Environment assignment problem matriks 3x3.

    State  : flatten matriks cost → shape (9,)
    Action : index 0-5, masing-masing mewakili satu permutasi assignment
    Reward : -total_cost dari assignment yang dipilih
    Done   : True setelah satu aksi (single-step episode)
    """

    def __init__(self, cost_low=1, cost_high=10, seed=None):
        """
        Parameters
        ----------
        cost_low  : int, batas bawah nilai cost matriks
        cost_high : int, batas atas nilai cost matriks
        seed      : int opsional, untuk reproducibility
        """
        self.cost_low  = cost_low
        self.cost_high = cost_high
        self.n_agents  = 3
        self.n_tasks   = 3
        self.n_actions = len(ALL_PERMUTATIONS)   # 6
        self.state_dim = self.n_agents * self.n_tasks  # 9

        self.rng          = np.random.default_rng(seed)
        self.cost_matrix  = None
        self.current_state = None
        self.done         = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset environment: generate matriks cost baru.
        Returns
        -------
        state : np.ndarray shape (9,), dtype float32
        """
        self.cost_matrix = self.rng.integers(
            self.cost_low, self.cost_high + 1,
            size=(self.n_agents, self.n_tasks)
        ).astype(np.float32)

        self.current_state = self.cost_matrix.flatten()
        self.done = False
        return self.current_state.copy()

    def step(self, action):
        """
        Eksekusi satu aksi assignment.

        Parameters
        ----------
        action : int, index 0-5

        Returns
        -------
        next_state : np.ndarray (9,)  — sama dengan state (single-step)
        reward     : float
        done       : bool (selalu True)
        info       : dict berisi detail assignment
        """
        assert not self.done, "Episode sudah selesai. Panggil reset() dulu."
        assert 0 <= action < self.n_actions, f"Action {action} tidak valid."

        assignment = ALL_PERMUTATIONS[action]  # tuple (task_for_agent0, task_for_agent1, task_for_agent2)
        total_cost = sum(
            self.cost_matrix[agent][task]
            for agent, task in enumerate(assignment)
        )

        reward = -float(total_cost)
        self.done = True

        info = {
            "assignment"    : assignment,
            "total_cost"    : float(total_cost),
            "cost_matrix"   : self.cost_matrix.copy(),
            "optimal_cost"  : self._optimal_cost(),
            "optimal_action": self._optimal_action(),
        }

        return self.current_state.copy(), reward, self.done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _optimal_cost(self):
        """Hitung cost minimum dari semua permutasi (brute force, valid untuk 3x3)."""
        min_cost = float("inf")
        for perm in ALL_PERMUTATIONS:
            cost = sum(self.cost_matrix[a][t] for a, t in enumerate(perm))
            if cost < min_cost:
                min_cost = cost
        return float(min_cost)

    def _optimal_action(self):
        """Return action index dengan cost minimum."""
        best_idx, best_cost = 0, float("inf")
        for idx, perm in enumerate(ALL_PERMUTATIONS):
            cost = sum(self.cost_matrix[a][t] for a, t in enumerate(perm))
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        return best_idx

    def action_to_assignment(self, action):
        """Konversi action index → tuple assignment."""
        return ALL_PERMUTATIONS[action]

    def render(self):
        """Tampilkan matriks cost saat ini ke console."""
        print("\n=== Cost Matrix ===")
        print(f"{'':>10} Task0  Task1  Task2")
        for i, row in enumerate(self.cost_matrix):
            print(f"  Agent{i}  {row[0]:>5.1f}  {row[1]:>5.1f}  {row[2]:>5.1f}")
        print()

    def get_all_permutations(self):
        """Return semua permutasi dengan cost-nya (berguna untuk analisis)."""
        results = []
        for idx, perm in enumerate(ALL_PERMUTATIONS):
            cost = sum(self.cost_matrix[a][t] for a, t in enumerate(perm))
            results.append({"action": idx, "assignment": perm, "cost": float(cost)})
        return sorted(results, key=lambda x: x["cost"])


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    env = AssignmentEnv(seed=42)
    state = env.reset()
    env.render()

    print(f"State (flatten): {state}")
    print(f"Semua permutasi beserta cost:")
    for item in env.get_all_permutations():
        print(f"  Action {item['action']} → {item['assignment']}  cost={item['cost']:.1f}")

    # Coba aksi terbaik
    best_action = env._optimal_action()
    next_state, reward, done, info = env.step(best_action)
    print(f"\nAksi terbaik: {best_action} → assignment {info['assignment']}")
    print(f"Total cost  : {info['total_cost']:.1f}")
    print(f"Optimal cost: {info['optimal_cost']:.1f}")
    print(f"Reward      : {reward:.1f}")
