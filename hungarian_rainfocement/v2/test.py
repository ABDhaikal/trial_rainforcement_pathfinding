"""
test.py
=======
Testing & Evaluasi DQN untuk Assignment Problem 3x3.

Fitur:
- Load model dari file .pth hasil training
- Evaluasi pada N episode test
- Bandingkan dengan Hungarian Method (scipy)
- Export hasil test ke CSV
- Export detail per-episode ke CSV
- Print ringkasan performa
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment  # Hungarian Method

from environment import AssignmentEnv, ALL_PERMUTATIONS

# ============================================================
# Konfigurasi Testing
# ============================================================
TEST_CONFIG = {
    "model_path"         : "dqn_model.pth",
    "config_path"        : "config.json",
    "num_test_episodes"  : 500,
    "cost_low"           : 1,
    "cost_high"          : 10,
    "seed"               : 123,          # seed berbeda dari training
    "output_detail_csv"  : "test_detail.csv",
    "output_summary_csv" : "test_summary.csv",
    "verbose_episodes"   : 10,           # print detail N episode pertama
}


# ============================================================
# DQN Network (harus identik dengan train.py)
# ============================================================
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


# ============================================================
# Load Model
# ============================================================
def load_model(model_path, config_path=None):
    """Load DQN model dari checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ambil hidden_size dari config jika ada
    hidden_size = 64
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        hidden_size = cfg.get("hidden_size", 64)
        print(f"[Config] Loaded dari {config_path}")

    model = DQNetwork(state_dim=9, action_dim=6, hidden_size=hidden_size).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["policy_net_state_dict"])
    model.eval()

    epsilon = checkpoint.get("epsilon", 0.0)
    print(f"[Model] Loaded dari: {model_path}")
    print(f"[Model] Epsilon saat disimpan: {epsilon:.4f}")
    return model, device


# ============================================================
# Hungarian Method (baseline)
# ============================================================
def hungarian_solve(cost_matrix):
    """
    Selesaikan assignment problem dengan Hungarian Method.
    Returns (assignment_tuple, total_cost)
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = tuple(col_ind)  # col_ind[i] = task untuk agent i
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return assignment, float(total_cost)


# ============================================================
# DQN Predict
# ============================================================
def dqn_predict(model, state, device):
    """
    Greedy prediction dari model DQN.
    Returns (action, q_values)
    """
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_t).squeeze().cpu().numpy()
    action = int(np.argmax(q_values))
    return action, q_values


# ============================================================
# Main Testing
# ============================================================
def test():
    print("\n" + "="*60)
    print("  Testing DQN — Assignment Problem 3x3")
    print("="*60)

    # Validasi file model
    if not os.path.isfile(TEST_CONFIG["model_path"]):
        print(f"[ERROR] Model tidak ditemukan: {TEST_CONFIG['model_path']}")
        print("  Jalankan train.py terlebih dahulu.")
        return

    # Load model
    model, device = load_model(TEST_CONFIG["model_path"], TEST_CONFIG["config_path"])
    print(f"[Device] {device}\n")

    # Environment
    env = AssignmentEnv(
        cost_low  = TEST_CONFIG["cost_low"],
        cost_high = TEST_CONFIG["cost_high"],
        seed      = TEST_CONFIG["seed"],
    )

    # Tracking
    results = []
    dqn_costs      = []
    hungarian_costs = []
    optimal_counts = {"dqn": 0, "hungarian": 0}
    match_count    = 0  # berapa kali DQN == Hungarian

    # Header CSV detail
    with open(TEST_CONFIG["output_detail_csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "cost_matrix",
            "dqn_action", "dqn_assignment", "dqn_cost",
            "hungarian_assignment", "hungarian_cost",
            "optimal_cost",
            "dqn_is_optimal", "hungarian_is_optimal",
            "dqn_vs_hungarian_gap",
            "q_values",
        ])

    print(f"{'Ep':>5} | {'DQN Cost':>9} | {'Hungarian':>9} | {'Optimal':>8} | {'DQN Opt?':>8} | {'Gap':>6}")
    print("-" * 60)

    for episode in range(1, TEST_CONFIG["num_test_episodes"] + 1):
        state = env.reset()

        # DQN prediction
        dqn_action, q_values = dqn_predict(model, state, device)
        dqn_assignment = ALL_PERMUTATIONS[dqn_action]
        dqn_cost = float(sum(env.cost_matrix[a][t] for a, t in enumerate(dqn_assignment)))

        # Hungarian prediction
        hun_assignment, hun_cost = hungarian_solve(env.cost_matrix)

        # Optimal (brute force)
        optimal_cost = env._optimal_cost()

        dqn_is_optimal = abs(dqn_cost - optimal_cost) < 1e-6
        hun_is_optimal = abs(hun_cost  - optimal_cost) < 1e-6
        gap = dqn_cost - hun_cost

        if dqn_is_optimal: optimal_counts["dqn"] += 1
        if hun_is_optimal: optimal_counts["hungarian"] += 1
        if dqn_assignment == hun_assignment: match_count += 1

        dqn_costs.append(dqn_cost)
        hungarian_costs.append(hun_cost)

        # Simpan ke CSV detail
        with open(TEST_CONFIG["output_detail_csv"], "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                str(env.cost_matrix.tolist()),
                dqn_action,
                str(dqn_assignment),
                dqn_cost,
                str(hun_assignment),
                hun_cost,
                optimal_cost,
                int(dqn_is_optimal),
                int(hun_is_optimal),
                round(gap, 4),
                str([round(q, 4) for q in q_values]),
            ])

        # Verbose print
        if episode <= TEST_CONFIG["verbose_episodes"]:
            opt_mark = "✓" if dqn_is_optimal else "✗"
            print(f"{episode:>5} | {dqn_cost:>9.2f} | {hun_cost:>9.2f} | "
                  f"{optimal_cost:>8.2f} | {opt_mark:>8} | {gap:>+6.2f}")

    if TEST_CONFIG["num_test_episodes"] > TEST_CONFIG["verbose_episodes"]:
        print(f"  ... (total {TEST_CONFIG['num_test_episodes']} episode)")

    # ============================================================
    # Ringkasan
    # ============================================================
    n = TEST_CONFIG["num_test_episodes"]
    dqn_opt_rate  = optimal_counts["dqn"] / n * 100
    hun_opt_rate  = optimal_counts["hungarian"] / n * 100
    match_rate    = match_count / n * 100

    avg_dqn_cost  = np.mean(dqn_costs)
    avg_hun_cost  = np.mean(hungarian_costs)
    avg_gap       = avg_dqn_cost - avg_hun_cost
    avg_gap_pct   = avg_gap / avg_hun_cost * 100 if avg_hun_cost > 0 else 0

    summary = {
        "num_test_episodes"   : n,
        "dqn_optimal_count"   : optimal_counts["dqn"],
        "dqn_optimal_rate_pct": round(dqn_opt_rate, 2),
        "hun_optimal_count"   : optimal_counts["hungarian"],
        "hun_optimal_rate_pct": round(hun_opt_rate, 2),
        "match_count"         : match_count,
        "match_rate_pct"      : round(match_rate, 2),
        "avg_dqn_cost"        : round(avg_dqn_cost, 4),
        "avg_hungarian_cost"  : round(avg_hun_cost, 4),
        "avg_cost_gap"        : round(avg_gap, 4),
        "avg_cost_gap_pct"    : round(avg_gap_pct, 2),
        "min_dqn_cost"        : round(float(np.min(dqn_costs)), 4),
        "max_dqn_cost"        : round(float(np.max(dqn_costs)), 4),
        "std_dqn_cost"        : round(float(np.std(dqn_costs)), 4),
    }

    # Simpan ringkasan CSV
    with open(TEST_CONFIG["output_summary_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    # Print ringkasan
    print("\n" + "="*60)
    print("  RINGKASAN HASIL TESTING")
    print("="*60)
    print(f"  Total Episode          : {n}")
    print(f"")
    print(f"  === DQN ===")
    print(f"  Optimal Rate           : {optimal_counts['dqn']}/{n}  ({dqn_opt_rate:.1f}%)")
    print(f"  Avg Cost               : {avg_dqn_cost:.4f}")
    print(f"  Min / Max Cost         : {np.min(dqn_costs):.2f} / {np.max(dqn_costs):.2f}")
    print(f"  Std Cost               : {np.std(dqn_costs):.4f}")
    print(f"")
    print(f"  === Hungarian Method (Baseline) ===")
    print(f"  Optimal Rate           : {optimal_counts['hungarian']}/{n}  ({hun_opt_rate:.1f}%)")
    print(f"  Avg Cost               : {avg_hun_cost:.4f}")
    print(f"")
    print(f"  === Perbandingan ===")
    print(f"  DQN vs Hungarian Match : {match_count}/{n}  ({match_rate:.1f}%)")
    print(f"  Avg Cost Gap           : {avg_gap:+.4f}  ({avg_gap_pct:+.2f}%)")
    print(f"  (positif = DQN lebih mahal dari Hungarian)")
    print(f"")
    print(f"  Output Detail          : {TEST_CONFIG['output_detail_csv']}")
    print(f"  Output Summary         : {TEST_CONFIG['output_summary_csv']}")
    print("="*60)

    # Demo interaktif: uji satu matriks manual
    print("\n" + "="*60)
    print("  DEMO: Matriks Custom")
    print("="*60)
    custom_matrix = np.array([
        [10, 2, 8],
        [3, 2, 10],
        [6, 5, 30],
    ], dtype=np.float32)

    print("  Cost Matrix:")
    print(f"  {'':>8} Task0  Task1  Task2")
    for i, row in enumerate(custom_matrix):
        print(f"  Agent{i}   {row[0]:>5.1f}  {row[1]:>5.1f}  {row[2]:>5.1f}")

    state_custom = custom_matrix.flatten()
    action_custom, q_vals_custom = dqn_predict(model, state_custom, device)
    dqn_assign_custom = ALL_PERMUTATIONS[action_custom]
    dqn_cost_custom = sum(custom_matrix[a][t] for a, t in enumerate(dqn_assign_custom))

    hun_assign_custom, hun_cost_custom = hungarian_solve(custom_matrix)

    print(f"\n  DQN Assignment      : {dqn_assign_custom}  →  Cost = {dqn_cost_custom:.1f}")
    print(f"  Hungarian Assignment: {hun_assign_custom}  →  Cost = {hun_cost_custom:.1f}")
    print(f"\n  Q-values per action:")
    for i, (perm, qv) in enumerate(zip(ALL_PERMUTATIONS, q_vals_custom)):
        marker = " ← DQN pilih" if i == action_custom else ""
        print(f"    Action {i} {perm}  Q={qv:>8.4f}{marker}")
    print("="*60)


if __name__ == "__main__":
    test()
