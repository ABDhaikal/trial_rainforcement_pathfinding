"""
train.py
========
Training DQN untuk Assignment Problem 3x3.

Fitur:
- Flat DQN (FC layers) dengan Experience Replay & Target Network
- Epsilon-greedy exploration
- Export model (.pth)
- Export Q-table per episode ke CSV
- Export ringkasan training ke CSV
- Plot kurva training (loss & reward)
"""

import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

from environment import AssignmentEnv, ALL_PERMUTATIONS

# ============================================================
# Konfigurasi Hyperparameter
# ============================================================
CONFIG = {
    # Training
    "num_episodes"      : 2000,
    "max_steps"         : 1,          # single-step episode
    "batch_size"        : 64,
    "replay_buffer_size": 5000,

    # DQN
    "gamma"             : 0.99,       # discount factor (tidak terlalu berpengaruh di single-step)
    "lr"                : 1e-3,
    "hidden_size"       : 64,

    # Epsilon-greedy
    "epsilon_start"     : 1.0,
    "epsilon_end"       : 0.01,
    "epsilon_decay"     : 0.995,

    # Target network update
    "target_update_freq": 50,         # setiap N episode

    # Export
    "save_model_path"   : "dqn_model.pth",
    "save_qtable_path"  : "qtable_export.csv",
    "save_log_path"     : "training_log.csv",
    "save_config_path"  : "config.json",
    "plot_path"         : "training_plot.png",

    # Environment
    "cost_low"          : 1,
    "cost_high"         : 10,
    "seed"              : 42,

    # Logging
    "log_interval"      : 100,        # print setiap N episode
    "qtable_log_interval": 200,       # simpan Q-table snapshot setiap N episode
}

# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN Network
# ============================================================
class DQNetwork(nn.Module):
    """
    Flat DQN: input 9 nilai cost → output 6 Q-values (satu per permutasi)
    """
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
# DQN Agent
# ============================================================
class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.action_dim = action_dim
        self.gamma      = config["gamma"]
        self.batch_size = config["batch_size"]

        self.epsilon       = config["epsilon_start"]
        self.epsilon_end   = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] Menggunakan: {self.device}")

        # Networks
        self.policy_net = DQNetwork(state_dim, action_dim, config["hidden_size"]).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim, config["hidden_size"]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["lr"])
        self.loss_fn   = nn.MSELoss()
        self.replay    = ReplayBuffer(config["replay_buffer_size"])

    def select_action(self, state, greedy=False):
        """Epsilon-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    def get_q_values(self, state):
        """Return semua Q-values untuk satu state (numpy array)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.squeeze().cpu().numpy()

    def train_step(self):
        """Satu langkah update dari replay buffer. Return loss."""
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Q(s, a) dari policy net
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max Q'(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "epsilon"              : self.epsilon,
        }, path)
        print(f"[Model] Tersimpan di: {path}")


# ============================================================
# Export Q-Table
# ============================================================
def export_qtable(agent, env, path, episode):
    """
    Generate Q-table dengan mensample beberapa state acak,
    lalu simpan ke CSV.
    """
    n_samples = 50  # jumlah state contoh yang di-log
    rows = []

    for sample_idx in range(n_samples):
        state = env.reset()
        q_values = agent.get_q_values(state)

        row = {
            "episode"   : episode,
            "sample_idx": sample_idx,
            "state"     : str(state.tolist()),
        }
        for a_idx in range(len(ALL_PERMUTATIONS)):
            perm = ALL_PERMUTATIONS[a_idx]
            row[f"Q_action{a_idx}_{perm}"] = round(float(q_values[a_idx]), 4)

        best_action = int(np.argmax(q_values))
        row["best_action"]     = best_action
        row["best_assignment"] = str(ALL_PERMUTATIONS[best_action])
        rows.append(row)

    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists or episode == CONFIG["qtable_log_interval"]:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[Q-Table] Snapshot episode {episode} tersimpan di: {path}")


# ============================================================
# Plot Training
# ============================================================
def plot_training(rewards, losses, costs, config):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("DQN Training — Assignment Problem 3x3", fontsize=14)

    # Reward
    axes[0].plot(rewards, alpha=0.4, color="steelblue", label="Reward per episode")
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(range(window-1, len(rewards)), moving_avg, color="navy", label=f"Moving avg ({window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward per Episode")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    filtered_loss = [l for l in losses if l is not None]
    axes[1].plot(filtered_loss, alpha=0.4, color="tomato", label="Loss")
    if len(filtered_loss) >= window:
        ma_loss = np.convolve(filtered_loss, np.ones(window)/window, mode="valid")
        axes[1].plot(range(window-1, len(filtered_loss)), ma_loss, color="darkred", label=f"Moving avg ({window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Cost vs Optimal
    axes[2].plot([c["agent"] for c in costs], alpha=0.5, color="orange", label="Agent cost")
    axes[2].plot([c["optimal"] for c in costs], alpha=0.5, color="green", label="Optimal cost")
    if len(costs) >= window:
        ma_agent   = np.convolve([c["agent"] for c in costs], np.ones(window)/window, mode="valid")
        ma_optimal = np.convolve([c["optimal"] for c in costs], np.ones(window)/window, mode="valid")
        axes[2].plot(range(window-1, len(costs)), ma_agent,   color="darkorange", label=f"Agent MA")
        axes[2].plot(range(window-1, len(costs)), ma_optimal, color="darkgreen",  label=f"Optimal MA")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Cost")
    axes[2].set_title("Agent Cost vs Optimal Cost")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config["plot_path"], dpi=150)
    plt.close()
    print(f"[Plot] Tersimpan di: {config['plot_path']}")


# ============================================================
# Main Training Loop
# ============================================================
def train():
    # Simpan config
    with open(CONFIG["save_config_path"], "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"[Config] Tersimpan di: {CONFIG['save_config_path']}")

    # Inisialisasi
    env   = AssignmentEnv(cost_low=CONFIG["cost_low"], cost_high=CONFIG["cost_high"], seed=CONFIG["seed"])
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.n_actions, config=CONFIG)

    # Log CSV header
    with open(CONFIG["save_log_path"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "loss", "agent_cost", "optimal_cost",
                         "assignment", "epsilon", "is_optimal"])

    # Tracking
    all_rewards = []
    all_losses  = []
    all_costs   = []
    optimal_count = 0

    print("\n" + "="*55)
    print("  Memulai Training DQN — Assignment Problem 3x3")
    print("="*55)

    for episode in range(1, CONFIG["num_episodes"] + 1):
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.replay.push(state, action, reward, next_state, float(done))
        loss = agent.train_step()
        agent.decay_epsilon()

        if episode % CONFIG["target_update_freq"] == 0:
            agent.update_target_network()

        # Tracking
        all_rewards.append(reward)
        all_losses.append(loss)
        all_costs.append({"agent": info["total_cost"], "optimal": info["optimal_cost"]})

        is_optimal = int(info["total_cost"] == info["optimal_cost"])
        optimal_count += is_optimal

        # Log ke CSV
        with open(CONFIG["save_log_path"], "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                round(reward, 4),
                round(loss, 6) if loss is not None else "",
                info["total_cost"],
                info["optimal_cost"],
                str(info["assignment"]),
                round(agent.epsilon, 4),
                is_optimal,
            ])

        # Export Q-table snapshot
        if episode % CONFIG["qtable_log_interval"] == 0:
            export_qtable(agent, env, CONFIG["save_qtable_path"], episode)

        # Print progress
        if episode % CONFIG["log_interval"] == 0:
            recent_rewards  = all_rewards[-CONFIG["log_interval"]:]
            recent_optimal  = sum(1 for c in all_costs[-CONFIG["log_interval"]:]
                                  if c["agent"] == c["optimal"])
            avg_reward      = np.mean(recent_rewards)
            opt_rate        = recent_optimal / CONFIG["log_interval"] * 100
            # Format loss safely: show 5-decimals if loss is available (including 0.0), otherwise show 'N/A'
            formatted_loss = f"{loss:.5f}" if (loss is not None) else "N/A"
            print(f"Ep {episode:>5} | AvgReward: {avg_reward:>7.2f} | "
                f"OptimalRate: {opt_rate:>5.1f}% | ε: {agent.epsilon:.3f} | "
                f"Loss: {formatted_loss}")

    # Simpan model & plot
    agent.save_model(CONFIG["save_model_path"])
    plot_training(all_rewards, all_losses, all_costs, CONFIG)

    # Export Q-table final
    export_qtable(agent, env, CONFIG["save_qtable_path"], episode="FINAL")

    # Ringkasan
    total_optimal = sum(1 for c in all_costs if c["agent"] == c["optimal"])
    print("\n" + "="*55)
    print("  Training Selesai!")
    print("="*55)
    print(f"  Total Episode     : {CONFIG['num_episodes']}")
    print(f"  Optimal Assignment: {total_optimal} / {CONFIG['num_episodes']} "
          f"({total_optimal/CONFIG['num_episodes']*100:.1f}%)")
    print(f"  Avg Reward (akhir): {np.mean(all_rewards[-200:]):.3f}")
    print(f"  Model             : {CONFIG['save_model_path']}")
    print(f"  Q-Table           : {CONFIG['save_qtable_path']}")
    print(f"  Training Log      : {CONFIG['save_log_path']}")
    print(f"  Plot              : {CONFIG['plot_path']}")
    print("="*55)


if __name__ == "__main__":
    train()
