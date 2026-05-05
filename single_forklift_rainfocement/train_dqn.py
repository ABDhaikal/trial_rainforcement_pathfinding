from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from environment import PathFindingEnv


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_action(model: DQN, state: torch.Tensor, epsilon: float, action_size: int) -> int:
    if random.random() < epsilon:
        return random.randrange(action_size)

    with torch.no_grad():
        q_values = model(state.unsqueeze(0)).squeeze(0)
        return int(torch.argmax(q_values).item())


def optimize_model(
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    replay_buffer: deque[Transition],
    batch_size: int,
    gamma: float,
) -> float | None:
    if len(replay_buffer) < batch_size:
        return None

    batch = random.sample(replay_buffer, batch_size)
    states = torch.stack([transition.state for transition in batch])
    actions = torch.tensor([transition.action for transition in batch], dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float32)
    next_states = torch.stack([transition.next_state for transition in batch])
    dones = torch.tensor([transition.done for transition in batch], dtype=torch.float32)

    current_q = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        target_q = rewards + gamma * next_q * (1.0 - dones)

    loss = nn.SmoothL1Loss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def evaluate_policy(env: PathFindingEnv, model: DQN, episodes: int = 10) -> float:
    total_reward = 0.0
    model.eval()

    for _ in range(episodes):
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            with torch.no_grad():
                action = int(torch.argmax(model(state_tensor.unsqueeze(0))).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state_tensor = torch.tensor(next_state, dtype=torch.float32)

    model.train()
    return total_reward / episodes


def train(
    dataset_path: str,
    model_path: str,
    episodes: int,
    max_steps: int,
    batch_size: int,
    gamma: float,
    lr: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: float,
    target_update: int,
    seed: int,
) -> DQN:
    set_seed(seed)

    env = PathFindingEnv(dataset_path=dataset_path, max_steps=max_steps)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(input_dim=input_dim, output_dim=action_dim)
    target_net = DQN(input_dim=input_dim, output_dim=action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer: deque[Transition] = deque(maxlen=50_000)

    epsilon = epsilon_start
    best_eval_reward = -float("inf")

    print(f"Mulai training: input_dim={input_dim}, action_dim={action_dim}, episodes={episodes}")

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        episode_reward = 0.0
        episode_loss = None
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = select_action(policy_net, state_tensor, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            replay_buffer.append(
                Transition(
                    state=state_tensor,
                    action=action,
                    reward=float(reward),
                    next_state=next_state_tensor,
                    done=bool(terminated or truncated),
                )
            )

            loss = optimize_model(
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                gamma=gamma,
            )
            if loss is not None:
                episode_loss = loss

            state_tensor = next_state_tensor
            episode_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 25 == 0 or episode == 1:
            eval_reward = evaluate_policy(env, policy_net, episodes=5)
            best_eval_reward = max(best_eval_reward, eval_reward)
            print(
                f"Episode {episode}/{episodes} | reward={episode_reward:.2f} | "
                f"epsilon={epsilon:.3f} | loss={(episode_loss if episode_loss is not None else float('nan')):.4f} | "
                f"eval_avg_reward={eval_reward:.2f} | best_eval={best_eval_reward:.2f}"
            )
        else:
            print(
                f"Episode {episode}/{episodes} | reward={episode_reward:.2f} | "
                f"epsilon={epsilon:.3f} | loss={(episode_loss if episode_loss is not None else float('nan')):.4f}"
            )

    torch.save(policy_net.state_dict(), model_path)
    print(f"Model tersimpan di: {Path(model_path).resolve()}")
    return policy_net


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training DQN untuk PathFindingEnv single forklift")
    parser.add_argument("--dataset-path", type=str, default=str(CURRENT_DIR / "warehouse_dataset.npz"))
    parser.add_argument("--model-path", type=str, default=str(CURRENT_DIR / "trained_dqn_model.pth"))
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    train(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()