from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from environment import PathFindingEnv
from train_dqn import DQN


ACTION_NAMES = {
    0: "Kanan",
    1: "Kiri",
    2: "Atas",
    3: "Bawah",
    4: "Diam",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_greedy_action(model: DQN, state_tensor: torch.Tensor) -> int:
    with torch.no_grad():
        q_values = model(state_tensor.unsqueeze(0)).squeeze(0)
    return int(torch.argmax(q_values).item())


def capture_frame(
    env: PathFindingEnv,
    step: int,
    action: int | None,
    reward: float,
    total_reward: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    grid = env.grid.copy().astype(np.int8)

    ay, ax = env.agent_pos
    gy, gx = env.goal_pos

    if ay == gy and ax == gx:
        grid[gy, gx] = 4
    else:
        grid[gy, gx] = 3
        grid[ay, ax] = 2

    return {
        "grid": grid,
        "step": step,
        "action": action,
        "reward": float(reward),
        "total_reward": float(total_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "agent_pos": (int(ay), int(ax)),
        "goal_pos": (int(gy), int(gx)),
    }


def run_episode(env: PathFindingEnv, model: DQN, seed: int | None = None) -> dict[str, Any]:
    state, _ = env.reset(seed=seed)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    total_reward = 0.0
    terminated = False
    truncated = False
    step_count = 0
    frames: list[dict[str, Any]] = []

    frames.append(
        capture_frame(
            env=env,
            step=0,
            action=None,
            reward=0.0,
            total_reward=0.0,
            terminated=False,
            truncated=False,
        )
    )

    while not terminated and not truncated:
        step_count += 1
        action = select_greedy_action(model, state_tensor)
        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += float(reward)
        state_tensor = torch.tensor(next_state, dtype=torch.float32)

        frames.append(
            capture_frame(
                env=env,
                step=step_count,
                action=action,
                reward=float(reward),
                total_reward=total_reward,
                terminated=terminated,
                truncated=truncated,
            )
        )

    return {
        "frames": frames,
        "steps": step_count,
        "total_reward": total_reward,
        "success": bool(terminated and env.agent_pos == env.goal_pos),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def animate_episode(frames: list[dict[str, Any]], fps: int, title: str, save_gif: str | None = None) -> None:
    cmap = ListedColormap(
        [
            "#f5f5f5",  # 0: lantai
            "#2f4f4f",  # 1: rak
            "#3498db",  # 2: agen
            "#e74c3c",  # 3: goal
            "#2ecc71",  # 4: agen mencapai goal
        ]
    )

    initial_grid = frames[0]["grid"]
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(initial_grid, cmap=cmap, vmin=0, vmax=4)

    height, width = initial_grid.shape
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#c8c8c8", linestyle="-", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    info_text = ax.text(0.02, 1.02, "", transform=ax.transAxes, fontsize=10)

    legend_elements = [
        Patch(facecolor="#f5f5f5", edgecolor="gray", label="Lantai"),
        Patch(facecolor="#2f4f4f", edgecolor="gray", label="Rak"),
        Patch(facecolor="#3498db", edgecolor="gray", label="Agen"),
        Patch(facecolor="#e74c3c", edgecolor="gray", label="Goal"),
        Patch(facecolor="#2ecc71", edgecolor="gray", label="Goal Tercapai"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.33, 1.02))

    def update(frame_idx: int):
        frame = frames[frame_idx]
        img.set_data(frame["grid"])

        episode_text = frame.get("episode")
        action = frame["action"]
        action_name = "Mulai" if action is None else ACTION_NAMES.get(action, str(action))

        if frame["terminated"]:
            status = "SELESAI"
        elif frame["truncated"]:
            status = "MAKS STEP"
        else:
            status = "BERJALAN"

        prefix = f"Episode={episode_text} | " if episode_text is not None else ""
        info_text.set_text(
            f"{prefix}Step={frame['step']} | Aksi={action_name} | Reward={frame['reward']:.1f} | "
            f"Total={frame['total_reward']:.1f} | Status={status}"
        )

        return [img, info_text]

    interval = int(1000 / max(1, fps))
    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        blit=False,
        repeat=False,
    )

    if save_gif:
        save_path = Path(save_gif).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(save_path), writer=PillowWriter(fps=max(1, fps)))
        print(f"Animasi GIF tersimpan di: {save_path}")

    plt.tight_layout()
    plt.show()


def build_all_episode_frames(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_frames: list[dict[str, Any]] = []

    for episode_idx, result in enumerate(results, start=1):
        frames = result.get("frames", [])
        for frame in frames:
            merged = dict(frame)
            merged["episode"] = episode_idx
            all_frames.append(merged)

        # Tambahkan jeda singkat antar episode agar transisi terlihat.
        if frames and episode_idx < len(results):
            last_frame = dict(frames[-1])
            last_frame["episode"] = episode_idx
            for _ in range(3):
                all_frames.append(last_frame)

    return all_frames


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test model DQN single forklift + animasi matplotlib"
    )
    parser.add_argument("--dataset-path", type=str, default=str(CURRENT_DIR / "warehouse_dataset.npz"))
    parser.add_argument("--model-path", type=str, default=str(CURRENT_DIR / "trained_dqn_model.pth"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--animate-episode",
        type=int,
        default=0,
        help=(
            "Mode animasi: 0=animasikan semua episode (dari awal hingga akhir), "
            "-1=nonaktif, >0=episode tertentu"
        ),
    )
    parser.add_argument(
        "--save-gif",
        type=str,
        default="",
        help="Path file GIF output (opsional), contoh: episode1.gif",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    set_seed(args.seed)

    env = PathFindingEnv(dataset_path=args.dataset_path, max_steps=args.max_steps)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(input_dim=input_dim, output_dim=action_dim)

    model_file = Path(args.model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_file.resolve()}")

    state_dict = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"Mulai testing: episodes={args.episodes}, max_steps={args.max_steps}, "
        f"model={model_file.resolve()}"
    )

    results: list[dict[str, Any]] = []
    for episode in range(1, args.episodes + 1):
        result = run_episode(env=env, model=model, seed=args.seed + episode)
        results.append(result)

        print(
            f"Episode {episode}/{args.episodes} | steps={result['steps']} | "
            f"reward={result['total_reward']:.2f} | success={result['success']} | "
            f"terminated={result['terminated']} | truncated={result['truncated']}"
        )

    if args.animate_episode == 0 and results:
        save_gif = args.save_gif.strip() or None
        print("\nMode default: animasi seluruh episode dari state awal hingga akhir.")
        all_frames = build_all_episode_frames(results)
        animate_episode(
            frames=all_frames,
            fps=args.fps,
            title=f"Simulasi Semua Episode ({args.episodes} episode)",
            save_gif=save_gif,
        )

    if args.animate_episode > 0:
        target_episode = args.animate_episode
        if 1 <= target_episode <= len(results):
            save_gif = args.save_gif.strip() or None
            animate_episode(
                frames=results[target_episode - 1]["frames"],
                fps=args.fps,
                title=f"Simulasi Episode {target_episode}",
                save_gif=save_gif,
            )
        else:
            print(
                f"\nanimate-episode={target_episode} di luar rentang hasil (1-{len(results)})."
            )

    avg_reward = float(np.mean([item["total_reward"] for item in results])) if results else 0.0
    success_rate = float(np.mean([item["success"] for item in results]) * 100.0) if results else 0.0

    print("\nRingkasan Testing")
    print(f"- Rata-rata reward: {avg_reward:.2f}")
    print(f"- Success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    main()