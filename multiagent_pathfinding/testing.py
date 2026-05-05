"""
testing.py — Evaluate & animate trained CTDE agents

Usage:
  python testing.py                      # run 5 test episodes + animate 1
  python testing.py --episodes 10        # more episodes
  python testing.py --save-gif           # save animation as GIF
  python testing.py --seed 99            # different random seed

Animation legend:
  ■  dark grey  = obstacle
  ●  coloured   = agent  (A0..A3)
  ★  light fill = goal   (matching color)
  ░  trail      = path taken
"""

import os, argparse, time
import numpy as np
import matplotlib

# Force GUI backend so animation is shown in a separate window.
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("QtAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
import torch
import webbrowser

from environment import MultiAgentGridEnv
from training    import CTDEAgent, N_AGENTS, SEED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AGENT_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
                "#9B59B6", "#1ABC9C", "#E67E22", "#E91E63"]


def _open_in_new_window(path):
    """Open output artifact in a separate OS window."""
    abs_path = os.path.abspath(path)
    try:
        if os.name == "nt":
            os.startfile(abs_path)  # type: ignore[attr-defined]
        else:
            webbrowser.open(f"file://{abs_path}", new=2)
        print(f"Opened in new window -> {abs_path}")
    except Exception as e:
        print(f"Warning: unable to open output file ({e}).")


# ─────────────────────────────── Evaluator ──────────────────────────────── #

def load_agent(n_agents):
    env   = MultiAgentGridEnv(n_agents=n_agents)
    agent = CTDEAgent(
        obs_size          = env.obs_size,
        n_actions         = env.n_actions,
        n_agents          = n_agents,
        global_state_size = env.global_state_size,
    )
    agent.load("models/actor.pt")
    agent.epsilon = 0.0   # greedy
    return agent


def run_episode(env, agent, record=False):
    """Run one episode. If record=True, returns list of frames."""
    obs = env.reset()
    frames = []
    total_reward = 0.0
    step = 0

    if record:
        frames.append(_capture_frame(env))

    while True:
        actions = agent.select_actions(obs, explore=False)
        obs, rewards, dones, global_done, info = env.step(actions)
        total_reward += sum(rewards)
        step += 1

        if record:
            frames.append(_capture_frame(env))

        if global_done:
            break

    return total_reward, info, frames


def _capture_frame(env):
    """Snapshot of current state for animation."""
    return {
        "grid"        : env.grid.copy(),
        "agent_pos"   : list(env.agent_pos),
        "agent_goals" : list(env.agent_goals),
        "done_flags"  : list(env.done_flags),
        "step"        : env.steps,
    }


# ─────────────────────────────── Evaluation ─────────────────────────────── #

def evaluate(n_episodes=5, n_agents=N_AGENTS, seed=SEED):
    agent = load_agent(n_agents)
    results = []

    print(f"\n{'='*55}")
    print(f"  Evaluation  |  {n_agents} agents  |  {n_episodes} episodes")
    print(f"{'='*55}\n")

    for ep in range(n_episodes):
        env   = MultiAgentGridEnv(n_agents=n_agents, seed=seed + ep)
        reward, info, _ = run_episode(env, agent, record=False)
        results.append(info)
        print(f"Ep {ep+1:2d} | Reward={reward:7.1f} | "
              f"Goals={info['reached_goal']}/{n_agents} | "
              f"Steps={info['steps']:3d} | "
              f"Timeout={'yes' if info['timeout'] else 'no'}")

    print(f"\n{'─'*55}")
    print(f"  Avg goals  : {np.mean([r['reached_goal'] for r in results]):.2f} / {n_agents}")
    print(f"  Avg steps  : {np.mean([r['steps']        for r in results]):.1f}")
    print(f"  Success ep : {sum(r['reached_goal']==n_agents for r in results)}/{n_episodes}")
    print(f"{'─'*55}\n")

    return results


def evaluate_until_all_goal(n_agents=N_AGENTS, seed=SEED, max_episodes=1000):
    """
    Keep running test episodes until all agents reach goal in one episode.
    Stops at max_episodes to avoid infinite loops when the policy is weak.
    """
    agent = load_agent(n_agents)
    results = []

    print(f"\n{'='*55}")
    print(f"  Evaluation Until Success  |  {n_agents} agents")
    print(f"  Max episodes: {max_episodes}")
    print(f"{'='*55}\n")

    for ep in range(max_episodes):
        env = MultiAgentGridEnv(n_agents=n_agents, seed=seed + ep)
        reward, info, _ = run_episode(env, agent, record=False)
        results.append(info)

        success = info["reached_goal"] == n_agents
        print(
            f"Ep {ep+1:4d} | Reward={reward:7.1f} | "
            f"Goals={info['reached_goal']}/{n_agents} | "
            f"Steps={info['steps']:3d} | "
            f"Timeout={'yes' if info['timeout'] else 'no'} | "
            f"Success={'yes' if success else 'no'}"
        )

        if success:
            print(f"\nAll agents reached goal at episode {ep+1} (seed={seed + ep}).")
            print(f"{'─'*55}\n")
            return {
                "success": True,
                "episode_index": ep,
                "seed": seed + ep,
                "info": info,
                "results": results,
            }

    print("\nFailed to get all agents to goal within max episodes.")
    print(f"{'─'*55}\n")
    return {
        "success": False,
        "episode_index": None,
        "seed": None,
        "info": None,
        "results": results,
    }


# ─────────────────────────────── Animation ──────────────────────────────── #

def animate_episode(n_agents=N_AGENTS, seed=SEED, save_gif=False, show_animation=True):
    agent = load_agent(n_agents)
    env   = MultiAgentGridEnv(n_agents=n_agents, seed=seed)
    _, info, frames = run_episode(env, agent, record=True)

    print(f"Animating {len(frames)} frames …  "
          f"Goals={info['reached_goal']}/{n_agents}  Steps={info['steps']}")

    grid_height, grid_width = env.grid_height, env.grid_width

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.set_xticks(np.arange(grid_width))
    ax.set_yticks(np.arange(grid_height))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.grid(color="#0f3460", linewidth=0.8, zorder=0)
    ax.set_aspect("equal")

    # ── Draw static obstacles ─────────────────────────────────────────── #
    grid = frames[0]["grid"]
    for r in range(grid_height):
        for c in range(grid_width):
            if grid[r, c] == 1:
                rect = mpatches.FancyBboxPatch(
                    (c - 0.45, r - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.05",
                    fc="#2c3e50", ec="#4a6fa5", lw=0.8, zorder=1)
                ax.add_patch(rect)

    # ── Goal markers (static per episode) ────────────────────────────── #
    goal_stars = []
    for i, (gr, gc) in enumerate(frames[0]["agent_goals"]):
        col = AGENT_COLORS[i % len(AGENT_COLORS)]
        star = ax.plot(gc, gr, marker="*", markersize=18,
                       color=col, alpha=0.5, zorder=2)[0]
        goal_stars.append(star)

    # ── Trail lines (one per agent) ───────────────────────────────────── #
    trail_xs = [[] for _ in range(n_agents)]
    trail_ys = [[] for _ in range(n_agents)]
    trail_lines = []
    for i in range(n_agents):
        col = AGENT_COLORS[i % len(AGENT_COLORS)]
        line, = ax.plot([], [], color=col, alpha=0.35, lw=1.5, zorder=3)
        trail_lines.append(line)

    # ── Agent circles ────────────────────────────────────────────────── #
    agent_circles = []
    for i in range(n_agents):
        col = AGENT_COLORS[i % len(AGENT_COLORS)]
        circ = plt.Circle((0, 0), 0.35, color=col, zorder=5)
        ax.add_patch(circ)
        agent_circles.append(circ)

    # Agent labels
    agent_labels = []
    for i in range(n_agents):
        lbl = ax.text(0, 0, str(i), color="white", fontsize=9,
                      ha="center", va="center", fontweight="bold", zorder=6)
        agent_labels.append(lbl)

    # ── Step counter & title ──────────────────────────────────────────── #
    step_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        color="white", fontsize=11, va="top",
        fontfamily="monospace", zorder=7)

    goal_text = ax.text(
        0.98, 0.97, "", transform=ax.transAxes,
        color="#2ECC71", fontsize=11, va="top", ha="right",
        fontfamily="monospace", zorder=7)

    ax.set_title("CTDE Multi-Agent Path Planning",
                 color="white", fontsize=13, pad=8)

    # ── Legend ────────────────────────────────────────────────────────── #
    legend_elements = [
        mpatches.Patch(fc="#2c3e50", ec="#4a6fa5", label="Obstacle"),
        mpatches.Patch(fc="none", ec="none", label="★ Goal"),
        mpatches.Patch(fc="none", ec="none", label="● Agent"),
    ]
    for i in range(min(n_agents, 4)):
        legend_elements.append(
            mpatches.Patch(fc=AGENT_COLORS[i], label=f"Agent {i}"))
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#4a6fa5",
              labelcolor="white", fontsize=8, framealpha=0.85)

    # ── Animation update ──────────────────────────────────────────────── #
    def update(frame_idx):
        f = frames[frame_idx]

        for i in range(n_agents):
            r, c = f["agent_pos"][i]
            trail_xs[i].append(c)
            trail_ys[i].append(r)
            trail_lines[i].set_data(trail_xs[i], trail_ys[i])

            agent_circles[i].center = (c, r)
            agent_labels[i].set_position((c, r))

            # Dim agent circle when done
            alpha = 0.4 if f["done_flags"][i] else 1.0
            agent_circles[i].set_alpha(alpha)

            # Highlight goal when reached
            if f["done_flags"][i]:
                goal_stars[i].set_alpha(1.0)
                goal_stars[i].set_markersize(22)

        goals_done = sum(f["done_flags"])
        step_text.set_text(f"Step: {f['step']:3d}")
        goal_text.set_text(f"Goals: {goals_done}/{n_agents}")
        return (*trail_lines, *agent_circles, *agent_labels,
                step_text, goal_text, *goal_stars)

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=180,    # ms per frame
        blit=True,
        repeat=True,
    )

    backend = str(matplotlib.get_backend()).lower()
    non_interactive_backends = ["agg", "cairo", "svg", "pdf", "ps", "template"]
    backend_non_interactive = backend in non_interactive_backends
    force_gif_fallback = show_animation and backend_non_interactive

    saved_artifact = None
    if save_gif or force_gif_fallback:
        path = "models/episode_animation.gif"
        ani.save(path, writer="pillow", fps=6, dpi=90)
        print(f"Saved animation -> {path}")
        saved_artifact = path
    else:
        plt.tight_layout()
        plt.savefig("models/episode_animation_final.png", dpi=120,
                    facecolor=fig.get_facecolor())
        print("Saved final frame -> models/episode_animation_final.png")
        saved_artifact = "models/episode_animation_final.png"

        # Also save a multi-frame composite (every 20 steps)
        _save_composite(frames, n_agents)

    if show_animation:
        if backend_non_interactive:
            print(f"Backend '{matplotlib.get_backend()}' non-interactive; opening saved output instead.")
            if saved_artifact is not None:
                _open_in_new_window(saved_artifact)
            plt.close()
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: unable to display animation window ({e}).")
                if saved_artifact is not None:
                    _open_in_new_window(saved_artifact)
                plt.close()
    else:
        plt.close()


def _save_composite(frames, n_agents, max_panels=6):
    """Save a grid of key frames for quick visual inspection."""
    indices = np.linspace(0, len(frames)-1, min(max_panels, len(frames)), dtype=int)
    ncols = 3
    nrows = (len(indices) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    fig.patch.set_facecolor("#1a1a2e")
    if nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for panel_i, fi in enumerate(indices):
        ax = axes_flat[panel_i]
        f  = frames[fi]
        grid_height, grid_width = f["grid"].shape
        ax.set_facecolor("#16213e")
        ax.set_xlim(-0.5, grid_width-0.5)
        ax.set_ylim(-0.5, grid_height-0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(color="#0f3460", linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Step {f['step']}", color="white", fontsize=9)

        for r in range(grid_height):
            for c in range(grid_width):
                if f["grid"][r, c] == 1:
                    rect = mpatches.FancyBboxPatch(
                        (c-.45, r-.45), .9, .9,
                        boxstyle="round,pad=0.04",
                        fc="#2c3e50", ec="#4a6fa5", lw=0.5)
                    ax.add_patch(rect)

        for i, (gr, gc) in enumerate(f["agent_goals"]):
            ax.plot(gc, gr, marker="*", markersize=14,
                    color=AGENT_COLORS[i], alpha=0.6)

        for i, (ar, ac) in enumerate(f["agent_pos"]):
            col = AGENT_COLORS[i]
            alpha = 0.4 if f["done_flags"][i] else 1.0
            circ = plt.Circle((ac, ar), 0.35, color=col, alpha=alpha)
            ax.add_patch(circ)
            ax.text(ac, ar, str(i), color="white", fontsize=7,
                    ha="center", va="center", fontweight="bold")

    for j in range(len(indices), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("CTDE Agent Paths — Key Frames", color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig("models/episode_keyframes.png", dpi=120,
                facecolor=fig.get_facecolor())
    print("Saved key frames -> models/episode_keyframes.png")
    plt.close()


# ─────────────────────────────── Entry point ───────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,  default=5)
    parser.add_argument("--episode",   type=int,  default=None,
                        help="Alias for --episodes")
    parser.add_argument("--agents",    type=int,  default=N_AGENTS)
    parser.add_argument("--seed",      type=int,  default=SEED)
    parser.add_argument("--fixed-episodes", action="store_true",
                        help="Use fixed number of episodes instead of until-all-goal mode")
    parser.add_argument("--max-episodes", type=int, default=1000,
                        help="Safety cap for until-all-goal mode")
    parser.add_argument("--save-gif",  action="store_true")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable showing animation window at the end")
    args = parser.parse_args()

    episodes = args.episode if args.episode is not None else args.episodes

    if not args.fixed_episodes:
        run = evaluate_until_all_goal(
            n_agents=args.agents,
            seed=args.seed,
            max_episodes=args.max_episodes,
        )
        if run["success"]:
            animate_episode(
                n_agents=args.agents,
                seed=run["seed"],
                save_gif=args.save_gif,
                show_animation=not args.no_show,
            )
        else:
            last_seed = args.seed + args.max_episodes - 1
            print(f"Animating last attempted episode (seed={last_seed}) …")
            animate_episode(
                n_agents=args.agents,
                seed=last_seed,
                save_gif=args.save_gif,
                show_animation=not args.no_show,
            )
    else:
        evaluate(n_episodes=episodes, n_agents=args.agents, seed=args.seed)
        animate_episode(
            n_agents=args.agents,
            seed=args.seed,
            save_gif=args.save_gif,
            show_animation=not args.no_show,
        )
