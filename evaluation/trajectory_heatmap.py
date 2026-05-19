"""
evaluation/trajectory_heatmap.py
────────────────────────────────────────────────────────────────────────────
Generate spatial heatmaps showing where robots spend time, which goals they
prioritize, and where they position during endgame.

Usage
-----
    python evaluation/trajectory_heatmap.py \\
        --checkpoint artifacts/models/best_policy.pt \\
        --num-matches 50 \\
        --output artifacts/heatmaps/

Outputs
-------
  heatmap_red.png          — combined red alliance positions
  heatmap_blue.png         — combined blue alliance positions
  heatmap_endgame_red.png  — red positions during endgame only
  heatmap_endgame_blue.png — blue positions during endgame only
  goal_priority.png        — bar chart of how often each goal is scored into
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.env_wrapper import OverrideEnv, AGENT_IDS
from training.network     import Policy
from config.game_rules    import (
    FIELD_WIDTH, FIELD_HEIGHT, RENDER_SCALE, GOALS, ENDGAME_SECONDS,
)
from config.hyperparameters import HEATMAPS_DIR, CONTROL_DT

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("[Heatmap] matplotlib not installed. Run: pip install matplotlib")


GRID_RES = 2   # heatmap cell size in inches


def parse_args():
    p = argparse.ArgumentParser(description="VEX Override Trajectory Heatmap")
    p.add_argument("--checkpoint",  type=str, default=None)
    p.add_argument("--num-matches", type=int, default=20)
    p.add_argument("--output",      type=str, default=HEATMAPS_DIR)
    p.add_argument("--device",      type=str, default="auto")
    return p.parse_args()


def select_device(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def load_policies(ck_path, device):
    red  = Policy().to(device).eval()
    blue = Policy().to(device).eval()
    if ck_path and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device)
        if "red_policy"  in ck: red.load_state_dict(ck["red_policy"])
        if "blue_policy" in ck: blue.load_state_dict(ck["blue_policy"])
        print(f"[Heatmap] Loaded from {ck_path}")
    for pol in [red, blue]:
        pol.eval()
        for p in pol.parameters(): p.requires_grad_(False)
    return red, blue


def collect_trajectories(red_policy, blue_policy, device, n_matches):
    """
    Run n_matches headless and collect (x, y, phase, alliance) per step.
    Returns dict with trajectory arrays.
    """
    grid_w = int(FIELD_WIDTH  / GRID_RES)
    grid_h = int(FIELD_HEIGHT / GRID_RES)

    hm_red      = np.zeros((grid_h, grid_w))
    hm_blue     = np.zeros((grid_h, grid_w))
    hm_end_red  = np.zeros((grid_h, grid_w))
    hm_end_blue = np.zeros((grid_h, grid_w))

    goal_scores = np.zeros(len(GOALS))    # per goal: how often scored into

    policy_map = {
        "red1": red_policy, "red2": red_policy,
        "blue1": blue_policy, "blue2": blue_policy,
    }

    for match_i in range(n_matches):
        env   = OverrideEnv(headless=True, seed=match_i * 17)
        obs   = env.reset()
        masks = env.get_action_masks()
        done  = False

        while not done:
            actions = {}
            for rid in AGENT_IDS:
                o = torch.FloatTensor(obs[rid]).to(device)
                m = torch.BoolTensor(masks[rid]).to(device)
                with torch.no_grad():
                    c, d, _, _ = policy_map[rid].get_action(o, m, deterministic=True)
                actions[rid] = (c.cpu().numpy(), d.cpu().numpy())

            obs, rewards, done, info = env.step(actions)
            masks = env.get_action_masks()

            is_endgame = env.sim.rules_engine.endgame_active

            # Record robot positions
            for robot in env.sim.robots:
                rx = int(min(robot.body.position.x, FIELD_WIDTH  - 1) / GRID_RES)
                ry = int(min(robot.body.position.y, FIELD_HEIGHT - 1) / GRID_RES)
                rx = max(0, min(rx, grid_w - 1))
                ry = max(0, min(ry, grid_h - 1))

                if robot.alliance == "red":
                    hm_red[ry, rx] += 1
                    if is_endgame:
                        hm_end_red[ry, rx] += 1
                else:
                    hm_blue[ry, rx] += 1
                    if is_endgame:
                        hm_end_blue[ry, rx] += 1

        # Record goal stack heights at match end
        for goal in env.sim.goals:
            gidx = next((i for i, g in enumerate(GOALS) if g["id"] == goal.goal_id), None)
            if gidx is not None:
                goal_scores[gidx] += len(goal.stack)

        env.close()
        print(f"[Heatmap] Match {match_i+1}/{n_matches} done.")

    return {
        "red":       hm_red,
        "blue":      hm_blue,
        "end_red":   hm_end_red,
        "end_blue":  hm_end_blue,
        "goal_scores": goal_scores,
    }


def save_heatmap(grid, title, path, cmap="hot", overlay_goals=True):
    if not MATPLOTLIB_OK:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grid, origin="lower", cmap=cmap,
                   extent=[0, FIELD_WIDTH, 0, FIELD_HEIGHT],
                   interpolation="gaussian")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Field X (inches)")
    ax.set_ylabel("Field Y (inches)")
    plt.colorbar(im, ax=ax, label="Time spent (steps)")

    if overlay_goals:
        for g in GOALS:
            color = ("red" if g["alliance"] == "red" else
                     "blue" if g["alliance"] == "blue" else "white")
            ax.scatter(g["x"], g["y"], c=color, s=200, zorder=5,
                       edgecolors="black", linewidths=1.5)
            ax.annotate(g["label"], (g["x"], g["y"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="white",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[Heatmap] Saved → {path}")


def save_goal_priority(goal_scores, n_matches, path):
    if not MATPLOTLIB_OK:
        return
    labels = [g["label"] for g in GOALS]
    avg    = goal_scores / max(n_matches, 1)

    colors = []
    for g in GOALS:
        if g["alliance"] == "red":   colors.append("#E83030")
        elif g["alliance"] == "blue": colors.append("#3070D8")
        else:                         colors.append("#A0A0B0")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, avg, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("Average Stack Height per Goal at Match End", fontsize=14)
    ax.set_ylabel("Avg items in stack")
    ax.set_xlabel("Goal")
    for bar, val in zip(bars, avg):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[Heatmap] Goal priority chart → {path}")


def main():
    args   = parse_args()
    device = select_device(args.device)

    os.makedirs(args.output, exist_ok=True)

    red_policy, blue_policy = load_policies(args.checkpoint, device)

    print(f"[Heatmap] Collecting trajectories from {args.num_matches} matches...")
    data = collect_trajectories(red_policy, blue_policy, device,
                                args.num_matches)

    save_heatmap(data["red"],     "Red Alliance — All Match",
                 os.path.join(args.output, "heatmap_red.png"),      cmap="Reds")
    save_heatmap(data["blue"],    "Blue Alliance — All Match",
                 os.path.join(args.output, "heatmap_blue.png"),     cmap="Blues")
    save_heatmap(data["end_red"], "Red Alliance — Endgame Only",
                 os.path.join(args.output, "heatmap_endgame_red.png"),  cmap="Reds")
    save_heatmap(data["end_blue"],"Blue Alliance — Endgame Only",
                 os.path.join(args.output, "heatmap_endgame_blue.png"), cmap="Blues")
    save_goal_priority(data["goal_scores"], args.num_matches,
                       os.path.join(args.output, "goal_priority.png"))

    print("[Heatmap] Done.")


if __name__ == "__main__":
    main()
