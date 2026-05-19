"""
evaluation/evaluate.py
────────────────────────────────────────────────────────────────────────────
Evaluate trained VEX Override agents.

Modes
-----
  headless  — run N matches, print stats, optionally record videos
  render    — watch a single match in a pygame window
  interactive — you control red1 (WASD/E/Q/F/T), AI controls the rest

Usage
-----
    python evaluation/evaluate.py --checkpoint artifacts/models/best_policy.pt
    python evaluation/evaluate.py --checkpoint artifacts/models/best_policy.pt \\
                                   --record-video --num-matches 10
    python evaluation/evaluate.py --checkpoint artifacts/models/best_policy.pt \\
                                   --mode render
    python evaluation/evaluate.py --checkpoint artifacts/models/best_policy.pt \\
                                   --mode interactive
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.env_wrapper  import OverrideEnv, AGENT_IDS
from training.network      import Policy
from utils.observation_builder import build_all_observations, get_action_mask
from config.hyperparameters import (
    MODELS_DIR, VIDEOS_DIR, CONTROL_DT,
)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="VEX Override Policy Evaluator")
    p.add_argument("--checkpoint",  type=str,  default=None,
                   help="Path to .pt checkpoint (red policy is used)")
    p.add_argument("--mode",        type=str,  default="headless",
                   choices=["headless", "render", "interactive"],
                   help="Evaluation mode")
    p.add_argument("--num-matches", type=int,  default=10)
    p.add_argument("--record-video",action="store_true")
    p.add_argument("--device",      type=str,  default="auto")
    p.add_argument("--deterministic", action="store_true",
                   help="Use policy mode instead of sampling")
    return p.parse_args()


def select_device(s: str) -> torch.device:
    if s == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(s)


def load_policies(checkpoint_path: str, device: torch.device):
    """Load red and blue policies from a MAPPO checkpoint."""
    red_policy  = Policy().to(device).eval()
    blue_policy = Policy().to(device).eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=device)
        if "red_policy" in ck:
            red_policy.load_state_dict(ck["red_policy"])
        if "blue_policy" in ck:
            blue_policy.load_state_dict(ck["blue_policy"])
        print(f"[Eval] Loaded policies from {checkpoint_path}")
    else:
        print("[Eval] No checkpoint — using random policies.")

    for p in [red_policy, blue_policy]:
        p.eval()
        for param in p.parameters():
            param.requires_grad_(False)

    return red_policy, blue_policy


# ─────────────────────────────────────────────────────────────────────────────
def run_match_headless(
    red_policy: Policy,
    blue_policy: Policy,
    device: torch.device,
    record: bool = False,
    video_path: str = None,
    deterministic: bool = True,
    seed: int = None,
) -> dict:
    """
    Run one headless match and return result dict.

    Returns
    -------
    {winner, red_score, blue_score, steps, stack_heights, denial_rate}
    """
    env   = OverrideEnv(headless=True, seed=seed or int(time.time()))
    obs   = env.reset()
    masks = env.get_action_masks()
    done  = False
    stats = {"total_steps": 0}

    policy_map = {
        "red1": red_policy, "red2": red_policy,
        "blue1": blue_policy, "blue2": blue_policy,
    }

    # Headless recording via off-screen surface if requested
    recorder = None
    if record and video_path:
        try:
            import pygame
            from evaluation.video_recorder import VideoRecorder
            pygame.init()
            # Off-screen surface matching the sim dimensions
            _tmp_sim = env.sim
            recorder = VideoRecorder(video_path, _tmp_sim.screen_w,
                                     _tmp_sim.screen_h)
        except Exception as e:
            print(f"[Eval] Headless recording unavailable: {e}")

    while not done:
        actions = {}
        for rid in AGENT_IDS:
            o = torch.FloatTensor(obs[rid]).to(device)
            m = torch.BoolTensor(masks[rid]).to(device)
            with torch.no_grad():
                c, d, _, _ = policy_map[rid].get_action(
                    o, m, deterministic=deterministic)
            actions[rid] = (c.cpu().numpy(), d.cpu().numpy())

        obs, rewards, done, info = env.step(actions)
        masks = env.get_action_masks()
        stats["total_steps"] += 1

        if recorder:
            try:
                env.render()
                frame = pygame.surfarray.array3d(env.sim.screen).transpose(1, 0, 2)
                recorder.write_frame(frame)
            except Exception:
                pass

    if recorder:
        recorder.close()

    red_score  = info["red_score"]
    blue_score = info["blue_score"]
    winner     = ("red" if red_score  > blue_score else
                  "blue" if blue_score > red_score  else "tie")

    stack_heights = [len(g.stack) for g in env.sim.goals]

    env.close()
    return {
        "winner":        winner,
        "red_score":     red_score,
        "blue_score":    blue_score,
        "steps":         stats["total_steps"],
        "stack_heights": stack_heights,
        "avg_stack_h":   np.mean(stack_heights),
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_rendered_match(
    red_policy: Policy,
    blue_policy: Policy,
    device: torch.device,
    record: bool = False,
    video_path: str = None,
    deterministic: bool = True,
):
    """Run a full rendered match in a pygame window."""
    env   = OverrideEnv(headless=False, seed=int(time.time()))
    obs   = env.reset()
    masks = env.get_action_masks()

    recorder = None
    if record and video_path:
        try:
            from evaluation.video_recorder import VideoRecorder
            recorder = VideoRecorder(
                video_path, env.sim.screen_w, env.sim.screen_h, fps=30)
        except Exception as e:
            print(f"[Eval] Video recording unavailable: {e}")

    policy_map = {
        "red1": red_policy, "red2": red_policy,
        "blue1": blue_policy, "blue2": blue_policy,
    }

    clock   = pygame.time.Clock()
    running = True
    done    = False

    print("[Eval] Rendered match started. Press ESC to quit, R to reset.")

    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs   = env.reset()
                    masks = env.get_action_masks()
                    done  = False

        actions = {}
        for rid in AGENT_IDS:
            o = torch.FloatTensor(obs[rid]).to(device)
            m = torch.BoolTensor(masks[rid]).to(device)
            with torch.no_grad():
                c, d, _, _ = policy_map[rid].get_action(
                    o, m, deterministic=deterministic)
            actions[rid] = (c.cpu().numpy(), d.cpu().numpy())

        obs, rewards, done, info = env.step(actions)
        masks = env.get_action_masks()
        env.render()

        if recorder:
            surf  = env.sim.screen
            frame = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
            recorder.write_frame(frame)

        clock.tick(30)

    if recorder:
        recorder.close()

    env.close()
    return info


# ─────────────────────────────────────────────────────────────────────────────
def run_interactive_match(
    blue_policy: Policy,
    device: torch.device,
    record: bool = False,
    video_path: str = None,
):
    """
    YOU control Red alliance (WASD/E/Q/F/T keys).
    AI controls Blue alliance.
    """
    env   = OverrideEnv(headless=False, seed=int(time.time()))
    sim   = env.sim
    obs   = env.reset()
    masks = env.get_action_masks()

    recorder = None
    if record and video_path:
        try:
            from evaluation.video_recorder import VideoRecorder
            recorder = VideoRecorder(
                video_path, sim.screen_w, sim.screen_h, fps=30)
        except Exception as e:
            print(f"[Eval] Video recording unavailable: {e}")

    clock   = pygame.time.Clock()
    running = True
    done    = False

    print("[Eval] Interactive mode — YOU are Red (WASD/E/Q/F/T).")
    print("       AI controls Blue. Press ESC to quit.")

    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs   = env.reset()
                    masks = env.get_action_masks()
                    done  = False

        keys = pygame.key.get_pressed()

        # ── Human red1 control (WASD + Q/E/F/T) ──
        left = right = 0.0
        if keys[pygame.K_w]: left = right = 1.0
        if keys[pygame.K_s]: left = right = -1.0
        if keys[pygame.K_a]: left, right = -0.85,  0.85
        if keys[pygame.K_d]: left, right =  0.85, -0.85

        human_cont = np.array([left, right], dtype=np.float32)
        human_disc = np.array([
            float(keys[pygame.K_e]),                              # intake
            float(keys[pygame.K_q] and not keys[pygame.K_LSHIFT]),# score_pin
            float(keys[pygame.K_q] and keys[pygame.K_LSHIFT]),    # score_cup
            float(keys[pygame.K_t]),                              # toggle
            float(keys[pygame.K_f] and not keys[pygame.K_LSHIFT]),# flip_pin
            float(keys[pygame.K_f] and keys[pygame.K_LSHIFT]),    # flip_cup
            0.0,                                                   # match_load
        ], dtype=np.float32)

        actions = {
            "red1": (human_cont, human_disc),
            "red2": (np.zeros(2), np.zeros(7)),   # red2 idles
        }

        # ── AI blue control ──
        for rid in ["blue1", "blue2"]:
            o = torch.FloatTensor(obs[rid]).to(device)
            m = torch.BoolTensor(masks[rid]).to(device)
            with torch.no_grad():
                c, d, _, _ = blue_policy.get_action(o, m, deterministic=False)
            actions[rid] = (c.cpu().numpy(), d.cpu().numpy())

        obs, rewards, done, info = env.step(actions)
        masks = env.get_action_masks()
        env.render()

        if recorder:
            surf  = sim.screen
            frame = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
            recorder.write_frame(frame)

        clock.tick(30)

    if recorder:
        recorder.close()

    env.close()
    print(f"[Eval] Final: Red {info.get('red_score','?')} – "
          f"{info.get('blue_score','?')} Blue")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = select_device(args.device)

    os.makedirs(VIDEOS_DIR, exist_ok=True)

    red_policy, blue_policy = load_policies(args.checkpoint, device)

    if args.mode == "headless":
        print(f"\n[Eval] Running {args.num_matches} headless matches...")
        results = []
        wins = {"red": 0, "blue": 0, "tie": 0}

        for i in range(args.num_matches):
            vpath = None
            if args.record_video:
                vpath = os.path.join(VIDEOS_DIR, f"eval_match_{i+1:03d}.mp4")
            r = run_match_headless(
                red_policy, blue_policy, device,
                record=args.record_video, video_path=vpath,
                deterministic=args.deterministic, seed=i)
            results.append(r)
            wins[r["winner"]] += 1
            print(f"  Match {i+1:2d}: {r['winner'].upper():4s} "
                  f"| Red {r['red_score']:>4} – {r['blue_score']:>4} Blue "
                  f"| Avg stack: {r['avg_stack_h']:.2f} "
                  f"| Steps: {r['steps']}")

        print(f"\n[Eval] Summary over {args.num_matches} matches:")
        print(f"  Red  wins: {wins['red']:>3}  ({wins['red']/args.num_matches*100:.0f}%)")
        print(f"  Blue wins: {wins['blue']:>3}  ({wins['blue']/args.num_matches*100:.0f}%)")
        print(f"  Ties:      {wins['tie']:>3}")
        print(f"  Avg Red score:  {np.mean([r['red_score']  for r in results]):.1f}")
        print(f"  Avg Blue score: {np.mean([r['blue_score'] for r in results]):.1f}")
        print(f"  Avg stack h:    {np.mean([r['avg_stack_h'] for r in results]):.2f}")

    elif args.mode == "render":
        vpath = None
        if args.record_video:
            vpath = os.path.join(VIDEOS_DIR, "eval_rendered.mp4")
        run_rendered_match(
            red_policy, blue_policy, device,
            record=args.record_video, video_path=vpath,
            deterministic=args.deterministic)

    elif args.mode == "interactive":
        vpath = None
        if args.record_video:
            vpath = os.path.join(VIDEOS_DIR, "eval_interactive.mp4")
        run_interactive_match(
            blue_policy, device,
            record=args.record_video, video_path=vpath)


if __name__ == "__main__":
    main()
