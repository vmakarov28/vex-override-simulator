"""
run_with_bots.py
================
Visualisation runner that lets you assign any subset of the four
robots to the rule-based HeuristicBot.  Robots NOT in --bots either
use the trained policy from a checkpoint (--policy path) or stand
still (if no checkpoint given and the robot wasn't assigned a bot).

Examples
--------
    # All four robots controlled by the heuristic bot — no policy needed.
    python run_with_bots.py --bots all

    # Just red1 is a bot; the other three use the latest checkpoint.
    python run_with_bots.py --bots red1 --policy artifacts/models/final.pt

    # Bot vs Bot (red bots) against policy (blue bots).
    python run_with_bots.py --bots red1,red2 --policy artifacts/models/final.pt

    # No checkpoint, no bots → all robots idle (just lets you watch the field).
    python run_with_bots.py

Match runs once at full speed for up to --duration seconds, ESC to quit early.
A video is written to --video (default artifacts/videos/bots_run.mp4).
"""

import os
import sys
import warnings

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message="pkg_resources", category=UserWarning)

import argparse
import time
import numpy as np

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.env_wrapper import OverrideEnv, AGENT_IDS
from agents.heuristic_bot import HeuristicBot
from config.hyperparameters import CONTROL_DT


VALID_BOT_NAMES = {"red1", "red2", "blue1", "blue2", "all", "none"}


def parse_bots(arg: str):
    """Parse --bots into a set of robot_ids to control via HeuristicBot."""
    if not arg or arg.lower() in ("none", ""):
        return set()
    if arg.lower() == "all":
        return set(AGENT_IDS)
    out = set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in AGENT_IDS:
            raise ValueError(f"Unknown robot_id '{tok}'.  Valid: {AGENT_IDS} or 'all'/'none'")
        out.add(tok)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Run a match with rule-based bots controlling some/all robots.")
    p.add_argument("--bots",     type=str, default="all",
                   help="Comma-separated robot_ids to control with HeuristicBot.  "
                        "'all' = all four, 'none' = no bots.  Default: all.")
    p.add_argument("--policy",   type=str, default=None,
                   help="Optional checkpoint path; non-bot robots use this policy.")
    p.add_argument("--duration", type=float, default=120.0,
                   help="Max real-time seconds to run.  Default: 120 s.")
    p.add_argument("--video",    type=str, default="artifacts/videos/bots_run.mp4",
                   help="Output video path.  Use '' to skip recording.")
    p.add_argument("--seed",     type=int, default=None,
                   help="RNG seed.  Default: time-derived.")
    p.add_argument("--verbose",  action="store_true",
                   help="Print each bot's chosen reason every step (very noisy).")
    return p.parse_args()


# --------------------------------------------------------------------- #
def main():
    args = parse_args()
    bot_ids = parse_bots(args.bots)
    seed = args.seed if args.seed is not None else int(time.time()) & 0xFFFF

    print(f"[Bots] Bot-controlled robots: {sorted(bot_ids) if bot_ids else '(none)'}")
    if args.policy:
        print(f"[Bots] Other robots use policy: {args.policy}")
    else:
        idle_ids = sorted(set(AGENT_IDS) - bot_ids)
        if idle_ids:
            print(f"[Bots] No --policy given → robots {idle_ids} will idle (zero action)")

    # Build env (rendered, single-process).
    env = OverrideEnv(headless=False, seed=seed)
    obs   = env.reset()
    masks = env.get_action_masks()

    # Build bots after reset so sim.robots is populated.
    bots = {rid: HeuristicBot(rid, env.sim) for rid in bot_ids}

    # Build trainer-style policy loader for non-bot robots, if requested.
    trainer = None
    if args.policy and bot_ids != set(AGENT_IDS):
        # Lazy imports so a pure-bot run doesn't require torch.
        import torch
        from training.mappo import MAPPOTrainer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = MAPPOTrainer(device)
        try:
            trainer.load(args.policy)
            print(f"[Bots] Loaded policy from {args.policy}")
        except Exception as e:
            print(f"[Bots] WARNING: could not load checkpoint ({e}) — non-bot robots will idle")
            trainer = None

    # Video recorder
    recorder = None
    if args.video:
        try:
            import pygame  # imported via env render anyway
            os.makedirs(os.path.dirname(args.video), exist_ok=True)
            from evaluation.video_recorder import VideoRecorder
            recorder = VideoRecorder(args.video, env.sim.screen_w, env.sim.screen_h)
            print(f"[Bots] Recording video → {args.video}")
        except Exception as e:
            print(f"[Bots] Video recording unavailable: {e}")
            recorder = None

    import pygame  # for event loop
    clock    = pygame.time.Clock()
    sim_time = 0.0
    done     = False
    running  = True

    print(f"[Bots] Running for up to {args.duration:.0f}s — ESC to skip\n")

    while running and sim_time < args.duration and not done:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        # --- gather actions ---------------------------------------- #
        actions = {}
        # 1) bots
        for rid, bot in bots.items():
            cont, disc = bot.get_policy_action()
            actions[rid] = (cont, disc)
            if args.verbose:
                print(f"  [{rid}] {bot.last_reason}")

        # 2) trained policy for the rest (if available)
        if trainer is not None:
            need_policy_ids = [r for r in AGENT_IDS if r not in bots]
            if need_policy_ids:
                import torch
                with torch.no_grad():
                    agent_data = trainer.get_actions(obs, masks, deterministic=True)
                for rid in need_policy_ids:
                    actions[rid] = (agent_data[rid]["cont"].cpu().numpy(),
                                    agent_data[rid]["disc"].cpu().numpy())

        # 3) anyone still missing = idle (zero action)
        for rid in AGENT_IDS:
            if rid not in actions:
                actions[rid] = (np.zeros(2, dtype=np.float32),
                                np.zeros(7, dtype=np.float32))

        # --- step env -------------------------------------------- #
        obs, _, done, info = env.step(actions)
        masks = env.get_action_masks()
        env.render()

        if recorder:
            frame = pygame.surfarray.array3d(env.sim.screen).transpose(1, 0, 2)
            recorder.write_frame(frame)

        sim_time += CONTROL_DT
        clock.tick(30)  # cap render at 30 fps

    if done:
        rs = info.get("red_score", "?")
        bs = info.get("blue_score", "?")
        print(f"\n[Bots] Match over — Red {rs} : {bs} Blue")
    else:
        print(f"\n[Bots] Stopped early at t={sim_time:.1f}s (live score Red {env.sim.red_score} : {env.sim.blue_score} Blue)")

    if recorder:
        recorder.close()
        print(f"[Bots] Video saved → {args.video}")
    env.close()


if __name__ == "__main__":
    main()
