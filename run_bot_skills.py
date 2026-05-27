"""
run_bot_skills.py
=================
Watch the HeuristicBot play a VEX Robot Skills run — one robot alone on
the field, 60 seconds, single score counter, score screen at the end.

Usage
-----
    # Default: one skills run, rendered window
    python run_bot_skills.py

    # Several back-to-back runs (useful for seeing variance)
    python run_bot_skills.py --matches 5

    # Specific seed
    python run_bot_skills.py --seed 1234

    # Headless (no window) — faster, prints scores only
    python run_bot_skills.py --headless --matches 20

Controls (rendered window)
--------------------------
    R    — reset / run again
    ESC  — quit

The HeuristicBot is fully deterministic; the seed only controls the
field's initial element jitter.
"""

import os
import sys
import time
import argparse
import warnings

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message="pkg_resources", category=UserWarning)

# Make project root importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from simulation.simulator import OverrideSimulator
from agents.heuristic_bot import HeuristicBot
from config.game_rules import SKILLS_SECONDS
from config.hyperparameters import CONTROL_DT


def parse_args():
    p = argparse.ArgumentParser(
        description="Run VEX Robot Skills with the HeuristicBot driving."
    )
    p.add_argument("--matches",  type=int, default=1,
                   help="Number of back-to-back skills runs (default: 1).")
    p.add_argument("--duration", type=float, default=float(SKILLS_SECONDS),
                   help=f"Sim-seconds per run (default: {SKILLS_SECONDS}, the "
                        f"official Skills length).")
    p.add_argument("--seed",     type=int, default=None,
                   help="RNG seed for first match (subsequent matches use "
                        "seed+1, +2, ...).  Default: time-based.")
    p.add_argument("--headless", action="store_true",
                   help="Run without a window — faster, prints final scores.")
    p.add_argument("--video",    type=str,
                   default="artifacts/videos/skills_run.mp4",
                   help="Output video path.  Use '' to skip recording.")
    p.add_argument("--auto-close", action="store_true",
                   help="Skip the post-run wait-on-score-screen "
                        "(useful when only saving a video).")
    return p.parse_args()


def run_one(sim: OverrideSimulator, bot: HeuristicBot, duration: float,
            render: bool, recorder=None, auto_close: bool = False) -> tuple:
    """Drive one skills run.  Returns (score, completed, still_running)."""
    if render:
        import pygame
        clock = pygame.time.Clock()
    sim_time     = 0.0
    still_running = True
    completed    = False

    # Force the timer to start immediately — the bot won't tap a key, and
    # OverrideSimulator only starts the clock once it sees driver input.
    # Setting timer_started = True bypasses that.
    sim.timer_started = True

    while sim_time < duration and not sim.match_over and still_running:
        if render:
            import pygame
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    still_running = False
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    still_running = False
        action = bot.get_sim_action()
        sim.step(CONTROL_DT, [action])
        if render:
            sim.render()
            if recorder is not None:
                import pygame
                frame = pygame.surfarray.array3d(sim.screen).transpose(1, 0, 2)
                recorder.write_frame(frame)
            clock.tick(30)
        sim_time += CONTROL_DT

    # If we exited via time limit (not match_over signal), finalise SC5b.
    if not sim.match_over:
        final = sim.rules_engine.calculate_final_score(
            sim.goals, sim.toggles, sim.robots)
        sim.red_score  = final["red"]
        sim.blue_score = final["blue"]
        sim.match_over = True
        sim.match_phase = "ended"
    completed = True

    # Show the final score screen briefly so the recorder captures it.
    if render and still_running and recorder is not None:
        import pygame
        for _ in range(60):   # 2 s at 30 fps
            sim.render()
            frame = pygame.surfarray.array3d(sim.screen).transpose(1, 0, 2)
            recorder.write_frame(frame)
            clock.tick(30)

    # Hold on the score screen until R (reset) or ESC / window close —
    # unless auto-close is set (e.g. when only saving a video).
    if render and still_running and not auto_close:
        sim.render()
        import pygame
        waiting = True
        while waiting and still_running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    still_running = False
                    waiting = False
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        still_running = False
                        waiting = False
                    elif ev.key == pygame.K_r:
                        waiting = False  # caller will call sim.reset()
            sim.render()
            clock.tick(30)

    return int(sim.red_score), completed, still_running


def main():
    args      = parse_args()
    base_seed = args.seed if args.seed is not None else (int(time.time()) & 0xFFFF)
    render    = not args.headless

    print(f"[BotSkills] Matches    : {args.matches}")
    print(f"[BotSkills] Duration   : {args.duration:.0f}s per run")
    print(f"[BotSkills] Seed (base): {base_seed}")
    print(f"[BotSkills] Rendering  : {'on' if render else 'OFF (headless)'}")

    sim  = OverrideSimulator(headless=args.headless, skills_mode=True)
    sim.rng = np.random.default_rng(base_seed)
    bot  = HeuristicBot("red1", sim)

    # Optional video recorder (skipped if headless or --video '').
    recorder = None
    if render and args.video:
        try:
            os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
            from evaluation.video_recorder import VideoRecorder
            recorder = VideoRecorder(args.video, sim.screen_w, sim.screen_h)
            print(f"[BotSkills] Recording video -> {args.video}")
        except Exception as e:
            print(f"[BotSkills] Video recording unavailable: {e}")
            recorder = None

    scores = []
    for i in range(args.matches):
        seed = base_seed + i
        if i > 0:
            sim.reset()
            bot.reset()
        sim.rng = np.random.default_rng(seed)

        print(f"\n[BotSkills] Run {i+1}/{args.matches}  seed={seed}")
        score, completed, still_running = run_one(
            sim, bot, args.duration, render=render, recorder=recorder,
            auto_close=args.auto_close,
        )
        scores.append(score)
        print(f"[BotSkills] Run {i+1} -- Skills Score: {score}")
        if not still_running:
            print("[BotSkills] ESC / quit — stopping session.")
            break

    if scores:
        arr = np.array(scores)
        print(f"\n[BotSkills] {'='*46}")
        print(f"[BotSkills] {len(scores)} run(s)")
        print(f"[BotSkills]   mean = {arr.mean():6.1f}")
        print(f"[BotSkills]   std  = {arr.std():6.1f}")
        print(f"[BotSkills]   min  = {arr.min()}")
        print(f"[BotSkills]   max  = {arr.max()}")
        print(f"[BotSkills] {'='*46}")

    if recorder is not None:
        recorder.close()
        print(f"[BotSkills] Video saved -> {args.video}")

    if render:
        import pygame
        pygame.quit()


if __name__ == "__main__":
    main()
