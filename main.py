"""
main.py
────────────────────────────────────────────────────────────────────────────
Entry point for the VEX Override Neural Strategy Lab.

Modes
-----
  interactive   — keyboard-controlled 2v2 match (default)
  train         — MAPPO self-play training
  train-vis     — Training with periodic live visualization
  eval          — Evaluate a trained checkpoint
  heatmap       — Generate strategy heatmaps from a checkpoint

Usage
-----
  python main.py                              # interactive play
  python main.py --mode train                 # start/resume training
  python main.py --mode train --resume artifacts/models/latest.pt
  python main.py --mode train-vis
  python main.py --mode eval --checkpoint artifacts/models/best_policy.pt
  python main.py --mode eval --checkpoint artifacts/models/best_policy.pt --render
  python main.py --mode eval --checkpoint artifacts/models/best_policy.pt --interactive
  python main.py --mode heatmap --checkpoint artifacts/models/best_policy.pt
"""

import os
import sys
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="VEX Override Neural Strategy Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", type=str, default="interactive",
                   choices=["interactive", "train", "train-vis", "eval", "heatmap"],
                   help="Operation mode (default: interactive)")

    # Training options
    p.add_argument("--resume",           type=str, default=None,
                   help="Resume training from this checkpoint path")
    p.add_argument("--curriculum-stage", type=int, default=4,
                   help="Curriculum stage 1-6 (default: 4 = full 2v2)")
    p.add_argument("--total-steps",      type=int, default=None,
                   help="Total env steps to train (overrides config)")
    p.add_argument("--vis-every",        type=int, default=500,
                   help="(train-vis) Show visualization every N updates")

    # Evaluation options
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Checkpoint path for eval/heatmap")
    p.add_argument("--num-matches",  type=int, default=10,
                   help="Number of evaluation matches")
    p.add_argument("--record-video", action="store_true",
                   help="Save MP4 videos during evaluation")
    p.add_argument("--render",       action="store_true",
                   help="(eval) Show a rendered match window")
    p.add_argument("--interactive",  action="store_true",
                   help="(eval) Human vs AI mode (you control Red)")
    p.add_argument("--deterministic",action="store_true",
                   help="(eval) Use policy mode (no sampling noise)")

    # Device
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto | cpu | cuda | mps")

    return p.parse_args()


def run_interactive():
    """Launch the keyboard-controlled 2v2 simulator."""
    from simulation.simulator import OverrideSimulator
    sim = OverrideSimulator(headless=False)
    sim.run_interactive()


def run_train(args):
    """Launch MAPPO self-play training."""
    # Build argv for the training script
    train_argv = ["training/train_2policy_selfplay.py"]
    if args.resume:
        train_argv += ["--resume", args.resume]
    if args.curriculum_stage:
        train_argv += ["--curriculum-stage", str(args.curriculum_stage)]
    if args.total_steps:
        train_argv += ["--total-steps", str(args.total_steps)]
    train_argv += ["--device", args.device]

    # Import and call directly (avoids subprocess)
    old_argv = sys.argv
    sys.argv  = train_argv
    try:
        from training.train_2policy_selfplay import main as train_main
        train_main()
    finally:
        sys.argv = old_argv


def run_train_vis(args):
    """Launch training with visualization."""
    train_argv = ["training/train_with_visualization.py"]
    if args.resume:
        train_argv += ["--resume", args.resume]
    if args.total_steps:
        train_argv += ["--total-steps", str(args.total_steps)]
    train_argv += ["--vis-every", str(args.vis_every)]
    train_argv += ["--device", args.device]

    old_argv = sys.argv
    sys.argv  = train_argv
    try:
        from training.train_with_visualization import main as tv_main
        tv_main()
    finally:
        sys.argv = old_argv


def run_eval(args):
    """Launch evaluation."""
    from evaluation.evaluate import (
        load_policies, run_match_headless,
        run_rendered_match, run_interactive_match,
        select_device,
    )
    import numpy as np
    import os

    device = select_device(args.device)
    red_policy, blue_policy = load_policies(args.checkpoint, device)

    os.makedirs("artifacts/videos", exist_ok=True)

    if args.interactive:
        vpath = "artifacts/videos/interactive.mp4" if args.record_video else None
        run_interactive_match(blue_policy, device,
                              record=args.record_video, video_path=vpath)
    elif args.render:
        vpath = "artifacts/videos/rendered.mp4" if args.record_video else None
        run_rendered_match(red_policy, blue_policy, device,
                           record=args.record_video, video_path=vpath,
                           deterministic=args.deterministic)
    else:
        # Headless batch evaluation
        print(f"\nRunning {args.num_matches} headless evaluation matches...\n")
        results = []
        wins = {"red": 0, "blue": 0, "tie": 0}
        for i in range(args.num_matches):
            vpath = None
            if args.record_video:
                vpath = f"artifacts/videos/eval_match_{i+1:03d}.mp4"
            r = run_match_headless(
                red_policy, blue_policy, device,
                record=args.record_video, video_path=vpath,
                deterministic=args.deterministic, seed=i,
            )
            results.append(r)
            wins[r["winner"]] += 1
            print(f"  Match {i+1:2d}: {r['winner'].upper():<4} | "
                  f"Red {r['red_score']:>4} – {r['blue_score']:>4} Blue")

        print(f"\nResults:")
        for key, w in wins.items():
            print(f"  {key.capitalize():4}: {w:3} wins ({w/args.num_matches*100:.0f}%)")
        print(f"  Avg Red score:  {np.mean([r['red_score'] for r in results]):.1f}")
        print(f"  Avg Blue score: {np.mean([r['blue_score'] for r in results]):.1f}")


def run_heatmap(args):
    """Generate strategy heatmaps."""
    hm_argv = ["evaluation/trajectory_heatmap.py"]
    if args.checkpoint:
        hm_argv += ["--checkpoint", args.checkpoint]
    hm_argv += ["--num-matches", str(args.num_matches)]
    hm_argv += ["--device", args.device]

    old_argv = sys.argv
    sys.argv  = hm_argv
    try:
        from evaluation.trajectory_heatmap import main as hm_main
        hm_main()
    finally:
        sys.argv = old_argv


def main():
    args = parse_args()

    if args.mode == "interactive":
        print("[Main] Launching interactive simulator. Controls:")
        print("  WASD        = drive red1")
        print("  E           = intake")
        print("  Q           = score pin   |  Shift+Q = score cup")
        print("  F           = flip pin    |  Shift+F = flip cup")
        print("  T           = flip toggle")
        print("  M + 1..5    = match load (when in corner zone)")
        print("  R           = reset match")
        print("  ESC         = quit\n")
        run_interactive()

    elif args.mode == "train":
        run_train(args)

    elif args.mode == "train-vis":
        run_train_vis(args)

    elif args.mode == "eval":
        run_eval(args)

    elif args.mode == "heatmap":
        run_heatmap(args)

    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
