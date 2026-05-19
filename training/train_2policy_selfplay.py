"""
training/train_2policy_selfplay.py
────────────────────────────────────────────────────────────────────────────
Main MAPPO self-play training loop for VEX Override.

Run with:
    python training/train_2policy_selfplay.py
    python training/train_2policy_selfplay.py --resume artifacts/models/latest.pt
    python training/train_2policy_selfplay.py --curriculum-stage 3

Training flow per iteration:
  1. Collect ROLLOUT_STEPS env steps using the current red & blue policies.
  2. Occasionally swap blue policy with a frozen pool checkpoint (self-play).
  3. Run PPO_EPOCHS update passes on both alliance policies.
  4. Log stats, save checkpoint, run short evaluation.
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.env_wrapper import OverrideEnv, AGENT_IDS
from training.mappo import MAPPOTrainer
from utils.opponent_pool import OpponentPool
from training.network import Policy
from config.hyperparameters import (
    ROLLOUT_STEPS, TOTAL_ENV_STEPS,
    CHECKPOINT_EVERY, EVAL_EVERY_UPDATES, EVAL_NUM_MATCHES,
    LOG_EVERY_UPDATES, RECORD_VIDEO,
    MODELS_DIR, LOGS_DIR,
    CURRICULUM_STAGES, POOL_SAMPLE_PROB,
)


def parse_args():
    p = argparse.ArgumentParser(description="VEX Override MAPPO Self-Play Training")
    p.add_argument("--resume",            type=str,  default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--curriculum-stage",  type=int,  default=4,
                   help="Curriculum stage to run (1-6, default: 4 = full 2v2)")
    p.add_argument("--total-steps",       type=int,  default=TOTAL_ENV_STEPS,
                   help="Total environment steps to train for")
    p.add_argument("--no-pool",           action="store_true",
                   help="Disable opponent pool (pure self-play only)")
    p.add_argument("--device",            type=str,  default="auto",
                   help="Device: auto | cpu | cuda | mps")
    return p.parse_args()


def select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def run_evaluation(trainer, n_matches: int = 5, record: bool = False,
                   video_suffix: str = "") -> dict:
    """Run N evaluation matches and return statistics."""
    from evaluation.evaluate import run_match_headless
    wins = {"red": 0, "blue": 0, "tie": 0}
    scores = []
    for i in range(n_matches):
        result = run_match_headless(
            red_policy=trainer.red_policy,
            blue_policy=trainer.blue_policy,
            device=trainer.device,
            record=record,
            video_path=(f"{MODELS_DIR}/../videos/eval_{video_suffix}_{i}.mp4"
                        if record else None),
        )
        wins[result["winner"]] += 1
        scores.append((result["red_score"], result["blue_score"]))

    avg_r = np.mean([s[0] for s in scores])
    avg_b = np.mean([s[1] for s in scores])
    return {
        "wins": wins,
        "win_rate_red":  wins["red"]  / n_matches,
        "win_rate_blue": wins["blue"] / n_matches,
        "avg_red_score":  avg_r,
        "avg_blue_score": avg_b,
    }


def log_stats(step: int, stats: dict, eval_stats: dict = None,
              log_file: str = None):
    """Print and optionally write stats to a log file."""
    line_parts = [
        f"Step {step:>8,}",
        f"R_ploss={stats.get('red_policy_loss',0):.4f}",
        f"R_vloss={stats.get('red_value_loss',0):.4f}",
        f"R_ent={stats.get('red_entropy',0):.3f}",
        f"B_ploss={stats.get('blue_policy_loss',0):.4f}",
        f"B_vloss={stats.get('blue_value_loss',0):.4f}",
        f"B_ent={stats.get('blue_entropy',0):.3f}",
    ]
    if eval_stats:
        line_parts += [
            f"R_win={eval_stats['win_rate_red']:.2f}",
            f"R_score={eval_stats['avg_red_score']:.1f}",
            f"B_score={eval_stats['avg_blue_score']:.1f}",
        ]
    line = " | ".join(line_parts)
    print(line)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def main():
    args  = parse_args()
    device = select_device(args.device)
    print(f"[Train] Device: {device}")

    # Create output directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    # Stage config
    stage = next((s for s in CURRICULUM_STAGES
                  if s["id"] == args.curriculum_stage), CURRICULUM_STAGES[3])
    print(f"[Train] Curriculum stage {stage['id']}: {stage['name']}")
    print(f"        {stage['description']}")

    late_start = stage.get("late_start_prob", 0.0)

    # Build environment and trainer
    env     = OverrideEnv(headless=True, seed=42, late_start_prob=late_start)
    trainer = MAPPOTrainer(device)

    # Opponent pool
    pool      = None
    pool_path = os.path.join(MODELS_DIR, "pool.pt")
    if not args.no_pool:
        pool = OpponentPool(Policy, device)
        pool.load(pool_path)

    # Reusable Policy instance for pool inference (avoids repeated allocation)
    pool_blue_instance = Policy().to(device) if pool else None

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        trainer.load(args.resume)
        if pool:
            pool.load(pool_path)
    else:
        print("[Train] Starting from scratch.")

    log_file = os.path.join(LOGS_DIR, "train_log.txt")
    best_red_score = 0.0
    last_eval_stats = None
    accumulated_stats = {}

    obs   = env.reset()
    masks = env.get_action_masks()

    print(f"\n[Train] Beginning training — target {args.total_steps:,} env steps\n")
    t_start = time.time()

    while trainer.total_env_steps < args.total_steps:
        # ── Decide opponent policy for this rollout ────────────────────────
        # With probability POOL_SAMPLE_PROB, blue uses a historical policy so
        # red learns to beat a range of opponents (prevents mode collapse).
        # Blue's stored log_probs come from whichever policy took the action;
        # the PPO update then optimises the *current* blue policy against those
        # old log_probs — valid importance-weighted PPO.
        blue_override = None
        if pool and not pool.is_empty() and random.random() < POOL_SAMPLE_PROB:
            blue_override = pool.sample(pool_blue_instance)

        # ── Collect one rollout ────────────────────────────────────────────
        trainer.reset_buffers()
        episode_rewards = {"red": 0.0, "blue": 0.0}
        ep_steps = 0

        for _ in range(ROLLOUT_STEPS):
            # Get actions (blue may use a frozen pool policy)
            agent_data = trainer.get_actions(obs, masks,
                                             blue_policy_override=blue_override)

            # Decode to env action format
            actions = {}
            for rid in AGENT_IDS:
                c = agent_data[rid]["cont"].cpu().numpy()
                d = agent_data[rid]["disc"].cpu().numpy()
                actions[rid] = (c, d)

            # Step environment
            next_obs, rewards, done, info = env.step(actions)

            episode_rewards["red"]  += (rewards["red1"] + rewards["red2"]) / 2
            episode_rewards["blue"] += (rewards["blue1"] + rewards["blue2"]) / 2
            ep_steps += 1

            # Store in buffers
            trainer.store_transition(agent_data, rewards, done)

            # Reset if episode ended
            if done:
                obs   = env.reset()
                masks = env.get_action_masks()
                episode_rewards = {"red": 0.0, "blue": 0.0}
                ep_steps = 0
            else:
                obs   = next_obs
                masks = env.get_action_masks()

        # Bootstrap last value
        trainer.set_last_values(obs)

        # ── PPO update ─────────────────────────────────────────────────────
        update_stats = trainer.update()

        # Accumulate for logging
        for k, v in update_stats.items():
            accumulated_stats[k] = accumulated_stats.get(k, 0) + v

        # ── Logging ────────────────────────────────────────────────────────
        n_updates = trainer.total_updates
        if n_updates % LOG_EVERY_UPDATES == 0:
            avg_stats = {k: v / LOG_EVERY_UPDATES
                         for k, v in accumulated_stats.items()}
            log_stats(trainer.total_env_steps, avg_stats, last_eval_stats,
                      log_file)
            accumulated_stats = {}

        # ── Checkpoint + pool ──────────────────────────────────────────────
        if n_updates % CHECKPOINT_EVERY == 0:
            ck_path = os.path.join(MODELS_DIR,
                                   f"checkpoint_{n_updates:06d}.pt")
            trainer.save(ck_path)
            # Always save "latest" for easy resuming
            trainer.save(os.path.join(MODELS_DIR, "latest.pt"))
            if pool:
                pool.add(trainer.red_policy)   # add red policy to pool
                pool.save(pool_path)

        # ── Evaluation ────────────────────────────────────────────────────
        if n_updates % EVAL_EVERY_UPDATES == 0:
            suffix = f"{n_updates:06d}"
            try:
                eval_stats = run_evaluation(
                    trainer, n_matches=EVAL_NUM_MATCHES,
                    record=RECORD_VIDEO, video_suffix=suffix)
                last_eval_stats = eval_stats

                # Save best model
                if eval_stats["avg_red_score"] > best_red_score:
                    best_red_score = eval_stats["avg_red_score"]
                    trainer.save(os.path.join(MODELS_DIR, "best_policy.pt"))
                    print(f"[Train] ★ New best! Red avg score: {best_red_score:.1f}")
            except Exception as e:
                print(f"[Train] Evaluation error: {e}")

        # pool_blue_instance is refreshed at the top of each rollout loop above

    # ── Final save ────────────────────────────────────────────────────────
    trainer.save(os.path.join(MODELS_DIR, "final.pt"))
    elapsed = time.time() - t_start
    print(f"\n[Train] Training complete in {elapsed/3600:.1f} h — "
          f"{trainer.total_env_steps:,} steps, {trainer.total_updates} updates.")
    env.close()


if __name__ == "__main__":
    main()
