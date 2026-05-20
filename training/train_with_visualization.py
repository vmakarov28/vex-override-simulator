"""
training/train_with_visualization.py  (v9 – distributed rollout)
────────────────────────────────────────────────────────────────────────────
Workers collect full ROLLOUT_STEPS-step rollouts locally (CPU policy
inference + physics), then ship completed numpy buffers back once per update
cycle.  IPC drops from 16 384 pipe roundtrips to 32.

Observed timings (Ryzen 9 7900X + RTX 5080, 16 workers, v8.3):
  rollout  ~10  s
  ppo      ~9   s
  total    ~19  s/update  →  first log in ~3 min, 24 M steps in ~15 h

Run:
    python training/train_with_visualization.py
    python training/train_with_visualization.py --resume artifacts/models/latest.pt
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.env_wrapper import OverrideEnv, AGENT_IDS
from training.mappo       import MAPPOTrainer, RolloutBuffer
from training.vec_env     import SubprocVecEnv
from config.hyperparameters import (
    ROLLOUT_STEPS, TOTAL_ENV_STEPS,
    CHECKPOINT_EVERY, LOG_EVERY_UPDATES,
    MODELS_DIR, LOGS_DIR, VIDEOS_DIR,
    CONTROL_DT,
)

NUM_ENVS = 16

# Print per-update timing for the first N updates, then go silent.
# Set to 0 to disable entirely once training is confirmed stable.
DIAG_TIMING_UPDATES = 10

# v7: during the first EARLY_VIS_STEPS env steps, record a video every
# EARLY_VIS_INTERVAL updates (rather than args.vis_every).  Once past that
# cutoff we revert to the normal cadence.  Catches early behavioural
# regressions cheaply during the fast-learning phase.
EARLY_VIS_INTERVAL = 100
EARLY_VIS_STEPS    = 2_000_000


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--total-steps",  type=int,   default=TOTAL_ENV_STEPS)
    p.add_argument("--vis-every",    type=int,   default=400)
    p.add_argument("--vis-duration", type=float, default=120.0)
    p.add_argument("--device",       type=str,   default="auto")
    return p.parse_args()


def select_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(s)


def _serialize_policies(trainer: MAPPOTrainer) -> dict:
    """Convert GPU policy weights to numpy dicts for sending to CPU workers."""
    def to_np(model):
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return {
        "red_policy":  to_np(trainer.red_policy),
        "blue_policy": to_np(trainer.blue_policy),
        "red_critic":  to_np(trainer.red_critic),
        "blue_critic": to_np(trainer.blue_critic),
    }


# ─────────────────────────────────────────────────────────────────────────────
def run_rendered_match(trainer: MAPPOTrainer, duration_secs: float,
                       video_path: str = None):
    """Play one rendered match using the live GPU training policies."""
    import pygame

    rendered_env = OverrideEnv(headless=False, seed=int(time.time()) & 0xFFFF)
    obs   = rendered_env.reset()
    masks = rendered_env.get_action_masks()

    recorder = None
    if video_path:
        try:
            from evaluation.video_recorder import VideoRecorder
            recorder = VideoRecorder(video_path,
                                     rendered_env.sim.screen_w,
                                     rendered_env.sim.screen_h)
        except Exception as e:
            print(f"[Vis] Video recording unavailable: {e}")

    sim_time = 0.0
    clock    = pygame.time.Clock()
    running  = True
    done     = False

    print(f"[Vis] Watching for up to {duration_secs:.0f}s  (ESC to skip)")

    while running and sim_time < duration_secs and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        with torch.no_grad():
            agent_data = trainer.get_actions(obs, masks, deterministic=False)

        actions = {rid: (agent_data[rid]["cont"].cpu().numpy(),
                         agent_data[rid]["disc"].cpu().numpy())
                   for rid in AGENT_IDS}

        obs, _, done, info = rendered_env.step(actions)
        masks = rendered_env.get_action_masks()
        rendered_env.render()

        if recorder:
            frame = pygame.surfarray.array3d(
                rendered_env.sim.screen).transpose(1, 0, 2)
            recorder.write_frame(frame)

        sim_time += CONTROL_DT
        clock.tick(30)

    if done:
        rs = info.get("red_score", "?")
        bs = info.get("blue_score", "?")
        print(f"[Vis] Match over — Red {rs} : {bs} Blue")
        time.sleep(1.5)

    if recorder:
        recorder.close()
    rendered_env.close()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = select_device(args.device)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    print(f"[Train] Distributed rollout  envs={NUM_ENVS}  device={device}")

    trainer = MAPPOTrainer(device)
    if args.resume and os.path.exists(args.resume):
        trainer.load(args.resume)
        print(f"[Train] Resumed from {args.resume}")
    else:
        print("[Train] Starting from scratch.")

    # Workers spawn here; imports happen in parallel across all 32 processes
    vec = SubprocVecEnv(NUM_ENVS, base_seed=42)

    # Pre-allocate buffers (reset() each cycle — no CUDA realloc)
    red_bufs  = [RolloutBuffer(ROLLOUT_STEPS, device) for _ in range(NUM_ENVS)]
    blue_bufs = [RolloutBuffer(ROLLOUT_STEPS, device) for _ in range(NUM_ENVS)]

    log_file         = os.path.join(LOGS_DIR, "train_parallel_log.txt")
    comp_log_file    = os.path.join(LOGS_DIR, "reward_components_log.txt")
    accumulated_stats: dict = {}
    accumulated_components: dict = {}      # v7: per-signal reward sums
    n_component_rollouts: int = 0          # rollouts contributing to that sum
    episode_red:  list = []
    episode_blue: list = []

    print(f"[Train] Target {args.total_steps:,} steps | "
          f"vis every {args.vis_every} updates\n")

    try:
        while trainer.total_env_steps < args.total_steps:

            _t0 = time.time()

            # ── Serialise current GPU weights → send to workers ───────────────
            policy_data = _serialize_policies(trainer)

            # ── Workers collect ROLLOUT_STEPS steps in parallel ───────────────
            # Each worker: CPU inference + physics × 512 steps, ~1 s wall time.
            # Returns completed numpy buffer + episode scores.
            results = vec.collect_rollouts(policy_data, ROLLOUT_STEPS)

            _t1 = time.time()

            # ── Reset buffers and load worker data ────────────────────────────
            for rb in red_bufs:  rb.reset()
            for rb in blue_bufs: rb.reset()

            for i, result in enumerate(results):
                rd = result["red"]
                bd = result["blue"]

                for t in range(ROLLOUT_STEPS):
                    red_bufs[i].add(
                        obs0=rd["obs0"][t],   obs1=rd["obs1"][t],
                        cont0=rd["cont0"][t], cont1=rd["cont1"][t],
                        disc0=rd["disc0"][t], disc1=rd["disc1"][t],
                        lp0=rd["lp0"][t],     lp1=rd["lp1"][t],
                        reward=rd["reward"][t], value=rd["value"][t],
                        done=rd["done"][t],
                        mask0=rd["mask0"][t], mask1=rd["mask1"][t],
                    )
                    blue_bufs[i].add(
                        obs0=bd["obs0"][t],   obs1=bd["obs1"][t],
                        cont0=bd["cont0"][t], cont1=bd["cont1"][t],
                        disc0=bd["disc0"][t], disc1=bd["disc1"][t],
                        lp0=bd["lp0"][t],     lp1=bd["lp1"][t],
                        reward=bd["reward"][t], value=bd["value"][t],
                        done=bd["done"][t],
                        mask0=bd["mask0"][t], mask1=bd["mask1"][t],
                    )

                red_bufs[i].last_value  = result["last_rv"]
                blue_bufs[i].last_value = result["last_bv"]

                for rs, bs in result["episodes"]:
                    episode_red.append(rs)
                    episode_blue.append(bs)

                # v7: aggregate per-component reward sums across workers
                for cname, cval in result.get("reward_components", {}).items():
                    accumulated_components[cname] = (
                        accumulated_components.get(cname, 0.0) + cval)
            n_component_rollouts += NUM_ENVS

            trainer.total_env_steps += ROLLOUT_STEPS * NUM_ENVS

            _t2 = time.time()

            # ── Per-env GAE ───────────────────────────────────────────────────
            for i in range(NUM_ENVS):
                red_bufs[i].compute_gae()
                blue_bufs[i].compute_gae()

            _t3 = time.time()

            # ── Joint PPO update ──────────────────────────────────────────────
            stats = trainer.update_multi_env(red_bufs, blue_bufs)

            _t4 = time.time()

            if trainer.total_updates <= DIAG_TIMING_UPDATES:
                print(f"[Diag] upd={trainer.total_updates:4d} | "
                      f"collect={_t1-_t0:5.1f}s  load={_t2-_t1:4.1f}s  "
                      f"gae={_t3-_t2:4.1f}s  ppo={_t4-_t3:4.1f}s  "
                      f"total={_t4-_t0:5.1f}s", flush=True)

            for k, v in stats.items():
                accumulated_stats[k] = accumulated_stats.get(k, 0.0) + v

            n_up = trainer.total_updates

            # ── Logging ───────────────────────────────────────────────────────
            if n_up % LOG_EVERY_UPDATES == 0 and n_up > 0:
                avg      = {k: v / LOG_EVERY_UPDATES
                            for k, v in accumulated_stats.items()}
                n_eps    = len(episode_red)
                avg_red  = np.mean(episode_red)  if episode_red  else 0.0
                avg_blue = np.mean(episode_blue) if episode_blue else 0.0
                msg = (f"Step {trainer.total_env_steps:,} | "
                       f"R_loss={avg.get('red_policy_loss',  0):.4f} "
                       f"B_loss={avg.get('blue_policy_loss', 0):.4f} "
                       f"R_ent={avg.get('red_entropy', 0):.3f} | "
                       f"AvgScore R={avg_red:.1f} B={avg_blue:.1f} ({n_eps} eps)")
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")

                # v7: per-component reward signal logging.  Each value is
                # the *average reward delta per rollout* contributed by that
                # signal across all 4 robots and ROLLOUT_STEPS steps.  Quick
                # diagnostic for which signals are firing and at what magnitude.
                if accumulated_components and n_component_rollouts > 0:
                    avg_comp = {k: v / float(n_component_rollouts)
                                for k, v in accumulated_components.items()}
                    sorted_comp = sorted(avg_comp.items(),
                                         key=lambda kv: abs(kv[1]),
                                         reverse=True)
                    comp_line = f"Step {trainer.total_env_steps:,} | " + \
                        " ".join(f"{k}={v:+.3f}" for k, v in sorted_comp)
                    print("[Rwd] " + comp_line)
                    with open(comp_log_file, "a") as f:
                        f.write(comp_line + "\n")

                accumulated_stats      = {}
                accumulated_components = {}
                n_component_rollouts   = 0
                episode_red            = []
                episode_blue           = []

            # ── Checkpoint ────────────────────────────────────────────────────
            if n_up % CHECKPOINT_EVERY == 0 and n_up > 0:
                trainer.save(os.path.join(MODELS_DIR, "latest.pt"))
                trainer.save(os.path.join(MODELS_DIR,
                                          f"checkpoint_{n_up:06d}.pt"))

            # ── Visualization ─────────────────────────────────────────────────
            # v7: adaptive cadence — during the first EARLY_VIS_STEPS env
            # steps, record every EARLY_VIS_INTERVAL updates.  After that,
            # fall back to args.vis_every.
            if trainer.total_env_steps < EARLY_VIS_STEPS:
                vis_period = EARLY_VIS_INTERVAL
            else:
                vis_period = args.vis_every
            if n_up % vis_period == 0 and n_up > 0:
                video_path = os.path.join(VIDEOS_DIR, f"vis_{n_up:06d}.mp4")
                try:
                    run_rendered_match(trainer,
                                       duration_secs=args.vis_duration,
                                       video_path=video_path)
                except Exception as e:
                    print(f"[Vis] Visualization error (non-fatal): {e}")

    finally:
        vec.close()

    trainer.save(os.path.join(MODELS_DIR, "final.pt"))
    print(f"\n[Train] Done. {trainer.total_env_steps:,} steps, "
          f"{trainer.total_updates} updates.")

    # v7: record one final full-match video using the trained policies.
    final_video = os.path.join(VIDEOS_DIR, "final_result.mp4")
    print(f"[Train] Recording final result video → {final_video}")
    try:
        run_rendered_match(trainer,
                           duration_secs=float(args.vis_duration),
                           video_path=final_video)
    except Exception as e:
        print(f"[Train] Final-video recording failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
