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

    # Run three back-to-back matches, each with a fresh field.
    python run_with_bots.py --bots all --matches 3

    # No checkpoint, no bots → all robots idle (just lets you watch the field).
    python run_with_bots.py

Each match runs for up to --duration seconds.  ESC quits the whole session.
A video is written to --video (default artifacts/videos/bots_run.mp4).
With --matches > 1 each video is suffixed _m1, _m2, … before the extension.

Logging (always on)
-------------------
Every run writes TWO log files per match under artifacts/logs/:
  * `bots_events_<ts>_m<N>.csv`   — structured event log (SCORE, INTAKE,
                                     DROP, FLIP_PIN, FLIP_CUP, TOGGLE,
                                     PHASE) for easy post-hoc analysis.
                                     Columns: t_sim,step,robot,event,
                                     detail,x,y,score_red,score_blue.
  * `bots_reasons_<ts>_m<N>.log`  — per-step textual `last_reason` of
                                     each bot (the original v9.4 trace).
                                     Useful for figuring out *why* a bot
                                     made a particular decision.

Pass `--no-reasons-log` to skip the reasons log (events CSV is always
written — it's lightweight and indispensable for debugging).
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
from agents.bot_event_log import BotEventLog
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


def _video_path_for_match(base: str, match_idx: int, total: int) -> str:
    """Inject _m<N> suffix when running multiple matches."""
    if total <= 1 or not base:
        return base
    root, ext = os.path.splitext(base)
    return f"{root}_m{match_idx}{ext}"


def parse_args():
    p = argparse.ArgumentParser(description="Run a match with rule-based bots controlling some/all robots.")
    p.add_argument("--bots",     type=str, default="all",
                   help="Comma-separated robot_ids to control with HeuristicBot.  "
                        "'all' = all four, 'none' = no bots.  Default: all.")
    p.add_argument("--policy",   type=str, default=None,
                   help="Optional checkpoint path; non-bot robots use this policy.")
    p.add_argument("--duration", type=float, default=120.0,
                   help="Max sim-seconds per match.  Default: 120 s.")
    p.add_argument("--matches",  type=int, default=1,
                   help="Number of back-to-back matches to run.  Default: 1.")
    p.add_argument("--video",    type=str, default="artifacts/videos/bots_run.mp4",
                   help="Output video path.  Use '' to skip recording.")
    p.add_argument("--seed",     type=int, default=None,
                   help="RNG seed for the first match.  Subsequent matches use seed+1, etc.  "
                        "Note: HeuristicBot is fully deterministic — the seed only "
                        "controls field initialisation (element starting positions).")
    p.add_argument("--no-reasons-log", action="store_true",
                   help="Skip the per-step textual reasons log (events CSV is "
                        "always written under artifacts/logs/).")
    return p.parse_args()


# --------------------------------------------------------------------- #
def run_match(env, bots, trainer, args, match_idx: int,
              reasons_log=None, event_log=None, recorder=None):
    """Run a single match.  Returns (red_score, blue_score, completed, running, stats)."""
    import pygame
    clock    = pygame.time.Clock()
    sim_time = 0.0
    done     = False
    running  = True
    step_idx = 0
    info     = {}

    # Stats accumulators
    intakes  = {rid: 0 for rid in AGENT_IDS}
    drops    = {rid: 0 for rid in AGENT_IDS}
    pre_carry = {r.robot_id: (r.carrying_pin is not None or r.carrying_cup is not None)
                 for r in env.sim.robots}

    print(f"[Bots] Match {match_idx} — running for up to {args.duration:.0f}s — ESC to quit\n")

    while running and sim_time < args.duration and not done:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        # Snapshot pre-step state for event logger
        if event_log is not None:
            event_log.snapshot_pre(env.sim)

        # --- gather actions --------------------------------------- #
        actions = {}

        obs   = env._last_obs   if hasattr(env, "_last_obs")   else None
        masks = env._last_masks if hasattr(env, "_last_masks") else None

        # 1) bots
        for rid, bot in bots.items():
            cont, disc = bot.get_policy_action()
            actions[rid] = (cont, disc)
            if reasons_log is not None:
                reasons_log.write(
                    f"t={sim_time:7.3f} step={step_idx:6d}  [{rid}]  {bot.last_reason}\n"
                )

        # 2) trained policy for the rest (if available)
        if trainer is not None and obs is not None and masks is not None:
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
        obs_new, _, done, info = env.step(actions)
        env._last_obs   = obs_new
        env._last_masks = env.get_action_masks()
        env.render()

        # Diff and emit events from snapshot
        if event_log is not None:
            event_log.diff_and_emit(env.sim, sim_time, step_idx, bots=bots)

        if recorder:
            frame = pygame.surfarray.array3d(env.sim.screen).transpose(1, 0, 2)
            recorder.write_frame(frame)

        sim_time += CONTROL_DT
        step_idx += 1
        clock.tick(30)

    rs = info.get("red_score",  env.sim.red_score)
    bs = info.get("blue_score", env.sim.blue_score)
    completed = bool(done)

    # Build a per-robot stats dict
    stats = {
        "intakes":  dict(event_log.intakes_per_robot)  if event_log else {},
        "scores":   dict(event_log.scores_per_robot)   if event_log else {},
        "drops":    dict(event_log.drops_per_robot)    if event_log else {},
        "flips":    dict(event_log.flips_per_robot)    if event_log else {},
        "toggles":  dict(event_log.toggles_per_robot)  if event_log else {},
    }
    return rs, bs, completed, running, stats


def _print_match_summary(match_idx: int, rs: int, bs: int, stats: dict):
    winner = "Red" if rs > bs else ("Blue" if bs > rs else "Tie")
    print(f"\n[Bots] Match {match_idx} — Red {rs:3d} : {bs:3d} Blue  [{winner}]")
    print(f"  {'robot':<8}{'scores':>8}{'intakes':>10}{'drops':>8}{'flips':>8}{'toggles':>10}")
    for rid in AGENT_IDS:
        s  = stats.get("scores",  {}).get(rid, 0)
        i  = stats.get("intakes", {}).get(rid, 0)
        d  = stats.get("drops",   {}).get(rid, 0)
        f  = stats.get("flips",   {}).get(rid, 0)
        tg = stats.get("toggles", {}).get(rid, 0)
        print(f"  {rid:<8}{s:>8}{i:>10}{d:>8}{f:>8}{tg:>10}")


# --------------------------------------------------------------------- #
def main():
    args    = parse_args()
    bot_ids = parse_bots(args.bots)
    base_seed = args.seed if args.seed is not None else int(time.time()) & 0xFFFF

    print(f"[Bots] Bot-controlled robots : {sorted(bot_ids) if bot_ids else '(none)'}")
    print(f"[Bots] Matches to run        : {args.matches}")
    if args.policy:
        print(f"[Bots] Other robots use policy: {args.policy}")
    else:
        idle_ids = sorted(set(AGENT_IDS) - bot_ids)
        if idle_ids:
            print(f"[Bots] No --policy given → robots {idle_ids} will idle (zero action)")

    # ── Build env once — re-use across matches ────────────────────────── #
    seed = base_seed
    env  = OverrideEnv(headless=False, seed=seed)
    obs  = env.reset()
    env._last_obs   = obs
    env._last_masks = env.get_action_masks()

    # Build bots — created once, reset between matches
    bots = {rid: HeuristicBot(rid, env.sim) for rid in bot_ids}

    # Policy loader (optional, lazy torch import)
    trainer = None
    if args.policy and bot_ids != set(AGENT_IDS):
        import torch
        from training.mappo import MAPPOTrainer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = MAPPOTrainer(device)
        try:
            trainer.load(args.policy)
            print(f"[Bots] Loaded policy from {args.policy}")
        except Exception as e:
            print(f"[Bots] WARNING: could not load checkpoint ({e}) — non-bot robots idle")
            trainer = None

    import pygame

    results = []
    all_stats = []

    for match_idx in range(1, args.matches + 1):
        print(f"\n{'='*60}")
        print(f"[Bots] Starting match {match_idx}/{args.matches}  seed={seed}")
        print(f"{'='*60}")

        # ── Reasons log (per-step textual; opt-out via --no-reasons-log) ── #
        reasons_log = None
        reasons_path = None
        if not args.no_reasons_log:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "artifacts", "logs")
            os.makedirs(log_dir, exist_ok=True)
            ts           = time.strftime("%Y%m%d_%H%M%S")
            reasons_path = os.path.join(log_dir, f"bots_reasons_{ts}_m{match_idx}.log")
            reasons_log  = open(reasons_path, "w", buffering=1)
            print(f"[Bots] Reasons log  → {reasons_path}")
            reasons_log.write(
                f"# bots_reasons  match={match_idx}  seed={seed}  "
                f"bots={sorted(bot_ids)}\n"
                f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

        # ── Event CSV (always on; structured, parsable) ────────────────── #
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "artifacts", "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts           = time.strftime("%Y%m%d_%H%M%S")
        events_path  = os.path.join(log_dir, f"bots_events_{ts}_m{match_idx}.csv")
        event_log    = BotEventLog(events_path)
        print(f"[Bots] Events CSV   → {events_path}")

        # ── Video recorder (one file per match) ───────────────────────── #
        recorder  = None
        vid_path  = _video_path_for_match(args.video, match_idx, args.matches)
        if vid_path:
            try:
                os.makedirs(os.path.dirname(vid_path) or ".", exist_ok=True)
                from evaluation.video_recorder import VideoRecorder
                recorder = VideoRecorder(vid_path, env.sim.screen_w, env.sim.screen_h)
                print(f"[Bots] Recording video → {vid_path}")
            except Exception as e:
                print(f"[Bots] Video recording unavailable: {e}")
                recorder = None

        # ── Run the match ─────────────────────────────────────────────── #
        rs, bs, completed, still_running, stats = run_match(
            env, bots, trainer, args, match_idx,
            reasons_log=reasons_log, event_log=event_log, recorder=recorder,
        )

        # ── Report result ─────────────────────────────────────────────── #
        winner = "Red" if rs > bs else ("Blue" if bs > rs else "Tie")
        if completed:
            _print_match_summary(match_idx, rs, bs, stats)
        else:
            print(f"\n[Bots] Match {match_idx} stopped early — Red {rs} : {bs} Blue  [{winner}]")
            _print_match_summary(match_idx, rs, bs, stats)
        results.append((rs, bs))
        all_stats.append(stats)

        if recorder:
            recorder.close()
            print(f"[Bots] Video saved → {vid_path}")

        # Finalise event log with summary block
        try:
            event_log.write_summary(env.sim, match_idx, rs, bs)
        except Exception:
            pass
        event_log.close()
        print(f"[Bots] Events CSV closed → {events_path}")

        if reasons_log is not None:
            reasons_log.write(
                f"\n# ended {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"result=Red {rs}:Blue {bs}\n"
            )
            reasons_log.close()
            print(f"[Bots] Reasons log closed → {reasons_path}")

        # If user hit ESC, stop the whole session
        if not still_running:
            print("[Bots] ESC — stopping session early.")
            break

        # ── Reset for next match ──────────────────────────────────────── #
        if match_idx < args.matches:
            seed += 1
            env.sim.rng = None   # let reset pick up the new seed if needed
            obs  = env.reset()
            env._last_obs   = obs
            env._last_masks = env.get_action_masks()
            for bot in bots.values():
                bot.reset()
            print(f"[Bots] Field reset for match {match_idx + 1}.")

    # ── Session summary ───────────────────────────────────────────────── #
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"[Bots] Session summary ({len(results)} matches)")
        for i, (r, b) in enumerate(results, 1):
            print(f"  Match {i}: Red {r:3d} — {b:3d} Blue")
        avg_r = sum(r for r, _ in results) / len(results)
        avg_b = sum(b for _, b in results) / len(results)
        print(f"  Average  : Red {avg_r:.1f} — {avg_b:.1f} Blue")

        # Cumulative per-robot totals
        totals = {rid: {"scores": 0, "intakes": 0, "drops": 0,
                        "flips": 0, "toggles": 0} for rid in AGENT_IDS}
        for st in all_stats:
            for rid in AGENT_IDS:
                for k in totals[rid]:
                    totals[rid][k] += st.get(k, {}).get(rid, 0)
        print(f"\n  Cumulative per-robot stats:")
        print(f"  {'robot':<8}{'scores':>8}{'intakes':>10}{'drops':>8}{'flips':>8}{'toggles':>10}")
        for rid in AGENT_IDS:
            t = totals[rid]
            print(f"  {rid:<8}{t['scores']:>8}{t['intakes']:>10}"
                  f"{t['drops']:>8}{t['flips']:>8}{t['toggles']:>10}")
        print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
