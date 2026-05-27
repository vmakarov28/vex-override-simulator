"""
bench.py — fast headless benchmark for heuristic bot iteration.

Runs N matches of 4-bot self-play, prints per-match scores and aggregate
stats.  No video, no event log, no pygame window — just sim + bots as fast
as possible.

Usage:
    python bench.py                       # 20 matches, seeds 1000..1019
    python bench.py --matches 30          # 30 matches
    python bench.py --seed 5000           # base seed (matches use seed+i)
    python bench.py --duration 90         # 90 sim-seconds per match
"""
import argparse
import json
import os
import sys
import time
import numpy as np

# Silence pygame banner
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from training.env_wrapper import OverrideEnv, AGENT_IDS
from agents.heuristic_bot import HeuristicBot

CONTROL_DT = 0.05    # 20 Hz, matches run_with_bots.py


def run_one(env, bots, duration: float):
    obs = env.reset()
    # Re-initialise bot internal state across matches
    for b in bots.values():
        b.reset()

    sim_time = 0.0
    done     = False
    info     = {}
    rs = bs = 0
    while sim_time < duration and not done:
        actions = {}
        for rid, bot in bots.items():
            cont, disc = bot.get_policy_action()
            actions[rid] = (cont, disc)
        _, _, done, info = env.step(actions)
        sim_time += CONTROL_DT

    if not env.sim.match_over:
        final = env.sim.rules_engine.calculate_final_score(
            env.sim.goals, env.sim.toggles, env.sim.robots)
        rs = final["red"]
        bs = final["blue"]
    else:
        rs = info.get("red_score",  env.sim.red_score)
        bs = info.get("blue_score", env.sim.blue_score)
    return int(rs), int(bs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches",  type=int,   default=20)
    p.add_argument("--seed",     type=int,   default=1000)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--quiet",    action="store_true")
    p.add_argument("--out",      type=str, default=None,
                   help="Optional: write per-seed results JSON to this path.")
    p.add_argument("--compare",  type=str, default=None,
                   help="Optional: compare against per-seed JSON from a prior run.")
    args = p.parse_args()

    env  = OverrideEnv(headless=True, seed=args.seed)
    bots = {rid: HeuristicBot(rid, env.sim) for rid in AGENT_IDS}

    per_seed = {}   # seed -> {"red": rs, "blue": bs, "total": rs+bs}
    reds, blues, totals = [], [], []
    t_start = time.time()
    for i in range(args.matches):
        seed = args.seed + i
        # Re-seed sim RNG per match so element jitter varies
        env.rng = np.random.default_rng(seed)
        rs, bs = run_one(env, bots, args.duration)
        per_seed[seed] = {"red": rs, "blue": bs, "total": rs + bs}
        reds.append(rs)
        blues.append(bs)
        totals.append(rs + bs)
        if not args.quiet:
            print(f"  match {i+1:3d}/{args.matches}  seed={seed}  "
                  f"red={rs:4d}  blue={bs:4d}  total={rs+bs:4d}")
    elapsed = time.time() - t_start

    reds   = np.array(reds)
    blues  = np.array(blues)
    totals = np.array(totals)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({str(k): v for k, v in per_seed.items()}, f, indent=2)
        print(f"  saved per-seed JSON -> {args.out}")

    if args.compare:
        try:
            with open(args.compare) as f:
                base = json.load(f)
            shared = sorted(set(int(k) for k in base) & set(per_seed.keys()))
            if shared:
                deltas = np.array([per_seed[s]["total"] - base[str(s)]["total"]
                                   for s in shared])
                wins   = int((deltas > 0).sum())
                losses = int((deltas < 0).sum())
                ties   = int((deltas == 0).sum())
                print()
                print(f"  vs {args.compare} ({len(shared)} shared seeds):")
                print(f"    mean d_total : {deltas.mean():+6.2f}  "
                      f"std={deltas.std():.2f}")
                print(f"    seed wins   : {wins}  losses: {losses}  ties: {ties}")
                # Simple paired t-stat (heuristic significance)
                if deltas.std() > 1e-6:
                    t = deltas.mean() / (deltas.std() / np.sqrt(len(deltas)))
                    print(f"    paired t    : {t:+.2f}  "
                          f"(|t|>=2 ~ p<0.05 for n={len(deltas)})")
        except Exception as e:
            print(f"  compare failed: {e}")
    print()
    print(f"  matches : {args.matches}")
    print(f"  duration: {args.duration:.0f}s sim each, {elapsed:.1f}s wall")
    print(f"  red     : mean={reds.mean():6.1f}  std={reds.std():5.1f}  "
          f"min={reds.min()}  max={reds.max()}")
    print(f"  blue    : mean={blues.mean():6.1f}  std={blues.std():5.1f}  "
          f"min={blues.min()}  max={blues.max()}")
    print(f"  total   : mean={totals.mean():6.1f}  std={totals.std():5.1f}  "
          f"min={totals.min()}  max={totals.max()}")
    return totals.mean()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
