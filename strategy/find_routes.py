"""
strategy/find_routes.py
───────────────────────
Search for the best Autonomous and Robot Skills routes for the robot in
config/robot_profile.py, then verify each route in the full physics sim.

    venv\\Scripts\\python.exe -m strategy.find_routes              # everything
    venv\\Scripts\\python.exe -m strategy.find_routes --mode skills --beam 400
    venv\\Scripts\\python.exe -m strategy.find_routes --mode auton --video

Outputs go to artifacts/routes/:
    <name>.json   machine-readable route (waypoints + actions per robot)
    <name>.png    field picture with the driven path
    <name>.mp4    (with --video) physics replay
    ROUTES.md     human-readable summary of every route
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import game_rules as GR
from config.robot_profile import DEFAULT_PROFILE, RobotProfile
from strategy.world import World, _pin_halves
from strategy.planner import Task, Planner, plan_robot
from strategy import executor

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts", "routes")


def _start(rid):
    st = GR.ROBOT_STARTS[rid]
    return (st["pos"][0], st["pos"][1], st["angle"])


def _preload(alliance):
    return _pin_halves("red_yellow" if alliance == "red" else "blue_yellow")


# ═══════════════════════════════════════════════════════════════════════════
# AUTONOMOUS (two robots, one alliance)
# ═══════════════════════════════════════════════════════════════════════════
AUTON_MARGIN = 0.8      # s of slack so physics execution finishes in time
SKILLS_MARGIN = 2.0


def _footprint(W, actions, final_state, initial_stacks):
    """What one robot's plan changes on the field."""
    objs = frozenset(a["obj"] for a in actions if "obj" in a and not a.get("pull_nested"))
    pulled = frozenset(a["obj"] for a in actions if a.get("pull_nested"))
    goals = frozenset(a["goal"] for a in actions if a["type"] == "place")
    stacks = tuple(final_state.stacks[i] if g.gid in goals else initial_stacks[i]
                   for i, g in enumerate(W.goals))
    toggles = list(("yellow",) * 4)
    for a in actions:
        if a["type"] == "toggle":
            toggles[a["toggle"] - 1] = a["colour"]
    return {"objs": objs | pulled, "taken": objs, "pulled": pulled, "goals": goals,
            "stacks": stacks, "toggles": tuple(toggles)}


def _plan_one(W, alliance, rid, objective, profile, beam, lam, other=None,
              horizon=None):
    horizon = horizon or (GR.AUTONOMOUS_SECONDS - AUTON_MARGIN)
    init = W.initial_stacks()
    kw = dict(pin=_preload(alliance))
    forbid_objs = forbid_goals = frozenset()
    if other is not None:
        forbid_objs, forbid_goals = other["objs"], other["goals"]
        kw.update(stacks=other["stacks"], toggles=other["toggles"])
    task = Task(W, alliance, rid, horizon, "auton", objective=objective,
                forbid_objs=forbid_objs, forbid_goals=forbid_goals)
    pl = Planner(task, profile, beam=beam, lam=lam)
    st = pl.search(pl.initial_state(*_start(rid), **kw))
    return st, st.actions()


def plan_auton_alliance(alliance="red", objective="points", beam=250,
                        profile: RobotProfile = None, lams=(0.5, 1.5, 3.0, 5.0),
                        rounds=3):
    """Joint plan for both robots by alternating re-planning: plan robot A,
    plan B around A, re-plan A around B, ... (elements and goals one robot
    uses are off-limits to the other; toggles either robot sets count for
    both)."""
    W = World("match", alliance)
    init = W.initial_stacks()
    ids = [f"{alliance}1", f"{alliance}2"]
    candidates = []
    # Toggles on our side of the Autonomous Line.  Seeds: robot A plans as if
    # the partner already set some subset of them, so the partner (planned
    # next, without those toggles) picks them up — breaks the "whoever
    # planned first keeps every toggle" local optimum.
    own_toggles = [t.tid for t in W.toggles
                   if (t.y >= t.x) == (alliance == "red")]
    seeds = [()] + [(t,) for t in own_toggles] + [tuple(own_toggles)]
    runs = [(o, lam, seed) for o in (ids, ids[::-1]) for lam in lams for seed in seeds]
    for (first, second), lam, seed in runs:
        if True:
            assumed = None
            if seed:
                tg = ["yellow"] * 4
                for tid in seed:
                    tg[tid - 1] = alliance
                assumed = {"objs": frozenset(), "goals": frozenset(),
                           "stacks": init, "toggles": tuple(tg)}
            sA, aA = _plan_one(W, alliance, first, objective, profile, beam, lam, assumed)
            fA = _footprint(W, aA, sA, init)
            sB, aB = _plan_one(W, alliance, second, objective, profile, beam, lam, fA)
            fB = _footprint(W, aB, sB, init)
            for _ in range(rounds):
                for rid, rid_other in ((first, second), (second, first)):
                    other = fB if rid == first else fA
                    st, acts = _plan_one(W, alliance, rid, objective, profile, beam, lam, other)
                    fp = _footprint(W, acts, st, init)
                    if rid == first:
                        sA, aA, fA = st, acts, fp
                    else:
                        sB, aB, fB = st, acts, fp
                    # joint state value (both footprints merged)
                    stacks = tuple(fA["stacks"][i] if g.gid in fA["goals"] else fB["stacks"][i]
                                   for i, g in enumerate(W.goals))
                    toggles = tuple(fA["toggles"][i] if fA["toggles"][i] != "yellow" else fB["toggles"][i]
                                    for i in range(4))
                    pts = W.score(stacks, toggles)[alliance]
                    pins, goals2 = W.awp_status(stacks, toggles, alliance)
                    candidates.append({"plans": {first: aA, second: aB}, "predicted": pts,
                                       "awp_pins": pins, "awp_goals": goals2,
                                       "lam": lam, "order": [first, second]})
    seen, uniq = set(), []
    def ckey(c):
        awp = c["awp_pins"] >= GR.AWP_MIN_PINS and c["awp_goals"] >= GR.AWP_MIN_GOALS_WITH_2
        return (-(awp if objective == "awp" else 0), -c["predicted"])
    for c in sorted(candidates, key=ckey):
        sig = json.dumps(c["plans"], sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    return uniq


def best_auton(alliance="red", objective="points", beam=250, profile=None,
               verify_top=6, verbose=True):
    cands = plan_auton_alliance(alliance, objective, beam, profile)
    best = None
    for c in cands[:verify_top]:
        res = executor.run_auton(c["plans"], profiles=_profiles(profile))
        c["physics"] = res["red_auton"] if alliance == "red" else res["blue_auton"]
        c["physics_awp"] = res["awp"].get(alliance, {})
        c["logs"] = res["logs"]
        if verbose:
            print(f"   candidate order={c['order']} lam={c['lam']}: planned "
                  f"{c['predicted']}  physics {c['physics']}  "
                  f"AWP={c['physics_awp'].get('awp')}")
        key = (c["physics_awp"].get("awp", False) if objective == "awp" else 0,
               c["physics"], c["predicted"])
        if best is None or key > best[0]:
            best = (key, c)
    return best[1]


def _profiles(profile):
    if profile is None:
        return None
    return {rid: profile for rid in ("red1", "red2", "blue1", "blue2")}


# ═══════════════════════════════════════════════════════════════════════════
# SKILLS (one robot, 60 s)
# ═══════════════════════════════════════════════════════════════════════════
def best_skills(beam=300, profile=None, verify_top=6, verbose=True,
                lams=(1.0, 2.0, 3.0, 4.5)):
    W = World("skills", "red")
    cands = []
    for lam in lams:
        for k in (8, 12):
            pl = Planner(Task(W, "red", "red1", GR.SKILLS_SECONDS - SKILLS_MARGIN, "skills",
                              allow_park=True), profile, beam=beam, k_nearest=k, lam=lam)
            s = pl.search(pl.initial_state(*_start("red1"), pin=_preload("red")), max_depth=120)
            cands.append({"plan": s.actions(), "predicted": s.own, "lam": lam, "k": k,
                          "t": s.t})
    seen, uniq = set(), []
    for c in sorted(cands, key=lambda c: -c["predicted"]):
        sig = json.dumps(c["plan"], sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    best = None
    for c in uniq[:verify_top]:
        res = executor.run_skills(c["plan"], profile=profile)
        c["physics"] = res["score"]
        c["logs"] = res["logs"]
        if verbose:
            n_fail = sum(1 for l in res["logs"]["red1"] if l["status"] != "ok")
            print(f"   candidate lam={c['lam']} k={c['k']}: planned {c['predicted']}"
                  f"  physics {c['physics']}  (failed steps: {n_fail})")
        key = (c["physics"], c["predicted"])
        if best is None or key > best[0]:
            best = (key, c)
    return best[1]


# ═══════════════════════════════════════════════════════════════════════════
# output
# ═══════════════════════════════════════════════════════════════════════════
def _fmt_actions(acts):
    lines = []
    for i, a in enumerate(acts, 1):
        sa = f"  → {a['score_after']} pts" if "score_after" in a else ""
        lines.append(f"{i:>2}. t≈{a['t_end']:>5.2f}s  {a['desc']}{sa}")
    return "\n".join(lines)


def save_auton(name, c, alliance, video=False, profile=None):
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, name + ".png")
    mp4 = os.path.join(OUT, name + ".mp4") if video else None
    res = executor.run_auton(c["plans"], profiles=_profiles(profile),
                             render_path=png, video_path=mp4)
    data = {"alliance": alliance, "planned_points": c["predicted"],
            "physics_points": res["red_auton" if alliance == "red" else "blue_auton"],
            "physics_awp": res["awp"].get(alliance), "plans": c["plans"],
            "action_log": res["logs"], "final_stacks": res["stacks"],
            "final_toggles": res["toggles"]}
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(data, f, indent=1)
    return data


def save_skills(name, c, video=False, profile=None):
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, name + ".png")
    mp4 = os.path.join(OUT, name + ".mp4") if video else None
    res = executor.run_skills(c["plan"], profile=profile, render_path=png, video_path=mp4)
    data = {"planned_points": c["predicted"], "physics_points": res["score"],
            "plan": c["plan"], "action_log": res["logs"],
            "final_stacks": res["stacks"], "final_toggles": res["toggles"]}
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(data, f, indent=1)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auton", "skills", "all"], default="all")
    ap.add_argument("--beam", type=int, default=250)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    prof = DEFAULT_PROFILE
    md = ["# Override routes — robot profile `%s`\n" % prof.name,
          "Generated by `python -m strategy.find_routes`.  Planned = planner's "
          "prediction; Physics = score the pymunk simulator actually awarded "
          "when the route was driven.\n",
          "Robot: front intake (any element), back intake (standing pins only), "
          "scores only out of the back.  Edit `config/robot_profile.py` to "
          "match your real speeds/timings and re-run.  In auton routes the running '→ pts' total includes the partner robot's placements (robots are planned around each other's full route).\n"]
    t0 = time.time()
    if args.mode in ("auton", "all"):
        for alliance in ("red", "blue"):
            for objective in ("points", "awp"):
                print(f"[routes] auton {alliance} / {objective} ...")
                c = best_auton(alliance, objective, args.beam, prof)
                name = f"auton_{alliance}_{objective}"
                d = save_auton(name, c, alliance, args.video, prof)
                md.append(f"\n## Autonomous — {alliance} alliance, objective: {objective}\n")
                awp = d["physics_awp"] or {}
                md.append(f"Planned **{d['planned_points']}** pts, physics **{d['physics_points']}** pts. "
                          f"AWP in physics: **{awp.get('awp')}** "
                          f"({awp.get('pins')} scored pins, {awp.get('goals_with_2')} goals with 2+, "
                          f"off perimeter: {awp.get('no_perimeter_contact')}).\n")
                md.append(f"![{name}]({name}.png)\n")
                for rid, acts in d["plans"].items():
                    md.append(f"\n**{rid}** (start {GR.ROBOT_STARTS[rid]['pos']}):\n```\n{_fmt_actions(acts)}\n```\n")
    if args.mode in ("skills", "all"):
        print("[routes] skills ...")
        c = best_skills(args.beam, prof)
        d = save_skills("skills", c, args.video, prof)
        md.append("\n## Robot Skills (60 s, driver or programming)\n")
        md.append(f"Planned **{d['planned_points']}** pts, physics **{d['physics_points']}** pts.\n")
        md.append("![skills](skills.png)\n")
        md.append(f"```\n{_fmt_actions(d['plan'])}\n```\n")
    with open(os.path.join(OUT, "ROUTES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[routes] done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
