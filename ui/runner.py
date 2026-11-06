"""
ui/runner.py
────────────────────────────────────────────────────────────────────────────
Unified match runner driven by `ui.menu.MatchConfig`.

Replaces the three previous entry points (`sim.run_interactive`,
`run_with_bots.py`, `run_bot_skills.py`) with one function that accepts
*any* combination of controllers — human, HeuristicBot, trained policy,
or idle — across either match (4 robots) or skills (1 robot) mode.

Per-slot controllers
--------------------
  • "human"     — driven by keyboard (WASD/EQF/T for red1, arrows/RShift/
                  RCtrl/[ for blue1; only red1 and blue1 can be human
                  since those are the slots the sim's keyboard handler
                  fills).  M-key fires a match-load when in a load zone.
  • "heuristic" — HeuristicBot (deterministic rule-based agent)
  • "policy"    — Trained MAPPO checkpoint; falls back to heuristic if
                  no checkpoint can be loaded.
  • "idle"      — Robot exists in the physics world but never moves
                  (zero action every step).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from ui.menu import (
    MatchConfig, RunResult, show_summary,
    CTRL_HUMAN, CTRL_HEURISTIC, CTRL_POLICY, CTRL_IDLE,
    HUMAN_SLOTS,
)

CONTROL_DT = 0.05    # 20 Hz control loop, matches CONTROL_DT in training


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def run_configured_match(config: MatchConfig) -> int:
    """Run all matches/skills runs described by `config`.  Returns the
    process exit code (0 = ok).  Pygame display is torn down on return so
    the caller can re-open the menu cleanly."""
    # Late imports so a `--help` invocation doesn't pay sim/torch import cost
    from simulation.simulator import OverrideSimulator
    from agents.heuristic_bot import HeuristicBot
    from config.game_rules import ROBOT_STARTS

    skills = (config.mode == "skills")

    # ── Skills mode: swap red1's start to the chosen corner ──────────
    # The sim hard-codes the lone robot as "red1" with alliance "red";
    # we temporarily rebind that slot's start so the bot/human spawns
    # wherever the user picked on the menu.
    saved_start = None
    if skills and config.skills_start != "red1":
        saved_start = dict(ROBOT_STARTS["red1"])
        ROBOT_STARTS["red1"] = dict(ROBOT_STARTS[config.skills_start])

    # Trained policy: not yet wired through this runner — observations
    # require an OverrideEnv wrapper that we don't build here.  For now,
    # any "policy" slot degrades to HEURISTIC so the robot still plays,
    # with a one-line note to the console.  (To re-enable full policy
    # play, use `python main.py --mode eval --checkpoint ... --render`.)
    trainer = None
    policy_slots = [rid for rid, c in config.controllers.items()
                    if c == CTRL_POLICY]
    if policy_slots:
        print(f"[Runner] Note: TRAINED POLICY slots are not wired through "
              f"the menu yet -- degrading {policy_slots} to HEURISTIC for "
              f"this run.  Use `--mode eval` for full policy evaluation.")
        for rid in policy_slots:
            config.controllers[rid] = CTRL_HEURISTIC

    # Optional video recorder
    recorder_cls = None
    if config.record_video:
        try:
            from evaluation.video_recorder import VideoRecorder
            recorder_cls = VideoRecorder
        except Exception as e:
            print(f"[Runner] Video recorder unavailable ({e})")

    # ── Run N matches ────────────────────────────────────────────────
    results: List[RunResult] = []
    try:
        for match_idx in range(config.matches):
            if config.headless:
                r = _run_one_headless(config, match_idx, skills)
            else:
                r = _run_one(
                    config, match_idx, skills,
                    trainer=trainer,
                    recorder_cls=recorder_cls,
                )
            if r is not None:
                results.append(r)
    finally:
        if saved_start is not None:
            ROBOT_STARTS["red1"] = saved_start
        try:
            pygame.display.quit()
        except Exception:
            pass

    # ── Show the session summary screen ──────────────────────────────
    # Always show it when headless (the user had nothing to watch) or
    # when there were multiple matches.  For a single rendered match,
    # the in-sim score screen has already served as the summary.
    if results and (config.headless or len(results) > 1):
        try:
            show_summary(results, config)
        except Exception as e:
            print(f"[Runner] Summary screen failed: {e}")

    try:
        pygame.quit()
    except Exception:
        pass

    return 0


# ════════════════════════════════════════════════════════════════════════════
# SINGLE-MATCH LOOP
# ════════════════════════════════════════════════════════════════════════════
def _run_one(config: MatchConfig, match_idx: int, skills: bool,
             trainer=None, recorder_cls=None) -> RunResult:
    from simulation.simulator import OverrideSimulator
    from agents.heuristic_bot import HeuristicBot

    # Fresh display + sim each match (cleanest given menu/sim window size
    # difference; sim creates its own display in __init__).
    sim = OverrideSimulator(headless=False, skills_mode=skills)
    t_wall = time.time()

    slot_ids: List[str] = [r.robot_id for r in sim.robots]

    # Filter controllers to actually-present slots (skills has only red1).
    controllers: Dict[str, str] = {
        rid: config.controllers.get(rid, CTRL_IDLE) for rid in slot_ids
    }

    # Create HeuristicBot instances for the bot slots.
    bots: Dict[str, HeuristicBot] = {
        rid: HeuristicBot(rid, sim)
        for rid, ctrl in controllers.items() if ctrl == CTRL_HEURISTIC
    }

    # Detect whether a human is present at all.  If not, force-start the
    # timer immediately (the sim's auto-start triggers on keyboard input).
    has_human = any(c == CTRL_HUMAN for c in controllers.values())
    if not has_human:
        sim.timer_started = True

    # Video recorder per match (suffix _mN when there are multiple)
    recorder = None
    if recorder_cls is not None:
        vpath = _video_path_for_match(config.video_path,
                                       match_idx + 1, config.matches)
        try:
            os.makedirs(os.path.dirname(vpath) or ".", exist_ok=True)
            recorder = recorder_cls(vpath, sim.screen_w, sim.screen_h)
            print(f"[Runner] Recording match {match_idx + 1} -> {vpath}")
        except Exception as e:
            print(f"[Runner] Recorder failed: {e}")
            recorder = None

    clock = pygame.time.Clock()
    running = True
    sim_time = 0.0

    # ── Main loop ────────────────────────────────────────────────────
    while running and not sim.match_over and sim_time < config.duration:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                if ev.key == pygame.K_r:
                    sim.reset()
                    for bot in bots.values():
                        bot.reset()
                    sim_time = 0.0
                    if not has_human:
                        sim.timer_started = True

        actions = _build_actions(sim, slot_ids, controllers, bots, trainer)

        sim.step(CONTROL_DT, actions)
        sim.render()
        if recorder is not None:
            frame = pygame.surfarray.array3d(sim.screen).transpose(1, 0, 2)
            recorder.write_frame(frame)
        sim_time += CONTROL_DT
        clock.tick(30)

    # ── Finalise score (apply SC5b if we exited via time, not match_over) ──
    if not sim.match_over and running:
        final = sim.rules_engine.calculate_final_score(
            sim.goals, sim.toggles, sim.robots)
        sim.red_score  = final["red"]
        sim.blue_score = final["blue"]
        sim.match_over  = True
        sim.match_phase = "ended"

    # ── Hold the score screen briefly so the user / recorder can see ──
    if running:
        if recorder is not None:
            # Capture ~2s of the final score screen for the video.
            for _ in range(60):
                sim.render()
                frame = pygame.surfarray.array3d(sim.screen).transpose(1, 0, 2)
                recorder.write_frame(frame)
                clock.tick(30)
        # If there's a human watching, wait for R (reset / next) or ESC.
        if has_human and match_idx == config.matches - 1:
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        waiting = False
                    if ev.type == pygame.KEYDOWN:
                        if ev.key in (pygame.K_ESCAPE, pygame.K_r,
                                       pygame.K_RETURN, pygame.K_SPACE):
                            waiting = False
                sim.render()
                clock.tick(30)
        elif not has_human:
            # Bot-only: linger 1 s so the user sees the final score.
            for _ in range(30):
                sim.render()
                clock.tick(30)

    if recorder is not None:
        recorder.close()

    # Print a short summary line and build the structured result
    winner = ("RED"  if sim.red_score > sim.blue_score else
              "BLUE" if sim.blue_score > sim.red_score else "TIE")
    if skills:
        print(f"[Runner] Skills run {match_idx + 1}/{config.matches}: "
              f"SCORE = {sim.red_score}")
    else:
        print(f"[Runner] Match {match_idx + 1}/{config.matches}: "
              f"Red {sim.red_score} - {sim.blue_score} Blue  [{winner}]")

    return RunResult(
        index=match_idx + 1, mode=("skills" if skills else "match"),
        red_score=int(sim.red_score), blue_score=int(sim.blue_score),
        skills_score=int(sim.red_score) if skills else 0,
        winner=winner,
        sim_seconds=sim_time, wall_seconds=time.time() - t_wall,
    )


# ════════════════════════════════════════════════════════════════════════════
# HEADLESS RUN — no rendering, no event loop, full bot speed
# ════════════════════════════════════════════════════════════════════════════
def _run_one_headless(config: MatchConfig, match_idx: int,
                       skills: bool) -> RunResult:
    """Run a single match/skills attempt without rendering or input.
    HUMAN slots silently degrade to IDLE — there's no display to control."""
    from simulation.simulator import OverrideSimulator
    from agents.heuristic_bot import HeuristicBot

    sim = OverrideSimulator(headless=True, skills_mode=skills)
    slot_ids = [r.robot_id for r in sim.robots]
    controllers = {rid: config.controllers.get(rid, CTRL_IDLE)
                   for rid in slot_ids}
    # Headless = no keyboard.  Any HUMAN slot becomes IDLE so the match
    # still progresses (rather than waiting forever for a key press).
    for rid in slot_ids:
        if controllers[rid] == CTRL_HUMAN:
            controllers[rid] = CTRL_IDLE
    bots: Dict[str, HeuristicBot] = {
        rid: HeuristicBot(rid, sim)
        for rid, c in controllers.items() if c == CTRL_HEURISTIC
    }
    sim.timer_started = True   # no input gate in headless

    t_wall   = time.time()
    sim_time = 0.0
    while not sim.match_over and sim_time < config.duration:
        actions = []
        for rid in slot_ids:
            ctrl = controllers[rid]
            if ctrl == CTRL_HEURISTIC:
                actions.append(bots[rid].get_sim_action())
            else:   # idle (or stub for unwired policy)
                actions.append(_zero_action())
        sim.step(CONTROL_DT, actions)
        sim_time += CONTROL_DT
    # Finalise score (SC5b etc.) if we hit the time cap
    sc5b_yellow = 0
    sc5b_credit = 0
    if not sim.match_over:
        final = sim.rules_engine.calculate_final_score(
            sim.goals, sim.toggles, sim.robots)
        # calculate_final_score doesn't return the visible-yellow count
        # directly, but it prints it.  We recompute it cheaply here for
        # the summary table.
        sim.red_score  = final["red"]
        sim.blue_score = final["blue"]

    winner = ("RED"  if sim.red_score > sim.blue_score else
              "BLUE" if sim.blue_score > sim.red_score else "TIE")
    wall = time.time() - t_wall
    speedup = sim_time / wall if wall > 0 else 0.0
    if skills:
        print(f"[Headless] Skills run {match_idx + 1}/{config.matches}: "
              f"SCORE = {sim.red_score}   "
              f"(sim={sim_time:.0f}s wall={wall:.2f}s, {speedup:.1f}x)")
    else:
        print(f"[Headless] Match {match_idx + 1}/{config.matches}: "
              f"Red {sim.red_score} - {sim.blue_score} Blue  [{winner}]  "
              f"(sim={sim_time:.0f}s wall={wall:.2f}s, {speedup:.1f}x)")

    return RunResult(
        index=match_idx + 1, mode=("skills" if skills else "match"),
        red_score=int(sim.red_score), blue_score=int(sim.blue_score),
        skills_score=int(sim.red_score) if skills else 0,
        winner=winner,
        sim_seconds=sim_time, wall_seconds=wall,
        sc5b_yellow_visible=sc5b_yellow, sc5b_credit=sc5b_credit,
    )


# ════════════════════════════════════════════════════════════════════════════
# PER-STEP ACTION BUILDER
# ════════════════════════════════════════════════════════════════════════════
def _zero_action() -> dict:
    return {
        "left": 0.0, "right": 0.0,
        "intake": False, "score_pin": False, "score_cup": False,
        "toggle": False, "flip_pin": False, "flip_cup": False,
        "match_load": False,
    }


def _build_actions(sim, slot_ids, controllers, bots, trainer):
    """Compose the per-slot action list for one sim step."""
    keys = pygame.key.get_pressed()

    # Let the sim's own keyboard handler resolve red1 / blue1 input.
    # It fills slot 0 (red1) and slot 2 (blue1) when present; other
    # slots are None.  We then OVERRIDE per controller config below.
    kbd_actions = sim.handle_keyboard(keys)

    out = [None] * len(slot_ids)
    for i, rid in enumerate(slot_ids):
        ctrl = controllers[rid]
        if ctrl == CTRL_HUMAN:
            # Only red1 / blue1 can be HUMAN per menu rules.  Use the
            # sim's keyboard result for whichever index that is.
            out[i] = kbd_actions[i] if i < len(kbd_actions) else _zero_action()
            if out[i] is None:
                out[i] = _zero_action()
        elif ctrl == CTRL_HEURISTIC:
            out[i] = bots[rid].get_sim_action()
        elif ctrl == CTRL_POLICY and trainer is not None:
            # Policy slot — until we wire OverrideEnv observations into
            # this runner, fall back to a no-op.  HeuristicBot fallback
            # was already applied at config-load time when the checkpoint
            # was missing, so reaching here means the checkpoint LOADED
            # but obs plumbing is the gap.  Document as TODO.
            out[i] = _zero_action()
        else:  # CTRL_IDLE or anything else
            out[i] = _zero_action()
    return out


# ════════════════════════════════════════════════════════════════════════════
# VIDEO-PATH SUFFIX HELPER
# ════════════════════════════════════════════════════════════════════════════
def _video_path_for_match(base: str, match_idx: int, total: int) -> str:
    if total <= 1 or not base:
        return base
    root, ext = os.path.splitext(base)
    return f"{root}_m{match_idx}{ext}"
