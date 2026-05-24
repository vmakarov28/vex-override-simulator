"""
agents/bot_event_log.py
=======================
Structured event logger for HeuristicBot debugging.

Why this exists
---------------
The free-form per-step `last_reason` log is hard to grep through when
debugging "why didn't the bot score?".  This module writes a structured
CSV alongside it with discrete events:

    SCORE       — pin/cup placed in a goal
    INTAKE      — pin/cup picked up from field
    DROP        — pin/cup released (intentional or knocked free)
    FLIP_PIN    — pin orientation toggled
    FLIP_CUP    — cup orientation toggled
    TOGGLE      — toggle owner changed by a robot
    STUCK       — bot detected itself sitting still and triggered escape
    PHASE       — autonomous → driver → endgame transitions

Events are detected by diffing simulator state between control steps,
so they are independent of whether env_wrapper's cooldown gates
accepted the bot's request.  This is the key debugging insight: if a
bot fires `flip_pin=True` but env_wrapper rejects it (cooldown), no
event is emitted — making the silent rejection visible by its absence.

Format
------
CSV with header:
    t_sim, step, robot, event, detail, x, y, score_red, score_blue

The `detail` column carries event-specific context (goal_id, color,
pin_id, etc.).  `x, y` are the robot's position when the event fired.
Score columns are for quick "did this score actually count?" checks.

Usage
-----
    logger = BotEventLog("artifacts/logs/events_m1.csv")
    # ... during run loop:
    logger.snapshot_pre(sim)         # before sim.step()
    # ... sim.step(...) happens ...
    logger.diff_and_emit(sim, t, step)
    logger.close()
"""

import csv
import os
from typing import Dict, List, Optional, Set, Tuple


class BotEventLog:
    """Diff-based structured event logger.

    The logger snapshots simulator state before each step and diffs it
    against the new state afterwards to detect scoring, intake, drop,
    flip, and toggle events.  This is more reliable than asking each
    bot whether its action succeeded, because env_wrapper's cooldown
    gates can silently reject actions.
    """

    HEADER = ["t_sim", "step", "robot", "event",
              "detail", "x", "y", "score_red", "score_blue"]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="", buffering=1, encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)
        self._closed = False

        # Snapshot caches (set by snapshot_pre)
        self._pre_carrying_pin: Dict[str, Optional[int]] = {}
        self._pre_carrying_cup: Dict[str, Optional[int]] = {}
        self._pre_goal_stacks:  Dict[int, List[Tuple[int, bool]]] = {}
        self._pre_toggle_owner: Dict[int, str] = {}
        self._pre_pin_flipped:  Dict[int, bool] = {}
        self._pre_cup_flipped:  Dict[int, bool] = {}
        self._pre_endgame: bool = False
        self._pre_phase:   str  = "init"

        # Cumulative tallies for end-of-match summary
        self.scores_per_robot:   Dict[str, int] = {}
        self.intakes_per_robot:  Dict[str, int] = {}
        self.drops_per_robot:    Dict[str, int] = {}
        self.flips_per_robot:    Dict[str, int] = {}
        self.toggles_per_robot:  Dict[str, int] = {}

    # ────────────────────────────────────────────────────────────────
    def snapshot_pre(self, sim):
        """Capture the world state before sim.step()."""
        self._pre_carrying_pin = {r.robot_id: id(r.carrying_pin) if r.carrying_pin else None
                                  for r in sim.robots}
        self._pre_carrying_cup = {r.robot_id: id(r.carrying_cup) if r.carrying_cup else None
                                  for r in sim.robots}
        self._pre_goal_stacks  = {g.goal_id: list(g.stack) for g in sim.goals}
        self._pre_toggle_owner = {t.toggle_id: t.owner for t in sim.toggles}
        self._pre_pin_flipped  = {id(p): bool(getattr(p, 'flipped', False))
                                  for p in sim.pins}
        self._pre_cup_flipped  = {id(c): bool(getattr(c, 'flipped', False))
                                  for c in sim.cups}
        self._pre_endgame = bool(getattr(sim.rules_engine, 'endgame_active', False))
        self._pre_phase   = str(getattr(sim, 'match_phase', 'init'))

    # ────────────────────────────────────────────────────────────────
    def diff_and_emit(self, sim, t_sim: float, step: int,
                      bots: Dict[str, object] = None):
        """Detect events by diffing snapshot against current state."""
        if self._closed:
            return

        red_s  = int(sim.rules_engine.red_score)
        blue_s = int(sim.rules_engine.blue_score)

        # ── Robot-level events ──────────────────────────────────
        for r in sim.robots:
            rid = r.robot_id
            rx, ry = float(r.body.position.x), float(r.body.position.y)

            pre_pin = self._pre_carrying_pin.get(rid)
            pre_cup = self._pre_carrying_cup.get(rid)
            cur_pin = id(r.carrying_pin) if r.carrying_pin else None
            cur_cup = id(r.carrying_cup) if r.carrying_cup else None

            # INTAKE: was None, now holding something
            if pre_pin is None and cur_pin is not None:
                color = getattr(r.carrying_pin, 'color', '?')
                pid   = getattr(r.carrying_pin, 'pin_id', -1)
                self._emit(t_sim, step, rid, "INTAKE",
                           f"pin id={pid} color={color}", rx, ry, red_s, blue_s)
                self.intakes_per_robot[rid] = self.intakes_per_robot.get(rid, 0) + 1
            if pre_cup is None and cur_cup is not None:
                cid = id(r.carrying_cup)
                self._emit(t_sim, step, rid, "INTAKE",
                           f"cup id={cid}", rx, ry, red_s, blue_s)
                self.intakes_per_robot[rid] = self.intakes_per_robot.get(rid, 0) + 1

            # DROP: was holding, now None, AND object did NOT get scored
            # (scoring will be detected separately by goal-stack diff)
            if pre_pin is not None and cur_pin is None:
                if not self._pin_id_now_in_some_stack(sim, pre_pin):
                    self._emit(t_sim, step, rid, "DROP",
                               f"pin id={pre_pin}", rx, ry, red_s, blue_s)
                    self.drops_per_robot[rid] = self.drops_per_robot.get(rid, 0) + 1
            if pre_cup is not None and cur_cup is None:
                if not self._cup_id_now_in_some_stack(sim, pre_cup):
                    self._emit(t_sim, step, rid, "DROP",
                               f"cup id={pre_cup}", rx, ry, red_s, blue_s)
                    self.drops_per_robot[rid] = self.drops_per_robot.get(rid, 0) + 1

        # ── Goal-stack events (SCORE) ─────────────────────────────
        for g in sim.goals:
            pre_n = len(self._pre_goal_stacks.get(g.goal_id, []))
            cur_n = len(g.stack)
            if cur_n > pre_n:
                # Last (cur_n - pre_n) entries are newly added
                for new_idx in range(pre_n, cur_n):
                    obj, is_pin = g.stack[new_idx]
                    obj_kind = "pin" if is_pin else "cup"
                    obj_color = getattr(obj, 'color', '') if is_pin else ''
                    obj_id    = id(obj)
                    scorer = self._find_scorer_for(obj, sim)
                    detail = f"{obj_kind} id={obj_id} goal={g.goal_id}"
                    if obj_color:
                        detail += f" color={obj_color}"
                    flipped = bool(getattr(obj, 'flipped', False))
                    detail += f" flipped={flipped}"
                    sx = float(scorer.body.position.x) if scorer else 0.0
                    sy = float(scorer.body.position.y) if scorer else 0.0
                    sid = scorer.robot_id if scorer else "?"
                    self._emit(t_sim, step, sid, "SCORE",
                               detail, sx, sy, red_s, blue_s)
                    if scorer:
                        self.scores_per_robot[sid] = self.scores_per_robot.get(sid, 0) + 1

        # ── Toggle events ──────────────────────────────────────
        for t in sim.toggles:
            pre_owner = self._pre_toggle_owner.get(t.toggle_id, t.owner)
            if pre_owner != t.owner:
                # Whoever was closest got credit
                actor = self._nearest_robot(sim, t.x, t.y)
                aid = actor.robot_id if actor else "?"
                ax = float(actor.body.position.x) if actor else 0.0
                ay = float(actor.body.position.y) if actor else 0.0
                self._emit(t_sim, step, aid, "TOGGLE",
                           f"id={t.toggle_id} {pre_owner}→{t.owner}",
                           ax, ay, red_s, blue_s)
                if actor:
                    self.toggles_per_robot[aid] = self.toggles_per_robot.get(aid, 0) + 1

        # ── Flip events (pin or cup orientation changed while carried) ──
        for p in sim.pins:
            pre_f = self._pre_pin_flipped.get(id(p))
            if pre_f is None:
                continue
            cur_f = bool(getattr(p, 'flipped', False))
            if pre_f != cur_f and getattr(p, 'carried_by', None):
                rid = p.carried_by
                rob = next((r for r in sim.robots if r.robot_id == rid), None)
                rx = float(rob.body.position.x) if rob else 0.0
                ry = float(rob.body.position.y) if rob else 0.0
                self._emit(t_sim, step, rid, "FLIP_PIN",
                           f"pin id={p.pin_id} flipped→{cur_f}",
                           rx, ry, red_s, blue_s)
                self.flips_per_robot[rid] = self.flips_per_robot.get(rid, 0) + 1
        for c in sim.cups:
            pre_f = self._pre_cup_flipped.get(id(c))
            if pre_f is None:
                continue
            cur_f = bool(getattr(c, 'flipped', False))
            if pre_f != cur_f and getattr(c, 'carried_by', None):
                rid = c.carried_by
                rob = next((r for r in sim.robots if r.robot_id == rid), None)
                rx = float(rob.body.position.x) if rob else 0.0
                ry = float(rob.body.position.y) if rob else 0.0
                self._emit(t_sim, step, rid, "FLIP_CUP",
                           f"cup id={id(c)} flipped→{cur_f}",
                           rx, ry, red_s, blue_s)
                self.flips_per_robot[rid] = self.flips_per_robot.get(rid, 0) + 1

        # ── Phase transitions ────────────────────────────────────
        cur_phase = str(getattr(sim, 'match_phase', 'init'))
        if cur_phase != self._pre_phase:
            self._emit(t_sim, step, "*", "PHASE",
                       f"{self._pre_phase}→{cur_phase}", 0.0, 0.0, red_s, blue_s)
        cur_endgame = bool(getattr(sim.rules_engine, 'endgame_active', False))
        if cur_endgame and not self._pre_endgame:
            self._emit(t_sim, step, "*", "PHASE",
                       "endgame_started", 0.0, 0.0, red_s, blue_s)

    # ────────────────────────────────────────────────────────────────
    def log_event(self, t_sim: float, step: int, robot: str,
                  event: str, detail: str, x: float = 0.0, y: float = 0.0,
                  red_s: int = 0, blue_s: int = 0):
        """Public hook for bots/runners to emit custom events."""
        if self._closed:
            return
        self._emit(t_sim, step, robot, event, detail, x, y, red_s, blue_s)

    def _emit(self, t_sim: float, step: int, robot: str,
              event: str, detail: str, x: float, y: float,
              red_s: int, blue_s: int):
        self._w.writerow([f"{t_sim:.3f}", step, robot, event,
                          detail, f"{x:.1f}", f"{y:.1f}", red_s, blue_s])

    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _pin_id_now_in_some_stack(sim, pin_obj_id: int) -> bool:
        for g in sim.goals:
            for obj, is_pin in g.stack:
                if is_pin and id(obj) == pin_obj_id:
                    return True
        return False

    @staticmethod
    def _cup_id_now_in_some_stack(sim, cup_obj_id: int) -> bool:
        for g in sim.goals:
            for obj, is_pin in g.stack:
                if (not is_pin) and id(obj) == cup_obj_id:
                    return True
        return False

    @staticmethod
    def _find_scorer_for(obj, sim):
        """Best-effort: robot whose carrying-slot was this obj last step.
        We can't see the pre-snapshot carrying_pin/cup here directly, so
        the scorer is whichever robot is currently closest to the goal
        the object was just placed in.  (Imperfect but informative.)
        """
        gid = getattr(obj, 'goal_id', None)
        if gid is None:
            return None
        goal = next((g for g in sim.goals if g.goal_id == gid), None)
        if goal is None:
            return None
        best = None
        best_d = float('inf')
        for r in sim.robots:
            d = ((float(r.body.position.x) - goal.x) ** 2 +
                 (float(r.body.position.y) - goal.y) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best = r
        return best

    @staticmethod
    def _nearest_robot(sim, x: float, y: float):
        best = None
        best_d = float('inf')
        for r in sim.robots:
            d = ((float(r.body.position.x) - x) ** 2 +
                 (float(r.body.position.y) - y) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best = r
        return best

    # ────────────────────────────────────────────────────────────────
    def write_summary(self, sim, match_idx: int, red_score: int, blue_score: int):
        """Append a human-readable summary block to the CSV's tail."""
        if self._closed:
            return
        self._f.write("\n# ─── MATCH SUMMARY ────────────────────────────\n")
        self._f.write(f"# match={match_idx}  Red {red_score} : {blue_score} Blue\n")
        self._f.write(f"# {'robot':<8}{'scores':>8}{'intakes':>10}"
                      f"{'drops':>8}{'flips':>8}{'toggles':>10}\n")
        for rid in ["red1", "red2", "blue1", "blue2"]:
            s  = self.scores_per_robot.get(rid, 0)
            i  = self.intakes_per_robot.get(rid, 0)
            d  = self.drops_per_robot.get(rid, 0)
            f  = self.flips_per_robot.get(rid, 0)
            tg = self.toggles_per_robot.get(rid, 0)
            self._f.write(f"# {rid:<8}{s:>8}{i:>10}{d:>8}{f:>8}{tg:>10}\n")
        self._f.write("# ──────────────────────────────────────────────\n")

    def close(self):
        if not self._closed:
            try:
                self._f.close()
            except Exception:
                pass
            self._closed = True
