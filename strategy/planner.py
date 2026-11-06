"""
strategy/planner.py
───────────────────
Beam search over macro-actions to find high-scoring routes.

Macro-actions
  pick_front  obj         drive forward until the front intake reaches it
  pick_rear   obj         reverse onto a STANDING pin (back intake)
  load        loader,what sit in an own Load Zone, take a Match Load
  place       goal,items  reverse onto a goal, place pin / cup / cup+pin /
                          pin+cup (orientation chosen for max points)
  toggle      tid         push the toggle to our colour (Skills: its
                          quadrant's colour)
  park        —           end the run in the Midfield

The planner scores with simulation/scoring.py, i.e. the same rules the
physics simulator uses.  Timing comes from the RobotProfile.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, FrozenSet

from config import game_rules as GR
from config.robot_profile import RobotProfile, DEFAULT_PROFILE
from simulation.scoring import can_place, QUADRANT_COLOR, TOGGLE_OF_QUADRANT
from strategy.world import World, move_to_reach, auton_side_ok, _pin_halves


HALF_VAL = {"red": 5, "blue": 5, "yellow": 10}


class S:
    """Search state (immutable by convention)."""
    __slots__ = ("t", "x", "y", "h", "pin", "cup", "taken", "pulled", "stacks",
                 "toggles", "loads", "parked", "used_goals", "parent", "act",
                 "own", "key_rank")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def child(self, **kw):
        c = S()
        for k in self.__slots__:
            setattr(c, k, kw[k] if k in kw else getattr(self, k))
        c.parent = self
        return c

    def actions(self) -> List[dict]:
        out = []
        s = self
        while s is not None and s.act is not None:
            out.append(s.act)
            s = s.parent
        return out[::-1]


@dataclass
class Task:
    """What to optimise."""
    world: World
    alliance: str = "red"
    robot_id: str = "red1"
    horizon: float = 15.0          # seconds available to this robot
    phase: str = "auton"           # auton | driver | skills
    match_time_left_at_start: float = 15.0   # for the Endgame centre-goal rule
    objective: str = "points"      # points | awp
    forbid_objs: FrozenSet[int] = frozenset()
    forbid_goals: FrozenSet[int] = frozenset()
    allow_park: bool = False
    endgame_seconds: float = GR.ENDGAME_SECONDS


class Planner:
    def __init__(self, task: Task, profile: RobotProfile = None,
                 beam: int = 300, k_nearest: int = 9, lam: float = 2.0):
        self.T = task
        self.W = task.world
        self.P = profile or DEFAULT_PROFILE
        self.beam = beam
        self.k = k_nearest
        self.lam = lam
        self.reach = self.P.length / 2.0 + 2.0        # centre → intake/score point

    # ── value ──────────────────────────────────────────────────────────────
    def own_score(self, stacks, toggles, parked=False) -> int:
        mid_owner = None
        if self.T.phase == "skills" and parked:
            mid_owner = "red"
        pts = self.W.score(stacks, toggles, mid_owner)
        v = pts["red"] if (self.W.skills or self.T.alliance == "red") else pts["blue"]
        if parked:
            v += GR.POINTS_MIDFIELD_PARK
        return v

    def held_potential(self, s: S) -> float:
        v = 0.0
        if s.pin is not None:
            v += 0.6 * max(HALF_VAL[s.pin[0]], HALF_VAL[s.pin[1]])
        if s.cup is not None:
            v += 2.0
        return v

    def rank(self, s: S) -> float:
        r = s.own + self.held_potential(s) - self.lam * s.t
        if self.T.objective == "awp":
            pins, goals2 = self.W.awp_status(s.stacks, s.toggles, self.T.alliance)
            r += 4.0 * min(pins, GR.AWP_MIN_PINS) + 8.0 * min(goals2, GR.AWP_MIN_GOALS_WITH_2)
        return r

    def final_value(self, s: S) -> float:
        if self.T.objective == "awp":
            pins, goals2 = self.W.awp_status(s.stacks, s.toggles, self.T.alliance)
            ok = pins >= GR.AWP_MIN_PINS and goals2 >= GR.AWP_MIN_GOALS_WITH_2
            return (1000 if ok else 0) + s.own
        return s.own

    # ── helpers ────────────────────────────────────────────────────────────
    def _touch_ok(self, x, y, on_line=False) -> bool:
        if self.T.phase != "auton" or on_line:
            return True
        return auton_side_ok(self.T.alliance, x, y)

    def _goal_ok(self, gi: int, t_arrive: float) -> bool:
        g = self.W.goals[gi]
        if g.gid in self.T.forbid_goals:
            return False
        if not self.W.skills and g.alliance not in ("neutral", self.T.alliance):
            return False                                  # SG9
        if not self._touch_ok(g.x, g.y):
            return False
        if (self.T.phase == "driver" and g.kind == "center" and
                self.T.match_time_left_at_start - t_arrive <= self.T.endgame_seconds):
            return False                                  # SG12
        return True

    def _height_ok(self, gi: int, stack, adding_pins: int) -> bool:
        g = self.W.goals[gi]
        n = sum(1 for it in stack if it[0] == "pin")
        # need to reach the current top to place on it
        top = GR.GOAL_HEIGHT[g.kind] + GR.PIN_STACK_HEIGHT * n
        return top <= self.P.max_place_height

    # ── successor generation ───────────────────────────────────────────────
    def expand(self, s: S) -> List[S]:
        P, W, T = self.P, self.W, self.T
        kids: List[S] = []
        if s.parked:
            return kids

        # ---- picks -------------------------------------------------------
        picks = []
        for o in W.objects:
            if o.oid in s.taken or o.oid in T.forbid_objs:
                continue
            if not self._touch_ok(o.x, o.y, o.on_line):
                continue
            if o.kind == "pin":
                if s.pin is None and P.front_intake_pins:
                    picks.append((o, "front", o.pin, None))
                if s.pin is None and o.vertical and P.rear_intake_vertical_pins:
                    picks.append((o, "rear", o.pin, None))
            else:
                has_nested = o.nested is not None and o.oid not in s.pulled
                if s.cup is None and P.front_intake_cups:
                    if has_nested:
                        if s.pin is None:
                            picks.append((o, "front", o.nested, o.clear_up))
                    else:
                        picks.append((o, "front", None, o.clear_up))
                if (has_nested and s.pin is None and o.vertical and
                        P.rear_intake_vertical_pins and P.rear_pulls_nested_pin):
                    picks.append((o, "rear_pull", o.nested, None))
        timed = []
        for o, side, pin, cup in picks:
            use = "front" if side == "front" else "rear"
            dt, nx, ny, nh = move_to_reach(s.x, s.y, s.h, o.x, o.y, self.reach, use, P)
            dt += P.intake_time
            timed.append((dt, o, side, pin, cup, nx, ny, nh))
        timed.sort(key=lambda z: z[0])
        for dt, o, side, pin, cup, nx, ny, nh in timed[: self.k]:
            t1 = s.t + dt
            if t1 > T.horizon:
                continue
            if side == "rear_pull":
                kid = s.child(t=t1, x=nx, y=ny, h=nh, pin=pin,
                              pulled=s.pulled | {o.oid})
                desc = f"back-intake the {o.nested_color} pin standing in cup @({o.x:.0f},{o.y:.0f})"
            else:
                new_pin = pin if pin is not None else s.pin
                new_cup = cup if (o.kind == "cup") else s.cup
                kid = s.child(t=t1, x=nx, y=ny, h=nh, pin=new_pin, cup=new_cup,
                              taken=s.taken | {o.oid})
                desc = ("back-intake " if side == "rear" else "front-intake ") + o.label
            kid.act = {"type": "pick_" + ("front" if side == "front" else "rear"),
                       "obj": o.oid, "x": o.x, "y": o.y, "pull_nested": side == "rear_pull",
                       "t_end": round(t1, 2), "desc": desc}
            kid.own = s.own
            kids.append(kid)

        # ---- match loads ---------------------------------------------------
        if T.phase in ("driver", "skills") and (s.pin is None or s.cup is None):
            cups, apins, ypins, bpins = s.loads
            options = []
            if s.pin is None and s.cup is None and cups > 0:
                if apins > 0: options.append(("cup+pin", True, True))
                if ypins > 0: options.append(("cup+yellow", True, True))
                if bpins > 0: options.append(("cup+blue", True, True))
            if s.pin is None:
                if apins > 0: options.append(("pin", True, False))
                if ypins > 0: options.append(("yellow", True, False))
                if bpins > 0: options.append(("blue", True, False))
            if s.cup is None and cups > 0:
                options.append(("cup", False, True))
            for (lx, ly) in W.loaders:
                dt, nx, ny, nh = move_to_reach(s.x, s.y, s.h, lx, ly, 0.0, "center", P)
                for what, gp, gc in options:
                    t1 = s.t + dt + P.match_load_time
                    if t1 > T.horizon:
                        continue
                    parts = what.split("+")
                    c, a, yv, b = s.loads
                    pin = s.pin
                    cupv = s.cup
                    for part in parts:
                        if part == "cup":
                            c -= 1; cupv = True
                        elif part == "pin":
                            a -= 1
                            pin = _pin_halves("red_yellow" if (T.alliance == "red") else "blue_yellow")
                        elif part == "yellow":
                            yv -= 1; pin = _pin_halves("yellow_yellow")
                        elif part == "blue":
                            b -= 1; pin = _pin_halves("blue_yellow")
                    kid = s.child(t=t1, x=nx, y=ny, h=nh, pin=pin, cup=cupv, loads=(c, a, yv, b))
                    kid.act = {"type": "load", "loader": [lx, ly], "what": what,
                               "t_end": round(t1, 2),
                               "desc": f"match load {what} at loader ({lx:.0f},{ly:.0f})"}
                    kid.own = s.own
                    kids.append(kid)

        # ---- placements ----------------------------------------------------
        if s.pin is not None or s.cup is not None:
            for gi, g in enumerate(W.goals):
                stack = s.stacks[gi]
                dt, nx, ny, nh = move_to_reach(s.x, s.y, s.h, g.x, g.y,
                                               self.reach - 1.0, "rear", P)
                t_arr = s.t + dt
                if t_arr > T.horizon or not self._goal_ok(gi, t_arr):
                    continue
                if not self._height_ok(gi, stack, 1):
                    continue
                travel_t = dt
                for items in self._place_options(s, stack):
                    best = self._best_orientation(s, gi, stack, items, travel_t)
                    if best is None:
                        continue
                    new_stack, own, flip_cost, flips = best
                    st_time = {("pin",): P.score_pin_time, ("cup",): P.score_cup_time}.get(
                        tuple(items), P.score_pair_time)
                    t1 = t_arr + flip_cost + st_time
                    if t1 > T.horizon:
                        continue
                    stacks = s.stacks[:gi] + (new_stack,) + s.stacks[gi + 1:]
                    kid = s.child(t=t1, x=nx, y=ny, h=nh,
                                  pin=None if "pin" in items else s.pin,
                                  cup=None if "cup" in items else s.cup,
                                  stacks=stacks, used_goals=s.used_goals | {g.gid})
                    kid.own = own
                    kid.act = {"type": "place", "goal": g.gid, "items": list(items),
                               "flip_pin": flips[0], "flip_cup": flips[1],
                               "t_end": round(t1, 2), "score_after": own,
                               "desc": f"reverse onto {g.label} and place {' then '.join(items)}"
                                       + (" (flip pin)" if flips[0] else "")
                                       + (" (flip cup)" if flips[1] else "")}
                    kids.append(kid)

        # ---- toggles -------------------------------------------------------
        for ti, tg in enumerate(W.toggles):
            target = tg.quadrant_color if W.skills else T.alliance
            if s.toggles[ti] == target:
                continue
            if not self._touch_ok(tg.x, tg.y):
                continue
            px, py = tg.x + tg.nx * 10.5, tg.y + tg.ny * 10.5
            dt, nx, ny, nh = move_to_reach(s.x, s.y, s.h, px, py, 0.0, "center", P)
            t1 = s.t + dt + P.toggle_time
            if t1 > T.horizon:
                continue
            toggles = s.toggles[:ti] + (target,) + s.toggles[ti + 1:]
            kid = s.child(t=t1, x=nx, y=ny, h=nh, toggles=toggles)
            kid.own = self.own_score(s.stacks, toggles)
            kid.act = {"type": "toggle", "toggle": tg.tid, "colour": target,
                       "t_end": round(t1, 2), "score_after": kid.own,
                       "desc": f"set toggle {tg.tid} ({['left','right','top','bottom'][tg.tid-1]} wall) to {target}"}
            kids.append(kid)

        # ---- park ----------------------------------------------------------
        if T.allow_park:
            dx, dy = 72 - s.x, 72 - s.y
            d = math.hypot(dx, dy) or 1.0
            l1 = (abs(dx) + abs(dy)) / d
            reach = (GR.MIDFIELD_HALF + 3.0) / l1
            dt, nx, ny, nh = move_to_reach(s.x, s.y, s.h, 72, 72, reach, "center", P)
            t1 = s.t + dt
            if t1 <= T.horizon:
                kid = s.child(t=t1, x=nx, y=ny, h=nh, parked=True)
                kid.own = self.own_score(s.stacks, s.toggles, parked=True)
                kid.act = {"type": "park", "x": round(nx, 1), "y": round(ny, 1),
                           "t_end": round(t1, 2), "score_after": kid.own,
                           "desc": "drive into the Midfield and stay there"}
                kids.append(kid)
        return kids

    def _place_options(self, s: S, stack):
        opts = []
        if s.pin is not None and can_place(stack, "pin"):
            opts.append(("pin",))
            if s.cup is not None:
                opts.append(("pin", "cup"))
        if s.cup is not None and can_place(stack, "cup"):
            opts.append(("cup",))
            if s.pin is not None:
                opts.append(("cup", "pin"))
        return opts

    def _best_orientation(self, s: S, gi: int, stack, items, travel_t: float):
        P = self.P
        pin_opts = [(s.pin, False)]
        if "pin" in items and P.can_flip_pin and s.pin[0] != s.pin[1]:
            pin_opts.append(((s.pin[1], s.pin[0]), True))
        cup_opts = [(s.cup, False)]
        if "cup" in items and P.can_flip_cup:
            cup_opts.append(((not s.cup), True))
        best = None
        for pin, fp in (pin_opts if "pin" in items else [(None, False)]):
            for cup, fc in (cup_opts if "cup" in items else [(None, False)]):
                new = list(stack)
                for it in items:
                    if it == "pin":
                        new.append(("pin", pin[0], pin[1]))
                    else:
                        new.append(("cup", cup))
                new = tuple(new)
                stacks = s.stacks[:gi] + (new,) + s.stacks[gi + 1:]
                own = self.own_score(stacks, s.toggles)
                n_flips = int(fp) + int(fc)
                flip_cost = max(0.0, n_flips * P.flip_time - travel_t)
                # tie-break: keep the most valuable half on top (visible
                # until something covers it) and prefer no flips
                top_bonus = 0.0
                if new and new[-1][0] == "pin":
                    top_bonus = 0.01 * HALF_VAL[new[-1][2]]
                elif new and new[-1][0] == "cup":
                    top_bonus = 0.005 * (1 if new[-1][1] else 0)
                key = own - 0.001 * n_flips + top_bonus - 2.0 * flip_cost
                if best is None or key > best[0]:
                    best = (key, new, own, flip_cost, (fp, fc))
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    # ── search ─────────────────────────────────────────────────────────────
    def initial_state(self, x, y, h, pin=None, cup=None, stacks=None, toggles=None,
                      loads=None, taken=frozenset(), pulled=frozenset()) -> S:
        W = self.W
        s = S(t=0.0, x=x, y=y, h=h, pin=pin, cup=cup, taken=frozenset(taken),
              pulled=frozenset(pulled),
              stacks=stacks if stacks is not None else W.initial_stacks(),
              toggles=toggles if toggles is not None else ("yellow",) * 4,
              loads=loads if loads is not None else W.initial_loads(),
              parked=False, used_goals=frozenset(), parent=None, act=None)
        s.own = self.own_score(s.stacks, s.toggles)
        return s

    def search(self, s0: S, max_depth: int = 80) -> S:
        beam = [s0]
        best = s0
        best_v = self.final_value(s0)
        for _ in range(max_depth):
            kids = []
            for s in beam:
                kids.extend(self.expand(s))
            if not kids:
                break
            for k in kids:
                v = self.final_value(k)
                if v > best_v or (v == best_v and k.t < best.t):
                    best, best_v = k, v
            seen = {}
            for k in kids:
                key = (k.taken, k.pulled, k.stacks, k.toggles, k.pin, k.cup, k.loads,
                       k.parked, int(k.x // 8), int(k.y // 8))
                r = self.rank(k)
                if key not in seen or r > seen[key][0]:
                    seen[key] = (r, k)
            ranked = sorted(seen.values(), key=lambda z: -z[0])
            beam = [k for _, k in ranked[: self.beam]]
        return best


def plan_robot(task: Task, start, profile: RobotProfile = None, beam: int = 300,
               lams=(0.5, 1.5, 3.0, 5.0), k_nearest: int = 9, **init_kw) -> S:
    """Run the beam search for several time-pressure weights; keep the best."""
    best = None
    for lam in lams:
        pl = Planner(task, profile, beam=beam, k_nearest=k_nearest, lam=lam)
        s0 = pl.initial_state(*start, **init_kw)
        b = pl.search(s0)
        if best is None or pl.final_value(b) > best[0] or (
                pl.final_value(b) == best[0] and b.t < best[1].t):
            best = (pl.final_value(b), b)
    return best[1]
