"""
agents/heuristic_bot.py
=======================
Rule-based "perfect-play" bot for VEX Override.  v9.3.1

Design philosophy
-----------------
The bot reads the live simulator state every control step and picks the
single best action via a strict-priority decision tree.  No learning, no
randomness — given the same state it makes the same decision.

The output of `get_sim_action()` matches the format env_wrapper feeds to
Simulator.step():
    {"left": float[-1,1], "right": float[-1,1],
     "intake": bool, "score_pin": bool, "score_cup": bool,
     "toggle": bool, "flip_pin": bool, "flip_cup": bool}

For training compatibility we also expose `get_policy_action()` which
returns the same decision packaged as `(cont, disc)` numpy arrays
matching the trained policy's action space (2 cont + 7 disc).

v9.3.1 cycling fix
------------------
The original dispatcher called _try_score() whenever the robot was
carrying ANYTHING, which drove the robot toward whatever _best_target_goal()
returned — even goals where the robot's inventory was useless (e.g. having
only a pin when the goal already had a pin on top and needed a cup).  The
robot oscillated at the SCORING_RADIUS boundary and never reached the cup
pickup priorities.

The rewrite adds:
  • _goal_needs(goal) — categorises each goal as 'pin', 'cup', or 'full'
  • _best_scoreable_goal(has_pin, has_cup) — only returns goals where the
    robot can LEGALLY place something it is currently carrying
  • _try_score_at_range(has_pin, has_cup) — fires only when already in range
  • Full-load strategy — when carrying only a pin and the best scoreable goal
    is an empty Type-A goal, fetch a cup first so both elements can be
    deposited in a single round-trip

Decision priority (highest first)
---------------------------------
1.  EMERGENCY INTERCEPT — charge at an opponent who is about to score.
2.  ENDGAME PARK — last 6 s: race to midfield for parking + SC5b majority.
3.  SCORE NOW — already within SCORING_RADIUS of a goal I can score in.
4.  PRE-PLACEMENT FLIP — orient cup or pin while approaching the goal.
5.  FULL-LOAD FETCH — if I have only a pin and the target is an empty goal,
    grab a cup first so I can score both in one trip.
6.  DRIVE TO SCOREABLE GOAL — I have what the goal needs; head there.
7.  FETCH MISSING ELEMENT — go pick up pin (if I have neither), or cup (if
    I have pin but no scoreable goal is reachable with pin alone).
8.  TOGGLE — flip opponent-owned toggles within cheap-detour range.
9.  DEFAULT — drift toward midfield centre.
"""

import math
from typing import Optional, Tuple, Dict, Any

# MUST import from config.game_rules — the live simulator uses these
# values.  The legacy game_rules.py in the project root is stale and
# has wrong radii that will break the bot.
from config.game_rules import (
    INTAKE_RADIUS, SCORING_RADIUS,
    TOGGLE_INTERACTION_RANGE, MIDFIELD_CENTER, MIDFIELD_HALF,
)

CENTER_GOAL_ID = 4
TWO_PI = 2.0 * math.pi


def _norm_angle(a: float) -> float:
    """Normalise angle to (-pi, pi]."""
    while a >  math.pi:  a -= TWO_PI
    while a <= -math.pi: a += TWO_PI
    return a


def _eff_clear_up(cup) -> bool:
    """True iff the CLEAR (white) side of the cup is currently UP."""
    flipped = getattr(cup, "flipped", False)
    return (not cup.clear_on_top) if flipped else cup.clear_on_top


# ─────────────────────────────────────────────────────────────────────────── #
class HeuristicBot:
    """Single-robot deterministic heuristic agent.

    Construct one per robot:
        bot = HeuristicBot(robot_id="red1", sim=sim)
        # every control step:
        cont, disc = bot.get_policy_action()
    """

    # Bot-side flip cooldown (ticks) — must be ≥ env_wrapper's COOLDOWN_FLIP
    FLIP_COOLDOWN_STEPS = 8

    def __init__(self, robot_id: str, sim):
        self.robot_id = robot_id
        self.sim = sim
        self._robot_cache = None
        # Sticky targets — prevents partners constantly swapping
        self._target_pin_id: Optional[int] = None
        self._target_cup_id: Optional[int] = None
        self._target_goal_id: Optional[int] = None
        # Per-bot flip lockouts
        self._flip_pin_lockout = 0
        self._flip_cup_lockout = 0
        # Diagnostic
        self.last_reason: str = "init"

    # ── Robot / alliance helpers ─────────────────────────────────────────── #

    @property
    def robot(self):
        if self._robot_cache is None or self._robot_cache.robot_id != self.robot_id:
            for r in self.sim.robots:
                if r.robot_id == self.robot_id:
                    self._robot_cache = r
                    break
        return self._robot_cache

    @property
    def alliance(self) -> str:
        return self.robot.alliance

    @property
    def opp_alliance(self) -> str:
        return "blue" if self.alliance == "red" else "red"

    def _pos(self) -> Tuple[float, float]:
        p = self.robot.body.position
        return float(p.x), float(p.y)

    # ── Geometry helpers ─────────────────────────────────────────────────── #

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _drive_to(self, target_xy: Tuple[float, float],
                  full_speed: bool = True) -> Tuple[float, float]:
        """Return (left, right) wheel commands to steer toward target_xy."""
        rx, ry = self._pos()
        tx, ty = target_xy
        desired = math.atan2(ty - ry, tx - rx)
        err = _norm_angle(desired - float(self.robot.body.angle))

        if abs(err) > math.radians(52):          # spin-in-place
            sgn = 1.0 if err > 0 else -1.0
            return -sgn, sgn

        base = 1.0 if full_speed else 0.7
        bias = max(-0.6, min(0.6, err * 1.2))
        left  = base - bias
        right = base + bias
        return max(-1.0, min(1.0, left)), max(-1.0, min(1.0, right))

    def _toggle_for_goal(self, goal) -> Optional[object]:
        """Return the toggle that controls this goal's yellow ownership."""
        if goal.goal_id == CENTER_GOAL_ID:
            return None
        dx = goal.x - 72.0
        dy = goal.y - 72.0
        tid = (1 if dx <= 0 else 2) if abs(dx) >= abs(dy) else (3 if dy <= 0 else 4)
        for t in self.sim.toggles:
            if t.toggle_id == tid:
                return t
        return None

    def _valid_goal_for_me(self, goal) -> bool:
        return goal.alliance in ("neutral", self.alliance)

    # ── Goal categorisation ──────────────────────────────────────────────── #

    @staticmethod
    def _goal_needs(goal) -> str:
        """What element type does this goal need placed next?

        'pin'  → goal is empty, or has a cup on top  (place a pin)
        'cup'  → goal has a pin on top               (place a cup)
        'full' → goal already has 3 elements
        """
        n = len(goal.stack)
        if n >= 3:
            return "full"
        if n == 0:
            return "pin"
        _, top_is_pin = goal.stack[-1]
        return "cup" if top_is_pin else "pin"

    def _best_scoreable_goal(self, has_pin: bool, has_cup: bool):
        """Best goal where I can LEGALLY place something I'm currently holding.

        Only considers goals compatible with current inventory:
          has_pin → goals that need a pin  (empty or cup-on-top)
          has_cup → goals that need a cup  (pin-on-top)
        Never returns a full goal or a goal whose next element I don't have.
        """
        my_pos = self._pos()
        best = None
        best_sc = -1e9
        for g in self.sim.goals:
            if not self._valid_goal_for_me(g):
                continue
            needs = self._goal_needs(g)
            if needs == "full":
                continue
            can_score = (needs == "pin" and has_pin) or (needs == "cup" and has_cup)
            if not can_score:
                continue
            d = self._dist(my_pos, (g.x, g.y))
            sc = -d
            tog = self._toggle_for_goal(g)
            if tog is not None and tog.owner == self.alliance:
                sc += 20.0
            if g.goal_id == CENTER_GOAL_ID and not self._we_have_midfield_majority():
                sc -= 15.0
            if sc > best_sc:
                best_sc = sc
                best = g
        if best is not None:
            self._target_goal_id = best.goal_id
        return best

    # ── Immediate scoring ────────────────────────────────────────────────── #

    def _try_score_at_range(self, has_pin: bool, has_cup: bool):
        """If already within SCORING_RADIUS of a goal I can score in, score.

        Iterates ALL goals so consecutive element placement works: after
        scoring a pin the robot is still at the same goal on the next tick
        and immediately places the cup without an extra approach step.
        """
        my_pos = self._pos()
        for g in self.sim.goals:
            if not self._valid_goal_for_me(g):
                continue
            d = self._dist(my_pos, (g.x, g.y))
            if d > SCORING_RADIUS:
                continue
            needs = self._goal_needs(g)
            if needs == "full":
                continue
            a = self._zero_action()
            if needs == "pin" and has_pin:
                a["score_pin"] = True
                self.last_reason = f"score_pin:{g.goal_id}"
                self._target_goal_id = g.goal_id
                return a
            if needs == "cup" and has_cup:
                a["score_cup"] = True
                self.last_reason = f"score_cup:{g.goal_id}"
                self._target_goal_id = g.goal_id
                return a
        return None

    # ── Element pickup helpers ───────────────────────────────────────────── #

    def _pin_value(self, pin) -> float:
        c = pin.color
        own_solid  = (self.alliance == "red"  and c == "red")  or (self.alliance == "blue" and c == "blue")
        opp_solid  = (self.alliance == "red"  and c == "blue") or (self.alliance == "blue" and c == "red")
        own_yellow = (self.alliance == "red"  and c == "red_yellow")  or (self.alliance == "blue" and c == "blue_yellow")
        opp_yellow = (self.alliance == "red"  and c == "blue_yellow") or (self.alliance == "blue" and c == "red_yellow")
        if own_solid:    return 1.0
        if own_yellow:   return 1.4
        if c == "yellow": return 1.2
        if opp_yellow:   return 0.5
        if opp_solid:    return -1.0
        return 0.1

    def _pick_best_pin(self):
        my_pos = self._pos()
        best = None
        best_sc = -1e9
        for p in self.sim.pins:
            if p.scored or p.carried_by is not None:
                continue
            d = self._dist(my_pos, (float(p.body.position.x), float(p.body.position.y)))
            val = self._pin_value(p)
            if val <= 0:
                continue
            if self._target_pin_id == p.pin_id:
                d *= 0.7   # sticky bonus
            sc = val * 12.0 - d
            if sc > best_sc:
                best_sc = sc
                best = p
        return best

    def _pick_best_cup(self):
        my_pos = self._pos()
        best = None
        best_d = float("inf")
        for c in self.sim.cups:
            if c.scored or c.carried_by is not None:
                continue
            d = self._dist(my_pos, (float(c.body.position.x), float(c.body.position.y)))
            if self._target_cup_id == id(c):
                d *= 0.7
            if d < best_d:
                best_d = d
                best = c
        return best

    def _action_get_pin(self, a: dict) -> dict:
        pin = self._pick_best_pin()
        if pin is None:
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=False)
            self.last_reason = "no_pin_available"
            return a
        px, py = float(pin.body.position.x), float(pin.body.position.y)
        d = self._dist(self._pos(), (px, py))
        if d <= INTAKE_RADIUS:
            a["intake"] = True
            self.last_reason = f"intake_pin:{pin.pin_id}"
        else:
            a["left"], a["right"] = self._drive_to((px, py))
            self.last_reason = f"approach_pin:{pin.pin_id}"
        self._target_pin_id = pin.pin_id
        return a

    def _action_get_cup(self, a: dict, cup) -> dict:
        cx, cy = float(cup.body.position.x), float(cup.body.position.y)
        d = self._dist(self._pos(), (cx, cy))
        if d <= INTAKE_RADIUS:
            a["intake"] = True
            self.last_reason = f"intake_cup:{id(cup)}"
        else:
            a["left"], a["right"] = self._drive_to((cx, cy))
            self.last_reason = f"approach_cup:{id(cup)}"
        self._target_cup_id = id(cup)
        return a

    # ── Top-level dispatcher ─────────────────────────────────────────────── #

    def get_sim_action(self) -> Dict[str, Any]:
        a = self._zero_action()
        if self._flip_pin_lockout > 0: self._flip_pin_lockout -= 1
        if self._flip_cup_lockout > 0: self._flip_cup_lockout -= 1

        # ── 1. EMERGENCY INTERCEPT ─────────────────────────────────────── #
        intercept = self._find_intercept_target()
        if intercept is not None:
            ox = float(intercept.body.position.x)
            oy = float(intercept.body.position.y)
            a["left"], a["right"] = self._drive_to((ox, oy), full_speed=True)
            self.last_reason = f"intercept:{intercept.robot_id}"
            return a

        # ── 2. ENDGAME PARK ────────────────────────────────────────────── #
        if self._endgame_should_park():
            a["left"], a["right"] = self._drive_to(self._best_park_position())
            self.last_reason = "endgame_park"
            return a

        has_pin = self.robot.carrying_pin is not None
        has_cup = self.robot.carrying_cup is not None

        # ── 3. SCORE NOW — if already at a scoreable goal ──────────────── #
        score_now = self._try_score_at_range(has_pin, has_cup)
        if score_now is not None:
            return score_now

        # ── 4. PRE-PLACEMENT FLIPS — orient elements en route ─────────── #
        # Find the goal we'd actually drive to next
        target_goal = self._best_scoreable_goal(has_pin, has_cup)
        if target_goal is not None:
            if has_cup and self._flip_cup_lockout == 0:
                should, _ = self._should_flip_cup_for_goal(target_goal)
                if should:
                    a["flip_cup"] = True
                    self._flip_cup_lockout = self.FLIP_COOLDOWN_STEPS
                    a["left"], a["right"] = self._drive_to((target_goal.x, target_goal.y))
                    self.last_reason = f"flip_cup_for_goal:{target_goal.goal_id}"
                    return a
            if has_pin and self._flip_pin_lockout == 0:
                should, _ = self._should_flip_pin_for_goal(target_goal)
                if should:
                    a["flip_pin"] = True
                    self._flip_pin_lockout = self.FLIP_COOLDOWN_STEPS
                    a["left"], a["right"] = self._drive_to((target_goal.x, target_goal.y))
                    self.last_reason = f"flip_pin_for_goal:{target_goal.goal_id}"
                    return a

        # ── 5. FETCH MISSING ELEMENTS ─────────────────────────────────── #
        #
        # Decision table:
        #   (F, F) → always get a pin first (cups can't start a stack)
        #   (T, F) → three sub-cases:
        #            (a) target goal needs 'cup' → get cup so we can score
        #                (this happens when all non-full goals have a pin
        #                 on top; _best_scoreable_goal returns None for us)
        #            (b) target goal needs 'pin' AND cups exist → fetch cup
        #                FIRST so we can deposit both in one round-trip
        #            (c) target goal needs 'pin' AND no cups left → just go
        #                score the pin now (fall through to step 6)
        #   (F, T) → target goal needs 'cup' (we have it) → drive there
        #            no pin needed — fall through to step 6
        #            if no such goal exists → get a pin to start a new stack
        #   (T, T) → fall through to step 6 (drive to best scoreable goal)

        if not has_pin and not has_cup:
            return self._action_get_pin(a)

        if has_pin and not has_cup:
            if target_goal is None:
                # No goal we can score a pin at right now — all accessible
                # goals already have a pin on top; we need a cup to unblock.
                cup = self._pick_best_cup()
                if cup is not None:
                    return self._action_get_cup(a, cup)
                # No cups anywhere → drift to midfield
                a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=False)
                self.last_reason = "no_cup_no_goal"
                return a
            else:
                needs = self._goal_needs(target_goal)
                if needs == "pin":
                    # We CAN score here with just a pin, but getting a cup
                    # first lets us deposit two elements in one trip.
                    cup = self._pick_best_cup()
                    if cup is not None:
                        return self._action_get_cup(a, cup)
                    # No cups available — go score the pin solo
                    # (fall through to step 6)

        if not has_pin and has_cup:
            if target_goal is None:
                # No goal has a pin on top yet — we have a cup but nowhere
                # to put it.  Pick up a pin to start a new stack.
                return self._action_get_pin(a)
            # Otherwise fall through to step 6

        # ── 6. DRIVE TO SCOREABLE GOAL ────────────────────────────────── #
        if target_goal is not None:
            a["left"], a["right"] = self._drive_to((target_goal.x, target_goal.y),
                                                    full_speed=True)
            self.last_reason = f"approach_goal:{target_goal.goal_id}"
            return a

        # ── 7. TOGGLE — cheap detour ──────────────────────────────────── #
        toggle = self._find_useful_toggle()
        if toggle is not None:
            tx, ty = float(toggle.x), float(toggle.y)
            d = self._dist(self._pos(), (tx, ty))
            if d <= TOGGLE_INTERACTION_RANGE:
                a["toggle"] = True
                self.last_reason = f"flip_toggle:{toggle.toggle_id}"
            else:
                a["left"], a["right"] = self._drive_to((tx, ty))
                self.last_reason = f"approach_toggle:{toggle.toggle_id}"
            return a

        # ── 8. DEFAULT ────────────────────────────────────────────────── #
        a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=False)
        self.last_reason = "default_midfield"
        return a

    # ── Policy-action API ────────────────────────────────────────────────── #

    def get_policy_action(self):
        """Return `(cont, disc)` numpy arrays matching env_wrapper format.

        cont = [left, right]
        disc = [intake, score_pin, score_cup, toggle, flip_pin, flip_cup, match_load]
        """
        import numpy as np
        a = self.get_sim_action()
        cont = np.array([a["left"], a["right"]], dtype=np.float32)
        disc = np.array([
            1.0 if a["intake"]    else 0.0,
            1.0 if a["score_pin"] else 0.0,
            1.0 if a["score_cup"] else 0.0,
            1.0 if a["toggle"]    else 0.0,
            1.0 if a["flip_pin"]  else 0.0,
            1.0 if a["flip_cup"]  else 0.0,
            0.0,  # match_load — never used by the bot
        ], dtype=np.float32)
        return cont, disc

    @staticmethod
    def _zero_action() -> Dict[str, Any]:
        return {"left": 0.0, "right": 0.0,
                "intake": False, "score_pin": False, "score_cup": False,
                "toggle": False, "flip_pin": False, "flip_cup": False}

    # ── Priority 1 — emergency intercept ─────────────────────────────────── #

    def _find_intercept_target(self):
        """Return an opponent robot we should charge to disrupt, or None."""
        # Don't abandon a near-complete score ourselves
        if self.robot.carrying_pin or self.robot.carrying_cup:
            for g in self.sim.goals:
                if self._valid_goal_for_me(g) and \
                   self._dist(self._pos(), (g.x, g.y)) <= SCORING_RADIUS + 4:
                    return None

        my_pos = self._pos()
        best = None
        best_threat = 0.0
        for opp in self.sim.robots:
            if opp.alliance == self.alliance:
                continue
            if opp.carrying_pin is None and opp.carrying_cup is None:
                continue
            ox, oy = float(opp.body.position.x), float(opp.body.position.y)
            for g in self.sim.goals:
                if g.alliance == self.alliance:
                    continue
                d_opp_goal = self._dist((ox, oy), (g.x, g.y))
                if d_opp_goal > SCORING_RADIUS * 1.8:
                    continue
                d_me_opp = self._dist(my_pos, (ox, oy))
                if d_me_opp > d_opp_goal + 8.0:
                    continue
                val = 1.0 if opp.carrying_pin else 0.5
                threat = val / (1.0 + d_opp_goal / 5.0)
                if threat > best_threat:
                    best_threat = threat
                    best = opp
        return best

    # ── Priority 2 — endgame park ────────────────────────────────────────── #

    def _endgame_should_park(self) -> bool:
        if not self.sim.rules_engine.endgame_active:
            return False
        if self.robot.carrying_pin or self.robot.carrying_cup:
            return False
        if self.sim.time_remaining > 6.0:
            return False
        return True

    def _best_park_position(self) -> Tuple[float, float]:
        return (72.0, 72.0)

    # ── Priority 4 helpers — pre-placement flip ───────────────────────────── #

    def _should_flip_cup_for_goal(self, goal):
        """Return (should_flip, goal) if carrying cup is wrong orientation
        for the given goal's top pin."""
        cup = self.robot.carrying_cup
        if cup is None or self._goal_needs(goal) != "cup":
            return False, None
        correct = self._correct_cup_clear_up(goal)
        if correct is None:
            return False, None
        if _eff_clear_up(cup) != correct:
            return True, goal
        return False, None

    def _should_flip_pin_for_goal(self, goal):
        """Return (should_flip, goal) if carrying yellow-half pin should
        have yellow facing UP for the given goal."""
        pin = self.robot.carrying_pin
        if pin is None or not pin.is_yellow:
            return False, None
        tog = self._toggle_for_goal(goal)
        if tog is None or tog.owner != self.alliance:
            if goal.goal_id != CENTER_GOAL_ID or not self._we_have_midfield_majority():
                return False, None
        if pin.get_up_color() != "yellow":
            return True, goal
        return False, None

    def _correct_cup_clear_up(self, goal) -> Optional[bool]:
        """For a goal whose top is a pin, what should cup.clear_up be?"""
        if not goal.stack or not goal.stack[-1][1]:
            return None
        top_pin, _ = goal.stack[-1]
        up = top_pin.get_up_color()
        if up == "red":
            return True  if self.alliance == "blue" else False
        if up == "blue":
            return True  if self.alliance == "red"  else False
        if up == "yellow":
            tog = self._toggle_for_goal(goal)
            if tog is None or tog.owner not in ("red", "blue"):
                return None
            return False if tog.owner == self.alliance else True
        return None

    # ── Priority 7 — toggle ──────────────────────────────────────────────── #

    def _find_useful_toggle(self):
        """Return the best toggle within detour range that isn't already mine."""
        my_pos = self._pos()
        best = None
        best_sc = -1e9
        for t in self.sim.toggles:
            if t.owner == self.alliance:
                continue
            d = self._dist(my_pos, (float(t.x), float(t.y)))
            if d > 36.0:
                continue
            sc = -d
            for g in self.sim.goals:
                if self._toggle_for_goal(g) is t:
                    for obj, is_pin in g.stack:
                        if is_pin and obj.is_yellow:
                            sc += 8.0
            if sc > best_sc:
                best_sc = sc
                best = t
        return best

    # ── Midfield helpers ─────────────────────────────────────────────────── #

    def _we_have_midfield_majority(self) -> bool:
        own = opp = 0
        for r in self.sim.robots:
            if self._in_midfield(float(r.body.position.x), float(r.body.position.y)):
                if r.alliance == self.alliance: own += 1
                else:                           opp += 1
        return own > opp

    @staticmethod
    def _in_midfield(x: float, y: float) -> bool:
        mx, my = MIDFIELD_CENTER
        return abs(x - mx) + abs(y - my) <= MIDFIELD_HALF + 10.0
