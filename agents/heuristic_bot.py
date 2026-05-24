"""
agents/heuristic_bot.py
=======================
Rule-based "perfect-play" bot for VEX Override.  v9.4

What changed in v9.4 vs v9.3.2
-------------------------------
The previous bot was correct but slow and uncoordinated, scoring ~0-20 pts
per alliance per match.  v9.4 targets 100+ pts per alliance via:

1. **Partner coordination** — module-level `_shared_targets` registry lets
   two alliance bots avoid claiming the same pin/cup/goal.  Each bot
   publishes its targets every step; siblings consult the registry when
   picking, then choose alternatives if a partner already owns a target.

2. **Stack-value goal selection** — `_goal_expected_value(goal, has_pin,
   has_cup)` scores each goal by the points unlockable with our current
   inventory, weighted by distance.  Strongly prefers alliance-owned goals,
   toggle-owned goals (yellow scoring), and goals that complete a stack
   we've already started.

3. **Pre-orient after pickup** — as soon as we pick up a pin or cup we
   choose the target goal and immediately start flipping toward the correct
   orientation while still driving toward the goal.  The env_wrapper's
   2-second flip cooldown overlaps with travel time instead of compounding
   on top of it.

4. **Faster navigation** — wider spin threshold (110°) lets the bot turn
   in arcs while driving instead of stopping and pivoting.  Approach
   geometry targets a point 9" inside the goal (still within SCORING_RADIUS
   = 12") so the robot does not crash into the goal's physics body and
   bounce out of range.  When inside scoring range, wheels are zeroed to
   prevent oscillation.

5. **Aggressive endgame** — last 8 s: dump any carried element into the
   nearest valid goal with no orientation checks.  Last 3 s: race to
   midfield regardless of inventory.

6. **Tightened lockouts** — `FLIP_COOLDOWN_STEPS = 40` matches the
   env_wrapper's `COOLDOWN_FLIP` so the bot doesn't waste 4/5 flip
   attempts on the silently-gated path.

7. **Smarter intercept** — only fires when an opponent is essentially
   touching one of their valid goals AND we're physically closer than
   they are.  Prevents giving up our own scoring trip to chase a phantom
   threat.

Decision priority (highest first)
---------------------------------
1.  ENDGAME PARK (last ~3 s, regardless of inventory)
2.  ENDGAME DUMP  (last ~8 s, dump any carried element somewhere legal)
3.  EMERGENCY INTERCEPT (pin-carrying opponent within scoring contact)
4.  SCORE NOW (within SCORING_RADIUS of a goal we can score in)
5.  PRE-PLACEMENT FLIP (orient pin/cup en route)
6.  FETCH MISSING ELEMENT (pin first if empty-handed, then cup)
7.  DRIVE TO SCOREABLE GOAL
8.  TOGGLE (proactive route to nearest unowned toggle)
9.  DEFAULT — drift toward midfield centre.

Opportunistic toggle injection (`_finalize_action`): any non-scoring step
auto-injects `toggle=True` when within `TOGGLE_INTERACTION_RANGE` of an
unowned toggle, so quadrant control gets claimed "for free" while driving.
"""

import math
from typing import Optional, Tuple, Dict, Any, List

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

    # ── Tunables ───────────────────────────────────────────────────────── #
    # Matches env_wrapper.COOLDOWN_FLIP so we don't waste attempts on the
    # silently-gated path.  Pin flips are expensive — 2 s of cooldown.
    FLIP_COOLDOWN_STEPS = 40

    # After this many consecutive steps chasing the same pin/cup without
    # picking it up, blacklist that element and try a different one.
    PICKUP_TIMEOUT = 50            # ~2.5 s at 20 Hz
    BLACKLIST_DURATION = 80        # ~4 s

    # Approach zones (× INTAKE_RADIUS = 10").
    INTAKE_SLOW_MULT = 1.6         # below 16" → 70% speed
    INTAKE_CRAWL_MULT = 1.05       # below 10.5" → 35% (don't fully stop)

    # _drive_to() turn behaviour
    SPIN_THRESHOLD_DEG = 110.0     # turn-in-place only above this error
    SLOW_RADIUS = 16.0             # below this distance from target, slow down

    # Goal approach: target a point this far from the goal center (inside
    # SCORING_RADIUS = 12") so we don't drive INTO the goal physics body
    # and bounce out of range.
    APPROACH_GOAL_INSET = 9.0

    # If within this distance, brake completely — we're in scoring range
    # and scoring will fire next step anyway.
    SCORE_BRAKE_RADIUS = SCORING_RADIUS  # 12"

    # Endgame timing (seconds remaining)
    ENDGAME_DUMP_TIME = 8.0
    ENDGAME_PARK_TIME = 3.0

    # Toggle routing (priority 8 only — opportunistic injection has no limit)
    TOGGLE_ROUTE_RANGE = 60.0

    # ── Module-level shared state for partner coordination ────────────── #
    # robot_id -> {"pin": pin_id, "cup": id(cup_obj), "goal": goal_id}
    _shared_targets: Dict[str, Dict[str, Optional[int]]] = {}

    def __init__(self, robot_id: str, sim):
        self.robot_id = robot_id
        self.sim = sim
        self._robot_cache = None
        # Sticky targets — prevents partners constantly swapping
        self._target_pin_id: Optional[int]  = None
        self._target_cup_id: Optional[int]  = None
        self._target_goal_id: Optional[int] = None
        # How many steps we've been chasing the current pin/cup target.
        self._target_pin_steps: int = 0
        self._target_cup_steps: int = 0
        # Elements we've given up on — avoid them until countdown expires.
        self._pin_blacklist: set = set()
        self._pin_blacklist_cd: int = 0
        self._cup_blacklist: set = set()
        self._cup_blacklist_cd: int = 0
        # Per-bot flip lockouts
        self._flip_pin_lockout = 0
        self._flip_cup_lockout = 0
        # Diagnostic
        self.last_reason: str = "init"
        # Initialise our slot in the shared registry
        HeuristicBot._shared_targets[self.robot_id] = {
            "pin": None, "cup": None, "goal": None,
        }

    def reset(self):
        """Clear all per-episode state.  Call after env.reset()."""
        self._robot_cache       = None
        self._target_pin_id     = None
        self._target_cup_id     = None
        self._target_goal_id    = None
        self._target_pin_steps  = 0
        self._target_cup_steps  = 0
        self._pin_blacklist     = set()
        self._pin_blacklist_cd  = 0
        self._cup_blacklist     = set()
        self._cup_blacklist_cd  = 0
        self._flip_pin_lockout  = 0
        self._flip_cup_lockout  = 0
        self.last_reason        = "init"
        HeuristicBot._shared_targets[self.robot_id] = {
            "pin": None, "cup": None, "goal": None,
        }

    # ── Robot / alliance helpers ───────────────────────────────────────── #

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

    # ── Geometry helpers ──────────────────────────────────────────────── #

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _drive_to(self, target_xy: Tuple[float, float],
                  full_speed: bool = True) -> Tuple[float, float]:
        """Return (left, right) wheel commands to steer toward target_xy.

        - Turn-in-place only when heading error exceeds SPIN_THRESHOLD_DEG.
        - Within SLOW_RADIUS, scale down throttle so we don't overshoot.
        """
        rx, ry = self._pos()
        tx, ty = target_xy
        dx, dy = tx - rx, ty - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return 0.0, 0.0
        desired = math.atan2(dy, dx)
        err = _norm_angle(desired - float(self.robot.body.angle))

        spin_thresh = math.radians(self.SPIN_THRESHOLD_DEG)
        if abs(err) > spin_thresh:
            sgn = 1.0 if err > 0 else -1.0
            return -sgn, sgn

        base = 1.0 if full_speed else 0.7
        # Slow approach when close to target — prevents overshoot
        if dist < self.SLOW_RADIUS:
            base *= max(0.4, dist / self.SLOW_RADIUS)
        bias = max(-0.7, min(0.7, err * 1.4))
        left  = base - bias
        right = base + bias
        return max(-1.0, min(1.0, left)), max(-1.0, min(1.0, right))

    def _approach_goal_point(self, goal) -> Tuple[float, float]:
        """Target a point INSET inside the goal radius, so the robot doesn't
        crash into the goal's physics body."""
        rx, ry = self._pos()
        dx, dy = goal.x - rx, goal.y - ry
        d = math.hypot(dx, dy)
        if d < self.APPROACH_GOAL_INSET:
            return (goal.x, goal.y)
        # Step back from the goal center along the line robot→goal
        sf = (d - self.APPROACH_GOAL_INSET) / d
        return (rx + dx * sf, ry + dy * sf)

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

    # ── Partner coordination via shared registry ──────────────────────── #

    def _publish_targets(self):
        """Update the shared registry with our current targets."""
        HeuristicBot._shared_targets[self.robot_id] = {
            "pin":  self._target_pin_id,
            "cup":  self._target_cup_id,
            "goal": self._target_goal_id,
        }

    def _partner_targets(self) -> Dict[str, set]:
        """Return targets owned by ALLIED partners (excluding ourselves)."""
        pins, cups, goals = set(), set(), set()
        # Build a fast alliance-lookup
        alliance_of: Dict[str, str] = {r.robot_id: r.alliance for r in self.sim.robots}
        for rid, tgt in HeuristicBot._shared_targets.items():
            if rid == self.robot_id:
                continue
            if alliance_of.get(rid) != self.alliance:
                continue
            if tgt.get("pin")  is not None: pins.add(tgt["pin"])
            if tgt.get("cup")  is not None: cups.add(tgt["cup"])
            if tgt.get("goal") is not None: goals.add(tgt["goal"])
        return {"pins": pins, "cups": cups, "goals": goals}

    # ── Goal categorisation ───────────────────────────────────────────── #

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

    def _goal_expected_value(self, goal, has_pin: bool, has_cup: bool) -> float:
        """Estimate points unlockable at this goal with our current inventory.

        Used to rank scoreable goals.  Higher = better target.
        """
        needs = self._goal_needs(goal)
        if needs == "full":
            return -1.0
        can_score = (needs == "pin" and has_pin) or (needs == "cup" and has_cup)
        if not can_score:
            return -1.0

        n = len(goal.stack)
        v = 0.0
        # Base: a pin placement scores 5 pts (UP half visible, post hides DOWN)
        # if it's the first pin; or 10 pts (both halves visible through cup gap)
        # if it's the second pin on top of a cup.
        if needs == "pin":
            v += 10.0 if n == 2 else 5.0   # 2nd pin (after pin+cup) scores both halves
        elif needs == "cup":
            # Cup itself = 0 pts, but enables the next pin → estimate as 8 pts
            # since a future pin on this cup would score ~10 pts.
            v += 6.0

        # Toggle bonus: if alliance owns the toggle, yellow halves placed here
        # double in value (5 → 10).  We don't know if our pin has yellow halves
        # without inspecting the carried pin, but a yellow-owning bias is fine.
        tog = self._toggle_for_goal(goal)
        if tog is not None and tog.owner == self.alliance:
            v += 4.0
        elif tog is not None and tog.owner == self.opp_alliance:
            v -= 2.0   # opponent gets the yellow bonus → mildly discourage

        # Alliance goals are sticky-good: only OUR alliance can ever score here
        if goal.alliance == self.alliance:
            v += 4.0

        # Center goal: encourage only if we have midfield majority potential
        if goal.goal_id == CENTER_GOAL_ID:
            v -= 2.0   # mild penalty: harder to defend, opponents also target it
            if self._we_have_midfield_majority():
                v += 6.0   # SC5b yellows go to us → much more valuable

        # If a partner has already claimed this goal, deprioritise hard
        partners = self._partner_targets()
        if goal.goal_id in partners["goals"]:
            v -= 10.0

        return v

    def _best_scoreable_goal(self, has_pin: bool, has_cup: bool):
        """Best goal where I can LEGALLY place something I'm currently holding.

        Picks by expected value MINUS distance penalty.
        """
        my_pos = self._pos()
        best = None
        best_sc = -1e9
        for g in self.sim.goals:
            if not self._valid_goal_for_me(g):
                continue
            v = self._goal_expected_value(g, has_pin, has_cup)
            if v < 0:
                continue
            d = self._dist(my_pos, (g.x, g.y))
            sc = v * 4.0 - d
            if sc > best_sc:
                best_sc = sc
                best = g
        if best is not None:
            self._target_goal_id = best.goal_id
        return best

    # ── Immediate scoring ─────────────────────────────────────────────── #

    def _try_score_at_range(self, has_pin: bool, has_cup: bool):
        """If already within SCORING_RADIUS of a goal I can score in, score.

        Iterates ALL goals so consecutive element placement works: after
        scoring a pin the robot is still at the same goal on the next tick
        and immediately places the cup without an extra approach step.

        Includes orientation checks for both pin and cup before scoring —
        flips if needed so element halves land facing the right direction.
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
                should_flip, _ = self._should_flip_pin_for_goal(g)
                if should_flip:
                    if self._flip_pin_lockout == 0:
                        a["flip_pin"] = True
                        self._flip_pin_lockout = self.FLIP_COOLDOWN_STEPS
                        self.last_reason = f"flip_pin_at_goal:{g.goal_id}"
                        return a
                    else:
                        # Hold position while flip cooldown ticks down
                        self.last_reason = f"await_pin_flip:{g.goal_id}"
                        return a
                a["score_pin"] = True
                self.last_reason = f"score_pin:{g.goal_id}"
                self._target_goal_id = g.goal_id
                return a
            if needs == "cup" and has_cup:
                cup = self.robot.carrying_cup
                correct = self._correct_cup_clear_up(g)
                needs_flip = (correct is not None and cup is not None
                              and _eff_clear_up(cup) != correct)
                if needs_flip:
                    if self._flip_cup_lockout == 0:
                        a["flip_cup"] = True
                        self._flip_cup_lockout = self.FLIP_COOLDOWN_STEPS
                        self.last_reason = f"flip_cup_at_goal:{g.goal_id}"
                        return a
                    else:
                        self.last_reason = f"await_cup_flip:{g.goal_id}"
                        return a
                a["score_cup"] = True
                self.last_reason = f"score_cup:{g.goal_id}"
                self._target_goal_id = g.goal_id
                return a
        return None

    # ── Element pickup helpers ────────────────────────────────────────── #

    def _pin_value(self, pin) -> float:
        """How valuable is THIS pin for our alliance?  Negative = avoid."""
        c = pin.color
        own_solid   = (self.alliance == "red"  and c == "red")  or (self.alliance == "blue" and c == "blue")
        opp_solid   = (self.alliance == "red"  and c == "blue") or (self.alliance == "blue" and c == "red")
        own_yellow  = (self.alliance == "red"  and c == "red_yellow")  or (self.alliance == "blue" and c == "blue_yellow")
        opp_yellow  = (self.alliance == "red"  and c == "blue_yellow") or (self.alliance == "blue" and c == "red_yellow")
        full_yellow = (c == "yellow" or c == "yellow_yellow")
        if own_yellow:   return 1.5
        if own_solid:    return 1.0
        if full_yellow:  return 1.2
        if opp_yellow:   return 0.4
        if opp_solid:    return -1.0
        return 0.1

    def _pick_best_pin(self):
        my_pos = self._pos()
        timed_out = self._target_pin_steps >= self.PICKUP_TIMEOUT
        if timed_out and self._target_pin_id is not None:
            self._pin_blacklist.add(self._target_pin_id)
            self._pin_blacklist_cd = max(self._pin_blacklist_cd, self.BLACKLIST_DURATION)
            self._target_pin_id = None
            self._target_pin_steps = 0

        partner_pins = self._partner_targets()["pins"]
        best = None
        best_sc = -1e9
        for p in self.sim.pins:
            if p.scored or p.carried_by is not None:
                continue
            if p.pin_id in self._pin_blacklist:
                continue
            val = self._pin_value(p)
            if val <= 0:
                continue
            d = self._dist(my_pos, (float(p.body.position.x), float(p.body.position.y)))
            if self._target_pin_id == p.pin_id:
                d *= 0.75   # sticky bonus
            if p.pin_id in partner_pins:
                d += 50.0   # partner already heading there → strong deprioritise
            sc = val * 12.0 - d
            if sc > best_sc:
                best_sc = sc
                best = p
        # Fallback: if everything was blacklisted or partner-claimed, ignore
        # both filters and pick the absolute closest valuable pin.
        if best is None:
            self._pin_blacklist.clear()
            self._pin_blacklist_cd = 0
            for p in self.sim.pins:
                if p.scored or p.carried_by is not None:
                    continue
                val = self._pin_value(p)
                if val <= 0:
                    continue
                d = self._dist(my_pos, (float(p.body.position.x), float(p.body.position.y)))
                sc = val * 12.0 - d
                if sc > best_sc:
                    best_sc = sc
                    best = p
        return best

    def _pick_best_cup(self):
        my_pos = self._pos()
        timed_out = self._target_cup_steps >= self.PICKUP_TIMEOUT
        if timed_out and self._target_cup_id is not None:
            self._cup_blacklist.add(self._target_cup_id)
            self._cup_blacklist_cd = max(self._cup_blacklist_cd, self.BLACKLIST_DURATION)
            self._target_cup_id = None
            self._target_cup_steps = 0

        partner_cups = self._partner_targets()["cups"]
        best = None
        best_d = float("inf")
        for c in self.sim.cups:
            if c.scored or c.carried_by is not None:
                continue
            cid = id(c)
            if cid in self._cup_blacklist:
                continue
            d = self._dist(my_pos, (float(c.body.position.x), float(c.body.position.y)))
            if self._target_cup_id == cid:
                d *= 0.75
            if cid in partner_cups:
                d += 50.0
            if d < best_d:
                best_d = d
                best = c
        if best is None:
            self._cup_blacklist.clear()
            self._cup_blacklist_cd = 0
            for c in self.sim.cups:
                if c.scored or c.carried_by is not None:
                    continue
                d = self._dist(my_pos, (float(c.body.position.x), float(c.body.position.y)))
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

        if self._target_pin_id != pin.pin_id:
            self._target_pin_id = pin.pin_id
            self._target_pin_steps = 0
        self._target_pin_steps += 1

        # Always fire intake — the sim rejects it when out of range, no penalty.
        a["intake"] = True
        if d <= INTAKE_RADIUS * self.INTAKE_CRAWL_MULT:
            # Don't fully stop — crawl forward so we don't drift out of range.
            a["left"], a["right"] = self._drive_to((px, py), full_speed=False)
            a["left"]  *= 0.35
            a["right"] *= 0.35
            self.last_reason = f"intake_pin:{pin.pin_id}"
        elif d <= INTAKE_RADIUS * self.INTAKE_SLOW_MULT:
            a["left"], a["right"] = self._drive_to((px, py), full_speed=False)
            self.last_reason = f"intake_pin:{pin.pin_id}"
        else:
            a["left"], a["right"] = self._drive_to((px, py), full_speed=True)
            self.last_reason = f"approach_pin:{pin.pin_id}"
        return a

    def _action_get_cup(self, a: dict, cup) -> dict:
        cx, cy = float(cup.body.position.x), float(cup.body.position.y)
        d = self._dist(self._pos(), (cx, cy))
        cid = id(cup)

        if self._target_cup_id != cid:
            self._target_cup_id = cid
            self._target_cup_steps = 0
        self._target_cup_steps += 1

        a["intake"] = True
        if d <= INTAKE_RADIUS * self.INTAKE_CRAWL_MULT:
            a["left"], a["right"] = self._drive_to((cx, cy), full_speed=False)
            a["left"]  *= 0.35
            a["right"] *= 0.35
            self.last_reason = f"intake_cup:{cid}"
        elif d <= INTAKE_RADIUS * self.INTAKE_SLOW_MULT:
            a["left"], a["right"] = self._drive_to((cx, cy), full_speed=False)
            self.last_reason = f"intake_cup:{cid}"
        else:
            a["left"], a["right"] = self._drive_to((cx, cy), full_speed=True)
            self.last_reason = f"approach_cup:{cid}"
        return a

    # ── Post-processing — opportunistic toggle injection ─────────────── #

    def _finalize_action(self, a: dict) -> dict:
        """Opportunistically fire toggle=True whenever passing near an
        unowned toggle, on ANY non-scoring travel or idle action.
        """
        if a["score_pin"] or a["score_cup"] or a["flip_pin"] or a["flip_cup"]:
            return a
        if a["toggle"]:
            return a
        my_pos = self._pos()
        best_t = None
        best_d = float("inf")
        for t in self.sim.toggles:
            if t.owner == self.alliance:
                continue
            d = self._dist(my_pos, (float(t.x), float(t.y)))
            if d <= TOGGLE_INTERACTION_RANGE and d < best_d:
                best_d = d
                best_t = t
        if best_t is not None:
            a["toggle"] = True
            self.last_reason = self.last_reason + f"+toggle:{best_t.toggle_id}"
        return a

    # ── Top-level dispatcher ──────────────────────────────────────────── #

    def get_sim_action(self) -> Dict[str, Any]:
        """Compute + finalise the action for this control step."""
        a = self._compute_action()
        # Publish our targets so siblings can see them next step.
        self._publish_targets()
        return self._finalize_action(a)

    def _compute_action(self) -> Dict[str, Any]:
        """Core decision tree — returns action dict (toggle NOT yet injected)."""
        a = self._zero_action()
        if self._flip_pin_lockout > 0: self._flip_pin_lockout -= 1
        if self._flip_cup_lockout > 0: self._flip_cup_lockout -= 1
        if self._pin_blacklist_cd > 0:
            self._pin_blacklist_cd -= 1
        else:
            self._pin_blacklist.clear()
        if self._cup_blacklist_cd > 0:
            self._cup_blacklist_cd -= 1
        else:
            self._cup_blacklist.clear()

        has_pin = self.robot.carrying_pin is not None
        has_cup = self.robot.carrying_cup is not None
        endgame = (self.sim.rules_engine.endgame_active and
                   self.sim.time_remaining is not None)

        # ── 1. ENDGAME PARK — last ~3 s, race to midfield ─────────────── #
        if endgame and self.sim.time_remaining <= self.ENDGAME_PARK_TIME:
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
            self.last_reason = "endgame_park"
            return a

        # ── 2. ENDGAME DUMP — last ~8 s, dump anything we have ─────────── #
        if endgame and self.sim.time_remaining <= self.ENDGAME_DUMP_TIME and (has_pin or has_cup):
            dump = self._try_dump_at_range(has_pin, has_cup)
            if dump is not None:
                return dump
            target = self._nearest_dump_goal(has_pin, has_cup)
            if target is not None:
                a["left"], a["right"] = self._drive_to(
                    self._approach_goal_point(target), full_speed=True)
                self.last_reason = f"endgame_dump_to:{target.goal_id}"
                self._target_goal_id = target.goal_id
                return a
            # Nothing legal to dump → head to midfield to at least park
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
            self.last_reason = "endgame_park_carry"
            return a

        # ── 3. EMERGENCY INTERCEPT ─────────────────────────────────────── #
        intercept = self._find_intercept_target()
        if intercept is not None:
            ox = float(intercept.body.position.x)
            oy = float(intercept.body.position.y)
            a["left"], a["right"] = self._drive_to((ox, oy), full_speed=True)
            self.last_reason = f"intercept:{intercept.robot_id}"
            return a

        # ── 4. SCORE NOW — if already at a scoreable goal ──────────────── #
        score_now = self._try_score_at_range(has_pin, has_cup)
        if score_now is not None:
            return score_now

        # ── 5. PRE-PLACEMENT FLIPS — orient elements en route ─────────── #
        target_goal = self._best_scoreable_goal(has_pin, has_cup)
        if target_goal is not None:
            if has_cup and self._flip_cup_lockout == 0:
                should, _ = self._should_flip_cup_for_goal(target_goal)
                if should:
                    a["flip_cup"] = True
                    self._flip_cup_lockout = self.FLIP_COOLDOWN_STEPS
                    a["left"], a["right"] = self._drive_to(
                        self._approach_goal_point(target_goal))
                    self.last_reason = f"flip_cup_for_goal:{target_goal.goal_id}"
                    return a
            if has_pin and self._flip_pin_lockout == 0:
                should, _ = self._should_flip_pin_for_goal(target_goal)
                if should:
                    a["flip_pin"] = True
                    self._flip_pin_lockout = self.FLIP_COOLDOWN_STEPS
                    a["left"], a["right"] = self._drive_to(
                        self._approach_goal_point(target_goal))
                    self.last_reason = f"flip_pin_for_goal:{target_goal.goal_id}"
                    return a

        # ── 6. FETCH MISSING ELEMENTS ─────────────────────────────────── #
        # Smart full-load strategy: if we have only a pin and the target goal
        # needs a pin, fetch a cup first so we can deposit both in one trip.

        if not has_pin and not has_cup:
            return self._action_get_pin(a)

        if has_pin and not has_cup:
            if target_goal is None:
                cup = self._pick_best_cup()
                if cup is not None:
                    return self._action_get_cup(a, cup)
                a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=False)
                self.last_reason = "no_cup_no_goal"
                return a
            else:
                needs = self._goal_needs(target_goal)
                if needs == "pin":
                    cup = self._pick_best_cup()
                    if cup is not None:
                        return self._action_get_cup(a, cup)

        if not has_pin and has_cup:
            if target_goal is None:
                return self._action_get_pin(a)

        # ── 7. DRIVE TO SCOREABLE GOAL ────────────────────────────────── #
        if target_goal is not None:
            # If we're already within scoring range, brake — score will fire
            # next step.  Prevents oscillation around the SCORING_RADIUS boundary.
            d = self._dist(self._pos(), (target_goal.x, target_goal.y))
            if d <= self.SCORE_BRAKE_RADIUS:
                a["left"], a["right"] = 0.0, 0.0
                self.last_reason = f"brake_at_goal:{target_goal.goal_id}"
                return a
            a["left"], a["right"] = self._drive_to(
                self._approach_goal_point(target_goal), full_speed=True)
            self.last_reason = f"approach_goal:{target_goal.goal_id}"
            return a

        # ── 8. TOGGLE — proactively route to nearest unowned toggle ──────#
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

        # ── 9. DEFAULT ────────────────────────────────────────────────── #
        a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=False)
        self.last_reason = "default_midfield"
        return a

    # ── Endgame dump helpers ──────────────────────────────────────────── #

    def _nearest_dump_goal(self, has_pin: bool, has_cup: bool):
        """Pick the nearest goal where we can legally drop something — no
        orientation, no value: just unload before time runs out."""
        my_pos = self._pos()
        best = None
        best_d = float("inf")
        for g in self.sim.goals:
            if not self._valid_goal_for_me(g):
                continue
            needs = self._goal_needs(g)
            if needs == "full":
                continue
            can = (needs == "pin" and has_pin) or (needs == "cup" and has_cup)
            if not can:
                continue
            d = self._dist(my_pos, (g.x, g.y))
            if d < best_d:
                best_d = d
                best = g
        return best

    def _try_dump_at_range(self, has_pin: bool, has_cup: bool):
        """Like _try_score_at_range but skips orientation checks — endgame
        only.  A dark-up cup is still better than no points."""
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
                self.last_reason = f"endgame_score_pin:{g.goal_id}"
                return a
            if needs == "cup" and has_cup:
                a["score_cup"] = True
                self.last_reason = f"endgame_score_cup:{g.goal_id}"
                return a
        return None

    # ── Policy-action API ─────────────────────────────────────────────── #

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
            0.0,   # match_load — sim doesn't process it from action dicts
        ], dtype=np.float32)
        return cont, disc

    @staticmethod
    def _zero_action() -> Dict[str, Any]:
        return {"left": 0.0, "right": 0.0,
                "intake": False, "score_pin": False, "score_cup": False,
                "toggle": False, "flip_pin": False, "flip_cup": False}

    # ── Priority 3 — emergency intercept ──────────────────────────────── #

    def _find_intercept_target(self):
        """Return an opponent we should charge to disrupt, or None.

        Only fires when:
          - we're not already inches from a scoreable goal (don't abandon
            our own near-complete trip)
          - opponent is carrying a pin AND within SCORING_RADIUS*1.2 of one
            of THEIR valid goals
          - we are closer to that goal than the opponent (intercept is
            actually feasible)
        """
        my_pos = self._pos()

        # Don't abandon our own scoring trip
        if self.robot.carrying_pin or self.robot.carrying_cup:
            for g in self.sim.goals:
                if self._valid_goal_for_me(g) and \
                   self._dist(my_pos, (g.x, g.y)) <= SCORING_RADIUS + 2:
                    return None

        best = None
        best_threat = 0.0
        for opp in self.sim.robots:
            if opp.alliance == self.alliance:
                continue
            if opp.carrying_pin is None:
                continue
            ox, oy = float(opp.body.position.x), float(opp.body.position.y)
            for g in self.sim.goals:
                # Opponent must be able to score here
                if g.alliance not in ("neutral", opp.alliance):
                    continue
                if self._goal_needs(g) == "full":
                    continue
                d_opp_goal = self._dist((ox, oy), (g.x, g.y))
                if d_opp_goal > SCORING_RADIUS * 1.2:
                    continue
                d_me_goal = self._dist(my_pos, (g.x, g.y))
                # Only intercept if we can plausibly beat them
                if d_me_goal > d_opp_goal + 4.0:
                    continue
                threat = 1.0 / (1.0 + d_opp_goal / 4.0)
                if threat > best_threat:
                    best_threat = threat
                    best = opp
        return best

    # ── Priority 5 helpers — pre-placement flip ──────────────────────── #

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
        have yellow facing UP for the given goal.

        Uses pin.up_half_name (a plain string) instead of get_up_color()
        (which returns an RGB tuple) so the comparison is correct.
        """
        pin = self.robot.carrying_pin
        if pin is None or not pin.is_yellow:
            return False, None
        tog = self._toggle_for_goal(goal)
        # Yellow scoring only happens if our alliance owns the toggle
        # (or, for center, we have midfield majority).  Otherwise yellow
        # orientation is irrelevant.
        own_yellow_scoring = (tog is not None and tog.owner == self.alliance) or \
                             (goal.goal_id == CENTER_GOAL_ID and self._we_have_midfield_majority())
        if not own_yellow_scoring:
            return False, None
        if pin.up_half_name != "yellow":
            return True, goal
        return False, None

    def _correct_cup_clear_up(self, goal) -> Optional[bool]:
        """For a goal whose top is a pin, what should cup.clear_up be?

        Returns True  → cup clear-side UP  (BLOCKS pin's UP half visibility)
        Returns False → cup dark-side UP   (ALLOWS pin's UP half to score)
        Returns None  → orientation doesn't matter / can't determine

        Formula: up_vis(pin_below) = not eff_clear_up(cup_above)
          • own-color pin facing up → we WANT that half visible → dark up (False)
          • opp-color pin facing up → we WANT to deny that half  → clear up (True)
        """
        if not goal.stack or not goal.stack[-1][1]:
            return None
        top_pin, _ = goal.stack[-1]
        up = top_pin.up_half_name
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

    # ── Priority 8 — proactive toggle routing ────────────────────────── #

    def _find_useful_toggle(self):
        """Return the best unowned toggle within TOGGLE_ROUTE_RANGE."""
        my_pos = self._pos()
        best = None
        best_sc = -1e9
        for t in self.sim.toggles:
            if t.owner == self.alliance:
                continue
            d = self._dist(my_pos, (float(t.x), float(t.y)))
            if d > self.TOGGLE_ROUTE_RANGE:
                continue
            sc = -d
            # Bonus if there are yellow pins in goals controlled by this toggle
            for g in self.sim.goals:
                if self._toggle_for_goal(g) is t:
                    for obj, is_pin in g.stack:
                        if is_pin and obj.is_yellow:
                            sc += 8.0
            if sc > best_sc:
                best_sc = sc
                best = t
        return best

    # ── Midfield helpers ──────────────────────────────────────────────── #

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
