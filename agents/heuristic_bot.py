"""
agents/heuristic_bot.py
=======================
Rule-based "perfect-play" bot for VEX Override.  v10.0 — "Always Scoring".

What changed in v10.0 vs v9.4
------------------------------
Root cause of the v9.4 0-pts-per-match bug: the bot sat still during the
40-step flip cooldown waiting for a *perfect* orientation that never came,
because the env_wrapper silently rejected most flip attempts.  Bots
oscillated forever between `flip_pin_at_goal` and `await_pin_flip` without
ever scoring.

v10.0 rewrites the priority tree around one principle:

    ► NEVER stop moving and NEVER wait for a flip when in scoring range.
      A 5-pt score is infinitely better than a 0-pt non-score.

Specific changes
----------------
1.  **Score first, optimise later.**  `_try_score_at_range` always returns
    a `score_pin` / `score_cup` action when in range with the right element
    type.  No more `await_pin_flip`.  The bot now scores even with
    suboptimal orientation.

2.  **Pre-flip ONLY during travel.**  Flip actions fire only while we are
    > PRE_FLIP_MIN_DIST inches from the target goal.  The 2-s cooldown
    overlaps with travel time so by the time we arrive we're already
    oriented.

3.  **Orientation-aware goal selection.**  `_score_value_now` scores each
    candidate goal by points unlockable *with the pin/cup in its current
    orientation*.  Bots prefer goals where the orientation is already right
    over goals that would need a flip — both robots in an alliance can then
    score sequentially without any cooldown overhead.

4.  **Bots never sit still.**  Whenever a step would otherwise return zero
    wheels (waiting for cooldown / no immediate task), the bot drifts toward
    its next planned target (often midfield, sometimes an alternate goal).

5.  **Stuck-escape behaviour.**  If linear velocity stays < STUCK_SPEED for
    STUCK_STEPS the bot performs a brief backup + rotate to break free.

6.  **Aggressive intake.**  Any unowned pin/cup within INTAKE_RADIUS gets
    grabbed automatically (intake=True is fired on every approach step).

7.  **Cup-orientation pre-flip.**  Same as pin: flip cup en route, then
    score immediately on arrival regardless of final orientation.

8.  **Chain stacking.**  After scoring a pin, the bot's next pickup is
    biased toward a cup (and vice-versa) so the partial stack at the
    just-scored goal can be completed without anyone returning.

Decision priority (highest first)
---------------------------------
1.  ENDGAME PARK  (last ~3 s, race to midfield)
2.  ENDGAME DUMP  (last ~8 s, score anything legal)
3.  SCORE NOW     (within SCORING_RADIUS of a valid goal — fire immediately)
4.  STUCK ESCAPE  (velocity dead for too long → back up + spin)
5.  PRE-FLIP EN ROUTE (only when target_goal is >PRE_FLIP_MIN_DIST away)
6.  DRIVE TO BEST SCOREABLE GOAL  (orientation-aware target selection)
7.  FETCH MISSING ELEMENT (pin first if empty-handed, else cup)
8.  TOGGLE PROACTIVE   (route to a useful unowned toggle)
9.  DEFAULT — drift toward the midfield, opportunistically toggle on pass.

Opportunistic toggle injection (`_finalize_action`):  any non-scoring step
auto-injects `toggle=True` when within TOGGLE_INTERACTION_RANGE of an
unowned toggle.

Determinism: no RNG, no learning — given identical state, identical action.
The seed in `OverrideEnv` only affects field initialisation (pin/cup
positions, toggle owners), not the bot.
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
    """Deterministic, rule-based agent for a single VEX Override robot.

    Construct once per robot.  Call `get_policy_action()` every control step
    to receive `(cont, disc)` numpy arrays in env_wrapper format.
    """

    # ── Tunables ───────────────────────────────────────────────────────── #
    FLIP_COOLDOWN_STEPS = 40       # matches env_wrapper.COOLDOWN_FLIP

    PICKUP_TIMEOUT     = 50        # ~2.5 s — chase the same pin/cup
    BLACKLIST_DURATION = 80        # ~4 s — then ignore it

    INTAKE_SLOW_MULT  = 1.6        # below 16" → 70% speed
    INTAKE_CRAWL_MULT = 1.05       # below 10.5" → 35% (don't fully stop)

    SPIN_THRESHOLD_DEG = 110.0     # turn-in-place only above this error
    SLOW_RADIUS        = 16.0      # below this distance, scale throttle down

    APPROACH_GOAL_INSET  = 9.0     # target this far from goal centre
    SCORE_BRAKE_RADIUS   = SCORING_RADIUS - 1.0   # 11"; brake just inside the boundary

    # Pre-flip only when more than this far from the target goal.
    # Inside this radius we score immediately, regardless of orientation.
    PRE_FLIP_MIN_DIST = 18.0

    # Stuck-escape
    STUCK_SPEED      = 1.5         # in/s — below this we count as not moving
    STUCK_STEPS      = 30          # 1.5 s of no movement → escape kicks in
    ESCAPE_DURATION  = 16          # 0.8 s of reverse + spin

    # Endgame timing (seconds remaining)
    ENDGAME_DUMP_TIME = 8.0
    ENDGAME_PARK_TIME = 3.0

    # Toggle proactive routing (priority 8)
    TOGGLE_ROUTE_RANGE = 60.0

    # ── Module-level shared state for partner coordination ────────────── #
    # robot_id -> {"pin": pin_id, "cup": id(cup_obj), "goal": goal_id}
    _shared_targets: Dict[str, Dict[str, Optional[int]]] = {}

    def __init__(self, robot_id: str, sim, event_log=None):
        self.robot_id = robot_id
        self.sim = sim
        self.event_log = event_log
        self._robot_cache = None

        # Sticky targets
        self._target_pin_id:  Optional[int] = None
        self._target_cup_id:  Optional[int] = None
        self._target_goal_id: Optional[int] = None
        self._target_pin_steps: int = 0
        self._target_cup_steps: int = 0

        # Blacklists for elements we've given up on
        self._pin_blacklist: set = set()
        self._pin_blacklist_cd: int = 0
        self._cup_blacklist: set = set()
        self._cup_blacklist_cd: int = 0

        # Per-bot flip lockouts (mirror env_wrapper cooldowns so we don't
        # waste actions, but we never WAIT for them when in scoring range).
        self._flip_pin_lockout = 0
        self._flip_cup_lockout = 0

        # Stuck detection
        self._low_speed_steps = 0
        self._escape_steps    = 0
        self._escape_dir      = 1   # +1 = back-spin one way, -1 the other

        # Last picked-up element type — biases next fetch.
        self._last_intake_type: Optional[str] = None  # "pin" / "cup"

        # Diagnostic
        self.last_reason: str = "init"

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
        self._low_speed_steps   = 0
        self._escape_steps      = 0
        self._escape_dir        = 1
        self._last_intake_type  = None
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

    def _speed(self) -> float:
        v = self.robot.body.velocity
        return math.hypot(float(v.x), float(v.y))

    # ── Geometry helpers ──────────────────────────────────────────────── #

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _drive_to(self, target_xy: Tuple[float, float],
                  full_speed: bool = True) -> Tuple[float, float]:
        """Wheel commands to steer toward target_xy.

        - Turn-in-place only when heading error exceeds SPIN_THRESHOLD_DEG.
        - Within SLOW_RADIUS, throttle is scaled by distance.
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
        if dist < self.SLOW_RADIUS:
            base *= max(0.4, dist / self.SLOW_RADIUS)
        bias = max(-0.7, min(0.7, err * 1.4))
        left  = base - bias
        right = base + bias
        return max(-1.0, min(1.0, left)), max(-1.0, min(1.0, right))

    def _approach_goal_point(self, goal) -> Tuple[float, float]:
        """Aim for a point INSET inside the goal radius so we don't crash
        into the goal's physics body and bounce out of range."""
        rx, ry = self._pos()
        dx, dy = goal.x - rx, goal.y - ry
        d = math.hypot(dx, dy)
        if d < self.APPROACH_GOAL_INSET:
            return (goal.x, goal.y)
        sf = (d - self.APPROACH_GOAL_INSET) / d
        return (rx + dx * sf, ry + dy * sf)

    def _toggle_for_goal(self, goal) -> Optional[object]:
        """Toggle that controls this goal's yellow ownership (None for centre)."""
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

    # ── Partner coordination ──────────────────────────────────────────── #

    def _publish_targets(self):
        HeuristicBot._shared_targets[self.robot_id] = {
            "pin":  self._target_pin_id,
            "cup":  self._target_cup_id,
            "goal": self._target_goal_id,
        }

    def _partner_targets(self) -> Dict[str, set]:
        """Targets owned by allied partners (excluding ourselves)."""
        pins, cups, goals = set(), set(), set()
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
        """What this goal needs next:  'pin'  /  'cup'  /  'full'."""
        n = len(goal.stack)
        if n >= 3:
            return "full"
        if n == 0:
            return "pin"
        _, top_is_pin = goal.stack[-1]
        return "cup" if top_is_pin else "pin"

    def _half_pts_for_me(self, half_name: str, goal) -> float:
        """Points we'd earn if this half lands visible on this goal."""
        if half_name == self.alliance:
            return 5.0
        if half_name in ("red", "blue"):
            return -5.0   # opponent half — gives them 5 pts; mildly bad
        if half_name == "yellow":
            if goal.goal_id == CENTER_GOAL_ID:
                # SC5b: yellow ownership decided at match end by midfield
                # majority.  Encourage if we have/can plausibly hold majority.
                if self._we_have_midfield_majority():
                    return 10.0
                return 0.0
            tog = self._toggle_for_goal(goal)
            if tog is None:
                return 0.0
            if tog.owner == self.alliance:
                return 10.0
            if tog.owner == self.opp_alliance:
                return -10.0   # gives them 10 pts
            return 0.0
        return 0.0

    def _score_value_now(self, goal, has_pin: bool, has_cup: bool) -> float:
        """Estimated points if we score the currently-held element NOW
        at `goal`, given its current orientation.  Negative = scoring here
        would actively help the opponent (don't do it).
        """
        needs = self._goal_needs(goal)
        if needs == "full":
            return -1.0
        n = len(goal.stack)

        if needs == "pin" and has_pin:
            pin = self.robot.carrying_pin
            if pin is None:
                return -1.0
            if n == 0:
                # First pin on empty goal: DOWN half hidden by post,
                # only UP half scores.
                return self._half_pts_for_me(pin.up_half_name, goal)
            else:
                # n == 2 (pin + cup), placing a second pin on top.
                # UP half always visible; DOWN half visible if cup is clear-up.
                cup_obj, _ = goal.stack[1]
                cup_clear_up = _eff_clear_up(cup_obj)
                up_pts = self._half_pts_for_me(pin.up_half_name, goal)
                dn_pts = (self._half_pts_for_me(pin.down_half_name, goal)
                          if cup_clear_up else 0.0)
                return up_pts + dn_pts

        if needs == "cup" and has_cup:
            # Cup itself = 0 pts.  But the orientation we drop here decides
            # whether the pin already on the goal contributes points and
            # whether a future pin on top will be visible.
            cup = self.robot.carrying_cup
            if cup is None:
                return -1.0
            if n == 0 or not goal.stack[-1][1]:
                return -1.0
            pin_below, _ = goal.stack[-1]
            # The cup we're about to place sits on pin_below.  When clear-up
            # (white up) the pin below loses its visible UP half — we want
            # that ONLY when the pin's UP half is the OPPONENT's colour.
            up = pin_below.up_half_name
            up_pts = self._half_pts_for_me(up, goal)
            # If pin_below.up_half_name is our colour (+pts) we want cup
            # clear-down (eff_clear_up=False) so the up half stays visible.
            # If it's opp colour (-pts) we want cup clear-up (eff_clear_up=True).
            want_clear_up = (up_pts < 0.0)
            cup_clear_up_now = _eff_clear_up(cup)
            return 3.0 if cup_clear_up_now == want_clear_up else -1.0

        return -1.0

    def _best_scoreable_goal(self, has_pin: bool, has_cup: bool):
        """Best goal where I can legally place what I'm holding NOW.

        Scoring uses orientation-aware expected value MINUS a distance
        penalty.  Partners' claimed goals are deprioritised.
        """
        my_pos = self._pos()
        partners = self._partner_targets()
        best = None
        best_sc = -1e9
        for g in self.sim.goals:
            if not self._valid_goal_for_me(g):
                continue
            needs = self._goal_needs(g)
            if needs == "full":
                continue
            if (needs == "pin" and not has_pin) or (needs == "cup" and not has_cup):
                continue
            v = self._score_value_now(g, has_pin, has_cup)
            # Even goals with v <= 0 are kept as fallbacks; just heavily penalised.
            d = self._dist(my_pos, (g.x, g.y))
            # Bonuses
            if g.alliance == self.alliance:
                v += 3.0    # alliance goals always preferred (only we can score)
            if g.goal_id == CENTER_GOAL_ID:
                v -= 1.5
            if g.goal_id in partners["goals"]:
                v -= 8.0
            sc = v * 4.0 - d
            if sc > best_sc:
                best_sc = sc
                best = g
        if best is not None:
            self._target_goal_id = best.goal_id
        return best

    # ── Immediate scoring ─────────────────────────────────────────────── #

    def _try_score_at_range(self, has_pin: bool, has_cup: bool):
        """Within SCORING_RADIUS of a valid goal?  Score immediately.

        We DO NOT wait for flip cooldown.  A 5-pt score is infinitely
        better than 0-pts spent waiting.  Orientation is pre-handled
        during travel (see priority 5).
        """
        my_pos = self._pos()
        # Iterate goals by best-value first so we pick the best one to score in.
        scored_candidates = []
        for g in self.sim.goals:
            if not self._valid_goal_for_me(g):
                continue
            d = self._dist(my_pos, (g.x, g.y))
            if d > SCORING_RADIUS:
                continue
            needs = self._goal_needs(g)
            if needs == "full":
                continue
            if (needs == "pin" and not has_pin) or (needs == "cup" and not has_cup):
                continue
            v = self._score_value_now(g, has_pin, has_cup)
            scored_candidates.append((v, d, g, needs))
        if not scored_candidates:
            return None
        # Highest value, then closest
        scored_candidates.sort(key=lambda t: (-t[0], t[1]))
        v, d, g, needs = scored_candidates[0]

        # Only refuse to score if the action would be NET NEGATIVE *and*
        # the bot is empty-handed afterward (so we lose nothing by waiting).
        # In practice this only triggers for placing a wrong-orientation cup
        # on an own-alliance pin — we'd rather flip first.
        if v < 0.0 and needs == "cup":
            # Try to flip the cup if we still can; otherwise score anyway.
            cup = self.robot.carrying_cup
            if cup is not None and self._flip_cup_lockout == 0:
                a = self._zero_action()
                a["flip_cup"] = True
                self._flip_cup_lockout = self.FLIP_COOLDOWN_STEPS
                # Drive in a tight circle so we don't drift out of range.
                a["left"], a["right"] = self._tight_circle()
                self.last_reason = f"flip_cup_in_zone:{g.goal_id}"
                return a
            # Cup cooldown still ticking — score anyway, lose a few points.

        a = self._zero_action()
        if needs == "pin":
            a["score_pin"] = True
            self.last_reason = f"score_pin:{g.goal_id}(v={v:.1f})"
        else:
            a["score_cup"] = True
            self.last_reason = f"score_cup:{g.goal_id}(v={v:.1f})"
        self._target_goal_id = g.goal_id
        return a

    def _tight_circle(self) -> Tuple[float, float]:
        """Return wheel commands for a small in-place arc — keeps the
        robot dynamic during cooldown waits."""
        return 0.45, -0.05

    # ── Element pickup helpers ────────────────────────────────────────── #

    def _pin_value(self, pin) -> float:
        """How valuable is this pin?  Negative = avoid."""
        c = pin.color
        own_solid    = (self.alliance == "red"  and c == "red")  or (self.alliance == "blue" and c == "blue")
        opp_solid    = (self.alliance == "red"  and c == "blue") or (self.alliance == "blue" and c == "red")
        own_yellow   = (self.alliance == "red"  and c == "red_yellow")  or (self.alliance == "blue" and c == "blue_yellow")
        opp_yellow   = (self.alliance == "red"  and c == "blue_yellow") or (self.alliance == "blue" and c == "red_yellow")
        full_yellow  = (c == "yellow" or c == "yellow_yellow")
        if own_yellow:   return 1.8
        if own_solid:    return 1.0
        if full_yellow:  return 1.4
        if opp_yellow:   return -1.5
        if opp_solid:    return -2.0
        return 0.1

    def _pick_best_pin(self):
        my_pos = self._pos()
        if self._target_pin_steps >= self.PICKUP_TIMEOUT and self._target_pin_id is not None:
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
            if getattr(p, 'is_nested', False):
                continue
            if p.pin_id in self._pin_blacklist:
                continue
            val = self._pin_value(p)
            if val <= 0:
                continue
            d = self._dist(my_pos, (float(p.body.position.x), float(p.body.position.y)))
            if self._target_pin_id == p.pin_id:
                d *= 0.75
            if p.pin_id in partner_pins:
                d += 50.0
            sc = val * 15.0 - d
            if sc > best_sc:
                best_sc = sc
                best = p
        if best is None:
            # Drop blacklist & partner filter — grab anything valuable
            self._pin_blacklist.clear()
            self._pin_blacklist_cd = 0
            for p in self.sim.pins:
                if p.scored or p.carried_by is not None:
                    continue
                if getattr(p, 'is_nested', False):
                    continue
                val = self._pin_value(p)
                if val <= 0:
                    continue
                d = self._dist(my_pos, (float(p.body.position.x), float(p.body.position.y)))
                sc = val * 15.0 - d
                if sc > best_sc:
                    best_sc = sc
                    best = p
        return best

    def _pick_best_cup(self):
        my_pos = self._pos()
        if self._target_cup_steps >= self.PICKUP_TIMEOUT and self._target_cup_id is not None:
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
            # Field is empty of valuable pins — drift to midfield while we wait.
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
            self.last_reason = "no_pin_available"
            return a
        px, py = float(pin.body.position.x), float(pin.body.position.y)
        d = self._dist(self._pos(), (px, py))

        if self._target_pin_id != pin.pin_id:
            self._target_pin_id = pin.pin_id
            self._target_pin_steps = 0
        self._target_pin_steps += 1

        a["intake"] = True
        if d <= INTAKE_RADIUS * self.INTAKE_CRAWL_MULT:
            a["left"], a["right"] = self._drive_to((px, py), full_speed=False)
            a["left"]  *= 0.4
            a["right"] *= 0.4
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
            a["left"]  *= 0.4
            a["right"] *= 0.4
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
        """Inject toggle=True if we're near an unowned toggle, on any
        non-scoring step.  Free quadrant-control claims while driving."""
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

    # ── Stuck-escape ──────────────────────────────────────────────────── #

    def _update_stuck_state(self):
        spd = self._speed()
        if spd < self.STUCK_SPEED:
            self._low_speed_steps += 1
        else:
            self._low_speed_steps = 0
        # Decrement an active escape
        if self._escape_steps > 0:
            self._escape_steps -= 1

    def _maybe_escape_action(self) -> Optional[dict]:
        if self._escape_steps > 0:
            a = self._zero_action()
            # Reverse + spin one way
            d = self._escape_dir
            a["left"]  = -0.8 * d
            a["right"] = -0.4 * d
            self.last_reason = "escape_stuck"
            return a
        if self._low_speed_steps >= self.STUCK_STEPS:
            # Start an escape burst.  Alternate direction for variety.
            self._escape_dir   = -self._escape_dir
            self._escape_steps = self.ESCAPE_DURATION
            self._low_speed_steps = 0
            return self._maybe_escape_action()
        return None

    # ── Top-level dispatcher ──────────────────────────────────────────── #

    def get_sim_action(self) -> Dict[str, Any]:
        a = self._compute_action()
        self._publish_targets()
        return self._finalize_action(a)

    def _compute_action(self) -> Dict[str, Any]:
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

        # Track stuck-state every step (escape kicks in before priority 1
        # so it overrides even endgame park if we're physically blocked).
        self._update_stuck_state()

        has_pin = self.robot.carrying_pin is not None
        has_cup = self.robot.carrying_cup is not None
        endgame = (self.sim.rules_engine.endgame_active and
                   self.sim.time_remaining is not None)

        # ── 1. ENDGAME PARK ─────────────────────────────────────────── #
        if endgame and self.sim.time_remaining <= self.ENDGAME_PARK_TIME:
            esc = self._maybe_escape_action()
            if esc is not None: return esc
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
            self.last_reason = "endgame_park"
            return a

        # ── 2. ENDGAME DUMP ─────────────────────────────────────────── #
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
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
            self.last_reason = "endgame_park_carry"
            return a

        # ── 3. SCORE NOW — within range with a valid element ──────────── #
        score_now = self._try_score_at_range(has_pin, has_cup)
        if score_now is not None:
            return score_now

        # ── 4. STUCK ESCAPE ────────────────────────────────────────────── #
        esc = self._maybe_escape_action()
        if esc is not None:
            return esc

        # ── 5. PRE-FLIP EN ROUTE ───────────────────────────────────────── #
        target_goal = self._best_scoreable_goal(has_pin, has_cup)
        if target_goal is not None:
            d_to_goal = self._dist(self._pos(), (target_goal.x, target_goal.y))
            if d_to_goal > self.PRE_FLIP_MIN_DIST:
                if has_pin and self._flip_pin_lockout == 0:
                    should = self._should_flip_pin_for_goal(target_goal)
                    if should:
                        a["flip_pin"] = True
                        self._flip_pin_lockout = self.FLIP_COOLDOWN_STEPS
                        a["left"], a["right"] = self._drive_to(
                            self._approach_goal_point(target_goal),
                            full_speed=True)
                        self.last_reason = f"flip_pin_en_route:{target_goal.goal_id}"
                        return a
                if has_cup and self._flip_cup_lockout == 0:
                    should = self._should_flip_cup_for_goal(target_goal)
                    if should:
                        a["flip_cup"] = True
                        self._flip_cup_lockout = self.FLIP_COOLDOWN_STEPS
                        a["left"], a["right"] = self._drive_to(
                            self._approach_goal_point(target_goal),
                            full_speed=True)
                        self.last_reason = f"flip_cup_en_route:{target_goal.goal_id}"
                        return a

        # ── 6. FETCH MISSING ELEMENT ───────────────────────────────────── #
        # Smart full-load: when target_goal needs the OTHER element type,
        # fetch that next so we deposit both in one trip.
        if not has_pin and not has_cup:
            # Bias toward whichever we just used (so the partial stack we
            # just contributed to gets completed): if last intake was a pin,
            # we likely already scored a pin → next we need a cup.
            if self._last_intake_type == "pin":
                cup = self._pick_best_cup()
                if cup is not None:
                    self._last_intake_type = "cup"
                    return self._action_get_cup(a, cup)
            return self._action_get_pin(a)

        if has_pin and not has_cup and target_goal is not None:
            needs = self._goal_needs(target_goal)
            if needs == "pin":
                # We can drop the pin — but only after a cup is at the goal
                # (handled by score_now above).  Meanwhile, grab a cup so we
                # can immediately follow up with a pin+cup placement.
                cup = self._pick_best_cup()
                if cup is not None:
                    return self._action_get_cup(a, cup)

        if not has_pin and has_cup and target_goal is not None:
            needs = self._goal_needs(target_goal)
            if needs == "cup":
                # Goal needs a cup — but to make this useful we also need a pin
                # to follow.  Grab the pin first.  (We deposit cup-then-pin
                # in two visits.)
                pin = self._pick_best_pin()
                if pin is not None:
                    return self._action_get_pin(a)

        if has_pin and not has_cup and target_goal is None:
            cup = self._pick_best_cup()
            if cup is not None:
                return self._action_get_cup(a, cup)
        if not has_pin and has_cup and target_goal is None:
            return self._action_get_pin(a)

        # ── 7. DRIVE TO SCOREABLE GOAL ─────────────────────────────────── #
        if target_goal is not None:
            d = self._dist(self._pos(), (target_goal.x, target_goal.y))
            if d <= self.SCORE_BRAKE_RADIUS:
                # We are inside the brake zone but score_now didn't fire,
                # which means we don't have the right element type.  Free
                # up the slot: do nothing wheel-wise but try a small jog
                # to keep us alive.
                a["left"], a["right"] = self._tight_circle()
                self.last_reason = f"brake_at_goal:{target_goal.goal_id}"
                return a
            a["left"], a["right"] = self._drive_to(
                self._approach_goal_point(target_goal), full_speed=True)
            self.last_reason = f"approach_goal:{target_goal.goal_id}"
            return a

        # ── 8. TOGGLE — proactively route to nearest unowned toggle ────── #
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

        # ── 9. DEFAULT ─────────────────────────────────────────────────── #
        # Drift toward midfield (parking position) — never freeze.
        a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
        self.last_reason = "default_midfield"
        return a

    # ── Endgame dump helpers ──────────────────────────────────────────── #

    def _nearest_dump_goal(self, has_pin: bool, has_cup: bool):
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
        """Return `(cont, disc)` numpy arrays matching env_wrapper format."""
        import numpy as np
        a = self.get_sim_action()
        # Track what type of element we last intaked so the chain-stacking
        # bias in section 6 knows which element to fetch next.
        if a["intake"]:
            # We don't know if the intake will SUCCEED until next step, but
            # if we just picked one up between the last step and now, our
            # carrying state will reflect it on the NEXT call (where the
            # branch picks up).  This heuristic flips on a successful pickup:
            r = self.robot
            if r.carrying_pin is not None and self._last_intake_type != "pin":
                self._last_intake_type = "pin"
            elif r.carrying_cup is not None and self._last_intake_type != "cup":
                self._last_intake_type = "cup"
        cont = np.array([a["left"], a["right"]], dtype=np.float32)
        disc = np.array([
            1.0 if a["intake"]    else 0.0,
            1.0 if a["score_pin"] else 0.0,
            1.0 if a["score_cup"] else 0.0,
            1.0 if a["toggle"]    else 0.0,
            1.0 if a["flip_pin"]  else 0.0,
            1.0 if a["flip_cup"]  else 0.0,
            0.0,
        ], dtype=np.float32)
        return cont, disc

    @staticmethod
    def _zero_action() -> Dict[str, Any]:
        return {"left": 0.0, "right": 0.0,
                "intake": False, "score_pin": False, "score_cup": False,
                "toggle": False, "flip_pin": False, "flip_cup": False}

    # ── Flip-decision helpers ─────────────────────────────────────────── #

    def _should_flip_cup_for_goal(self, goal) -> bool:
        """Should we flip the cup we're carrying before dropping it at `goal`?
        Only fires when goal currently has a pin on top (needs a cup)."""
        cup = self.robot.carrying_cup
        if cup is None or self._goal_needs(goal) != "cup":
            return False
        if not goal.stack or not goal.stack[-1][1]:
            return False
        pin_below, _ = goal.stack[-1]
        # We want cup.eff_clear_up = (the pin's up half is opponent's colour).
        up = pin_below.up_half_name
        up_pts = self._half_pts_for_me(up, goal)
        want_clear_up = (up_pts < 0.0)
        return _eff_clear_up(cup) != want_clear_up

    def _should_flip_pin_for_goal(self, goal) -> bool:
        """Should we flip the pin we're carrying before dropping it at `goal`?
        Only meaningful for two-tone pins (e.g. red_yellow).  Returns True
        when the OTHER orientation would yield strictly more points for us.
        """
        pin = self.robot.carrying_pin
        if pin is None:
            return False
        # Compute pts for current orientation
        n = len(goal.stack)
        if n == 0:
            cur = self._half_pts_for_me(pin.up_half_name, goal)
            alt = self._half_pts_for_me(pin.down_half_name, goal)
            return alt > cur + 1e-3
        if n == 2:
            cup_obj, _ = goal.stack[1]
            cup_clear_up = _eff_clear_up(cup_obj)
            cur = self._half_pts_for_me(pin.up_half_name, goal)
            if cup_clear_up:
                cur += self._half_pts_for_me(pin.down_half_name, goal)
            # Alt is reversed: down_half goes UP, up_half goes DOWN
            alt = self._half_pts_for_me(pin.down_half_name, goal)
            if cup_clear_up:
                alt += self._half_pts_for_me(pin.up_half_name, goal)
            return alt > cur + 1e-3
        return False

    # ── Priority 8 — proactive toggle routing ────────────────────────── #

    def _find_useful_toggle(self):
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
            for g in self.sim.goals:
                if self._toggle_for_goal(g) is t:
                    for obj, is_pin in g.stack:
                        if is_pin and obj.is_yellow:
                            sc += 10.0
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
