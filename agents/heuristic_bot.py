"""
agents/heuristic_bot.py
=======================
Rule-based "perfect-play" bot for VEX Override.  v10.1 — adds obstacle
avoidance, per-robot park targets, and a corrected priority tree to
fix the "stuck pushing into a goal" and "all four robots dog-pile
the center" problems that limited v10.0 to ~80 pts/match.

What changed in v10.1 vs v10.0
-------------------------------
1.  **Obstacle deflection.**  `_drive_to` now routes around any goal or
    other robot sitting on the direct path to its target.  This fixes
    the case where a bot wanted to drive past a goal (e.g. while
    parking or fetching) but its forward force just pushed it into
    the goal's physics body indefinitely.

2.  **Per-robot park targets.**  Endgame parking previously sent every
    robot to (72,72) — directly on top of the centre goal's physics
    body.  All four converged and collided.  v10.1 assigns each robot
    its own park slot at the cardinal points around the centre goal:
    red1=(60,72), red2=(72,60), blue1=(72,84), blue2=(84,72).  All
    inside the midfield diamond, well clear of the centre goal.

3.  **Priority tree corrected.**  v10.0 had a bug where a bot holding
    a pin with a target goal that needed a pin would *fetch a cup
    instead of going to the goal* (chain-stacking heuristic firing
    too aggressively).  v10.1 puts DRIVE-TO-TARGET-GOAL above FETCH,
    so a bot with matching inventory commits to the score.  Fetch
    only fires when there is no target goal.

4.  **Always-score, simplified.**  `_try_score_at_range` no longer
    refuses to score when orientation is suboptimal.  It scores the
    best in-range option every time.  No more bail-out paths leading
    to oscillation.

5.  **Stuck-at-goal fallback.**  When a bot is inside the brake
    radius of a goal but score_now didn't fire (no scoreable option
    here), it actively drives away to the nearest valid fetch
    target instead of doing a tight circle in place.

6.  **Toggle-takeover simplified at the sim level** (`Robot.try_toggle`
    rewrite): pressing toggle next to a non-own toggle sets it to the
    pressing robot's alliance directly, instead of the old 3-state
    cycle red→blue→yellow→… that made red robots accidentally hand
    blue toggles to the opponent.  The bot no longer needs to flip
    twice to claim a contested toggle.

Root principle (unchanged from v10.0):

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

    # v10.1: lowered spin threshold (was 110°).  v10.0's 110° made the bot
    # try to turn-on-the-fly even at near-reversing angles, which produced
    # wide circles instead of sharp corrections.  60° turns-in-place
    # whenever we need a >60° heading correction (sharp pivot).
    SPIN_THRESHOLD_DEG = 60.0
    # v10.1: reduced SLOW_RADIUS (was 16).  Bot held back too early and
    # got rammed by allies coming up from behind, or stalled in element
    # clusters.  8" gives a quick brake right at the goal.
    SLOW_RADIUS        = 8.0

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
    # Toggle defense — only divert to reclaim a stolen toggle if it is
    # within this distance.  Without this cap, an empty-handed bot would
    # cross the entire field to chase a single toggle, taking 30+ seconds
    # and often getting wall-trapped en route.
    TOGGLE_DEFENSE_MAX_DIST = 35.0

    # Force-score timeout: once carrying for this many steps with no
    # positive-value scoring option, accept any in-range goal (even at
    # a small point loss) to free the slot.  Without this, useless pins
    # block the inventory for the entire match — observed: red1 carrying
    # a yellow_yellow pin for 41 seconds because blue owned all the
    # toggles that would make yellow scoring positive.
    CARRY_PATIENCE_STEPS = 80     # 4 s at 20 Hz

    # Per-robot park slot — four cardinal points around the centre goal,
    # all inside the midfield diamond (|dx|+|dy| <= 24) but clear of the
    # centre goal's physics body (radius ~3").  Prevents all four robots
    # from converging on (72,72) and colliding during endgame park.
    PARK_TARGETS: Dict[str, Tuple[float, float]] = {
        "red1":  (60.0, 72.0),   # west of centre
        "red2":  (72.0, 60.0),   # north of centre
        "blue1": (72.0, 84.0),   # south of centre
        "blue2": (84.0, 72.0),   # east of centre
    }

    # Obstacle avoidance — when a goal or other robot sits on the direct
    # path to our target, deflect through a waypoint offset to its side.
    # Tuned: 13" goal-clearance accounts for the goal's drawn radius (10)
    # plus half-robot-extent + buffer.  17" robot clearance is robot
    # width (17) + small safety margin.
    AVOID_GOAL_RADIUS  = 13.0
    AVOID_ROBOT_RADIUS = 17.0

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

        # v10.1: carry-duration counters.  When we've been carrying an
        # element for too long without finding a positive-value scoring
        # option, force-score it at the least-bad available goal to free
        # the slot for a more useful pin/cup.  Without this, "useless"
        # pins (e.g. yellow_yellow when opponent owns all relevant
        # toggles) jam the inventory for the rest of the match.
        self._carry_pin_steps = 0
        self._carry_cup_steps = 0

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
        self._carry_pin_steps   = 0
        self._carry_cup_steps   = 0
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

    def _deflect_for_obstacles(self, target_xy: Tuple[float, float]
                                ) -> Tuple[float, float]:
        """Return a (possibly deflected) target that routes around any goal
        or other robot sitting on the direct path to `target_xy`.

        The deflection picks a waypoint on the side of the obstacle
        opposite to where the obstacle currently lies relative to the
        path — so the bot bends around the obstacle instead of pushing
        into its physics body indefinitely.

        Goals at/near the final destination are NOT treated as obstacles
        (we WANT to drive to them).  Allies are deflected with a slightly
        larger clearance than goals because they move.
        """
        rx, ry = self._pos()
        tx, ty = target_xy
        dx, dy = tx - rx, ty - ry
        total = math.hypot(dx, dy)
        if total < 6.0:
            return target_xy   # too close to bother — let the controller handle it
        ux, uy = dx / total, dy / total
        # Right-perpendicular to path direction (uy, -ux); left-perp = (-uy, ux).
        rp_x, rp_y = uy, -ux

        closest = None   # (proj_along_path, signed_perp, ox, oy, radius)

        # Goals — but skip any goal whose centre is within 12" of the
        # final destination (likely it IS the destination).
        for g in self.sim.goals:
            if math.hypot(g.x - tx, g.y - ty) < 12.0:
                continue
            gdx, gdy = g.x - rx, g.y - ry
            proj = ux * gdx + uy * gdy
            if proj <= 4.0 or proj >= total - 2.0:
                continue   # not on the path between us
            perp = rp_x * gdx + rp_y * gdy
            if abs(perp) > self.AVOID_GOAL_RADIUS:
                continue
            if closest is None or proj < closest[0]:
                closest = (proj, perp, g.x, g.y, self.AVOID_GOAL_RADIUS)

        # Other robots — same logic, larger clearance.
        for r in self.sim.robots:
            if r.robot_id == self.robot_id:
                continue
            ox, oy = float(r.body.position.x), float(r.body.position.y)
            if math.hypot(ox - tx, oy - ty) < 10.0:
                continue   # very close to my destination — don't deflect
            gdx, gdy = ox - rx, oy - ry
            proj = ux * gdx + uy * gdy
            if proj <= 4.0 or proj >= total - 2.0:
                continue
            perp = rp_x * gdx + rp_y * gdy
            if abs(perp) > self.AVOID_ROBOT_RADIUS:
                continue
            if closest is None or proj < closest[0]:
                closest = (proj, perp, ox, oy, self.AVOID_ROBOT_RADIUS)

        if closest is None:
            return target_xy

        proj, perp, ox, oy, rad = closest
        # Choose the side of the obstacle to pass on:
        #   perp > 0 → obstacle is to my path-right → pass on its left
        #   perp < 0 → obstacle is to my path-left  → pass on its right
        #   perp ≈ 0 → straight ahead; pick right arbitrarily for determinism
        side_sign = 1.0 if perp >= 0.0 else -1.0
        offset = rad + 5.0

        def _wp(side):
            return (ox - side * rp_x * offset, oy - side * rp_y * offset)

        def _in_field(p):
            return 5.0 < p[0] < 139.0 and 5.0 < p[1] < 139.0

        primary = _wp(side_sign)
        if _in_field(primary):
            return primary
        # Primary waypoint is outside the field — try the opposite side.
        alt = _wp(-side_sign)
        if _in_field(alt):
            return alt
        # Both deflections leave the field — clamp the primary inside it
        # so the bot drives toward a reachable point and keeps moving.
        return (max(5.0, min(139.0, primary[0])),
                max(5.0, min(139.0, primary[1])))

    def _drive_to(self, target_xy: Tuple[float, float],
                  full_speed: bool = True,
                  avoid_obstacles: bool = True) -> Tuple[float, float]:
        """Wheel commands to steer toward target_xy.

        - If `avoid_obstacles` (default), routes around goals/robots in
          the direct path via `_deflect_for_obstacles`.
        - Turn-in-place only when heading error exceeds SPIN_THRESHOLD_DEG.
        - Within SLOW_RADIUS, throttle is scaled by distance.
        """
        if avoid_obstacles:
            target_xy = self._deflect_for_obstacles(target_xy)
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

    def _park_target(self) -> Tuple[float, float]:
        """Per-robot park slot for endgame — avoids 4-robot pileup on centre."""
        return self.PARK_TARGETS.get(self.robot_id, MIDFIELD_CENTER)

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

    # ── Score-value functions ───────────────────────────────────────── #
    # v10.1: split into PIN_SCORE_VALUE and CUP_SCORE_VALUE so the bot can
    # compare current-orientation vs. flipped-orientation independently.

    def _pin_score_value(self, goal, up_half: str, down_half: str) -> float:
        """Points scored placing a pin in `goal` with given UP / DOWN halves.
        Accounts for goal post hiding DOWN of bottom-most pin and cup-below
        visibility of stack-position-2 pins."""
        n = len(goal.stack)
        if n == 0:
            # First pin: DOWN hidden by post, only UP visible.
            return self._half_pts_for_me(up_half, goal)
        # n == 2: pin on top of cup.
        cup_obj, _ = goal.stack[1]
        cup_clear_up = _eff_clear_up(cup_obj)
        up_pts = self._half_pts_for_me(up_half, goal)
        down_pts = (self._half_pts_for_me(down_half, goal)
                    if cup_clear_up else 0.0)
        return up_pts + down_pts

    def _cup_score_value(self, goal, cup_clear_up_at_place: bool) -> float:
        """Score delta to OUR alliance from placing a cup with given
        orientation on top of the pin already in `goal`.

        - cup_clear_up=True  → pin_below's UP half is HIDDEN by the cup.
                              Cup also "opens" the top for a future pin's
                              DOWN to be visible (slight setup bonus).
        - cup_clear_up=False → pin_below's UP half stays VISIBLE.

        Returns the change in our alliance's score from placing this cup.
        Cup itself = 0 pts; the only effect is visibility of pin_below
        and a small setup-bonus heuristic for stack potential.
        """
        if not goal.stack or not goal.stack[-1][1]:
            return -1e9   # invalid placement
        pin_below, _ = goal.stack[-1]
        pts_pin_up = self._half_pts_for_me(pin_below.up_half_name, goal)
        # Placing the cup CHANGES whether pin_below.UP is visible:
        #   cup_clear_up=True  → UP becomes hidden → delta = -pts_pin_up
        #   cup_clear_up=False → UP stays visible → delta = 0
        if cup_clear_up_at_place:
            immediate = -pts_pin_up
            setup    =  1.5    # future pin above's DOWN will be visible
        else:
            immediate =  0.0
            setup    =  0.5    # safer but locks in any future pin DOWN as hidden
        return immediate + setup

    def _best_score_at_goal(self, goal, has_pin: bool, has_cup: bool
                            ) -> Tuple[float, bool, Optional[str]]:
        """For `goal`, return (best_value, needs_flip, element_type).

        - Considers BOTH current orientation and flipped orientation of
          the element we're carrying.  Picks the better; returns
          needs_flip=True if the flipped state is strictly better.
        - element_type is "pin" or "cup" or None (cannot score here).
        """
        needs = self._goal_needs(goal)
        if needs == "full":
            return (-1e9, False, None)
        if needs == "pin" and has_pin:
            pin = self.robot.carrying_pin
            if pin is None:
                return (-1e9, False, None)
            v_now  = self._pin_score_value(goal, pin.up_half_name, pin.down_half_name)
            v_flip = self._pin_score_value(goal, pin.down_half_name, pin.up_half_name)
            if v_flip > v_now + 0.5:
                return (v_flip, True, "pin")
            return (v_now, False, "pin")
        if needs == "cup" and has_cup:
            cup = self.robot.carrying_cup
            if cup is None:
                return (-1e9, False, None)
            cur_clear = _eff_clear_up(cup)
            v_now  = self._cup_score_value(goal, cur_clear)
            v_flip = self._cup_score_value(goal, not cur_clear)
            if v_flip > v_now + 0.5:
                return (v_flip, True, "cup")
            return (v_now, False, "cup")
        return (-1e9, False, None)

    def _score_value_now(self, goal, has_pin: bool, has_cup: bool) -> float:
        """Compatibility shim: returns just the best value at `goal`."""
        v, _, _ = self._best_score_at_goal(goal, has_pin, has_cup)
        return v

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
        """Within SCORING_RADIUS of a valid goal?  Score the best option.

        v10.1 — flip-at-range: if the BEST orientation requires a flip
        and the flip cooldown is available, fire the flip (and brake
        in place for the single step it takes to resolve), then score
        next step with the now-correct orientation.  If the flip
        cooldown is active, score with the current (worse) orientation
        rather than waiting forever.

        Refuses to score only when even the best in-range option is
        strongly negative (would meaningfully help the opponent).
        Then `target_goal` logic in priority 6 picks a different
        destination instead of looping at this goal.
        """
        my_pos = self._pos()
        candidates = []
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
            v, needs_flip, ntype = self._best_score_at_goal(g, has_pin, has_cup)
            if v <= -1e8:
                continue
            candidates.append((v, d, g, ntype, needs_flip))
        if not candidates:
            return None

        # Best value first, ties broken by proximity.
        candidates.sort(key=lambda t: (-t[0], t[1]))
        v, d, g, ntype, needs_flip = candidates[0]

        # Refuse to score if the best option is meaningfully negative —
        # UNLESS we've been carrying this element so long it's blocking
        # the inventory.  Then accept the loss and free the slot.
        if v < -1.5:
            carry_too_long = (
                (ntype == "pin" and self._carry_pin_steps > self.CARRY_PATIENCE_STEPS) or
                (ntype == "cup" and self._carry_cup_steps > self.CARRY_PATIENCE_STEPS)
            )
            if not carry_too_long:
                return None
            # Fall through and score at the least-bad option.

        # Flip-at-range: if a flip would improve value and we can fire,
        # do so.  Brake for the one step the flip takes to resolve.
        if needs_flip:
            if ntype == "pin" and self._flip_pin_lockout == 0:
                a = self._zero_action()
                a["flip_pin"] = True
                self._flip_pin_lockout = self.FLIP_COOLDOWN_STEPS
                # Brake so we stay inside SCORING_RADIUS for next-step score.
                a["left"], a["right"] = 0.0, 0.0
                self.last_reason = f"flip_pin_at_range:{g.goal_id}"
                self._target_goal_id = g.goal_id
                return a
            if ntype == "cup" and self._flip_cup_lockout == 0:
                a = self._zero_action()
                a["flip_cup"] = True
                self._flip_cup_lockout = self.FLIP_COOLDOWN_STEPS
                a["left"], a["right"] = 0.0, 0.0
                self.last_reason = f"flip_cup_at_range:{g.goal_id}"
                self._target_goal_id = g.goal_id
                return a
            # Lockout active — fall through and score with current
            # (worse) orientation rather than stalling.

        a = self._zero_action()
        if ntype == "pin":
            a["score_pin"] = True
            self.last_reason = f"score_pin:{g.goal_id}(v={v:.1f})"
        else:
            a["score_cup"] = True
            self.last_reason = f"score_cup:{g.goal_id}(v={v:.1f})"
        self._target_goal_id = g.goal_id
        return a

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

    # Wall threshold (in inches from the field edge).  Anything inside
    # this margin is hard for a 17×15 robot to physically reach because
    # the robot's half-extent (~7.5") plus collision buffer (~2") is
    # larger than the offset.  Wall-pinned objects also tend to clump
    # against each other after being knocked free of their starting
    # cups, making intake unreliable.
    WALL_MARGIN_IN  = 9.0
    WALL_PENALTY_IN = 100.0   # add this many "virtual inches" to wall items

    def _is_wall_pinned(self, x: float, y: float) -> bool:
        return (x < self.WALL_MARGIN_IN or x > 144.0 - self.WALL_MARGIN_IN or
                y < self.WALL_MARGIN_IN or y > 144.0 - self.WALL_MARGIN_IN)

    def _pick_best_pin(self):
        """Pick a pin to fetch.

        v10.1 commit-and-blacklist:
          • If our current target is still valid, stick to it.
          • On timeout, blacklist the target PERSISTENTLY (no global
            clearing every BLACKLIST_DURATION steps — that was the
            "stuck on pin 1010 for 70 seconds" bug in v10.0).
          • Strictly exclude partner-targeted pins (race avoidance).
          • Heavy wall-pinned-object penalty so we prefer reachable
            field pins over wall-clustered loading-zone pins.
          • Fallback: if every valuable pin is blacklisted or
            partner-owned, clear the blacklist and try again rather
            than freezing.
        """
        my_pos = self._pos()

        # Stick to current target if still valid.
        if self._target_pin_id is not None:
            if self._target_pin_steps < self.PICKUP_TIMEOUT:
                for p in self.sim.pins:
                    if (p.pin_id == self._target_pin_id and
                        not p.scored and p.carried_by is None and
                        not getattr(p, 'is_nested', False)):
                        return p
            # Timeout or pin gone — blacklist it.  Stays blacklisted
            # for the remainder of the episode unless explicitly reset.
            self._pin_blacklist.add(self._target_pin_id)
            self._target_pin_id = None
            self._target_pin_steps = 0

        partner_pins = self._partner_targets()["pins"]
        best, best_sc = self._scan_pins(my_pos, partner_pins,
                                         skip_blacklist=True,
                                         skip_partner=True)
        if best is None:
            # Fallback: allow partner pins (race for them).
            best, best_sc = self._scan_pins(my_pos, partner_pins,
                                             skip_blacklist=True,
                                             skip_partner=False)
        if best is None:
            # Last resort: clear blacklist and try again.
            self._pin_blacklist.clear()
            best, best_sc = self._scan_pins(my_pos, partner_pins,
                                             skip_blacklist=False,
                                             skip_partner=False)
        return best

    def _scan_pins(self, my_pos, partner_pins,
                   skip_blacklist: bool, skip_partner: bool):
        best, best_sc = None, -1e9
        for p in self.sim.pins:
            if p.scored or p.carried_by is not None:
                continue
            if getattr(p, 'is_nested', False):
                continue
            if skip_blacklist and p.pin_id in self._pin_blacklist:
                continue
            if skip_partner and p.pin_id in partner_pins:
                continue
            val = self._pin_value(p)
            if val <= 0:
                continue
            px, py = float(p.body.position.x), float(p.body.position.y)
            d = self._dist(my_pos, (px, py))
            if self._is_wall_pinned(px, py):
                d += self.WALL_PENALTY_IN
            sc = val * 20.0 - d
            if sc > best_sc:
                best_sc, best = sc, p
        return best, best_sc

    def _pick_best_cup(self):
        """Pick a cup to fetch (same commit-and-blacklist pattern as pins)."""
        my_pos = self._pos()

        if self._target_cup_id is not None:
            if self._target_cup_steps < self.PICKUP_TIMEOUT:
                for c in self.sim.cups:
                    if (id(c) == self._target_cup_id and
                        not c.scored and c.carried_by is None):
                        return c
            self._cup_blacklist.add(self._target_cup_id)
            self._target_cup_id = None
            self._target_cup_steps = 0

        partner_cups = self._partner_targets()["cups"]
        best, best_d = self._scan_cups(my_pos, partner_cups,
                                        skip_blacklist=True,
                                        skip_partner=True)
        if best is None:
            best, best_d = self._scan_cups(my_pos, partner_cups,
                                            skip_blacklist=True,
                                            skip_partner=False)
        if best is None:
            self._cup_blacklist.clear()
            best, best_d = self._scan_cups(my_pos, partner_cups,
                                            skip_blacklist=False,
                                            skip_partner=False)
        return best

    def _scan_cups(self, my_pos, partner_cups,
                   skip_blacklist: bool, skip_partner: bool):
        best, best_d = None, float("inf")
        for c in self.sim.cups:
            if c.scored or c.carried_by is not None:
                continue
            cid = id(c)
            if skip_blacklist and cid in self._cup_blacklist:
                continue
            if skip_partner and cid in partner_cups:
                continue
            cx, cy = float(c.body.position.x), float(c.body.position.y)
            d = self._dist(my_pos, (cx, cy))
            if self._is_wall_pinned(cx, cy):
                d += self.WALL_PENALTY_IN
            if d < best_d:
                best_d, best = d, c
        return best, best_d

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
        """Inject toggle=True and intake=True when convenient on any
        non-scoring step.  Captures opportunistic gains while driving:

          • Toggles: claim any unowned toggle within TOGGLE_INTERACTION_RANGE.
          • Intake:  grab any unowned pin/cup within INTAKE_RADIUS as we
                     pass.  This is the v10.1 fix for "bot gets stuck in
                     element clusters" — instead of pushing into pins/cups
                     on the way to the goal, we GRAB them, removing them
                     from our path and gaining free inventory for future
                     scoring.
        """
        scoring = a["score_pin"] or a["score_cup"]
        flipping = a["flip_pin"] or a["flip_cup"]
        my_pos = self._pos()

        # ── Opportunistic toggle ─────────────────────────────────────
        if not scoring and not flipping and not a["toggle"]:
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

        # ── Opportunistic intake ─────────────────────────────────────
        # Fire intake when a VALUABLE element is within the sim's
        # effective intake range (intake radius from body OR front face).
        # Skip opp pins and skip wall-pinned objects.
        if not scoring and not a["intake"]:
            r = self.robot
            need_pin = r.carrying_pin is None
            need_cup = r.carrying_cup is None
            if need_pin or need_cup:
                # Sim's check: object within INTAKE_RADIUS of either body
                # center or front face (~half_len+2 inches forward).
                # Use 18" body-distance as conservative effective range.
                grab_thresh = INTAKE_RADIUS * 1.8
                if need_pin:
                    for p in self.sim.pins:
                        if p.scored or p.carried_by is not None: continue
                        if getattr(p, 'is_nested', False): continue
                        if self._pin_value(p) <= 0:
                            continue
                        px, py = float(p.body.position.x), float(p.body.position.y)
                        if self._is_wall_pinned(px, py):
                            continue
                        d = self._dist(my_pos, (px, py))
                        if d <= grab_thresh:
                            a["intake"] = True
                            self.last_reason += f"+intake_pin:{p.pin_id}"
                            break
                if not a["intake"] and need_cup:
                    for c in self.sim.cups:
                        if c.scored or c.carried_by is not None: continue
                        cx, cy = float(c.body.position.x), float(c.body.position.y)
                        if self._is_wall_pinned(cx, cy):
                            continue
                        d = self._dist(my_pos, (cx, cy))
                        if d <= grab_thresh:
                            a["intake"] = True
                            self.last_reason += f"+intake_cup:{id(c)}"
                            break
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
            # Blacklist current targets — if we're stuck, our chosen pin/cup
            # is probably unreachable from here.  Without this, the bot just
            # re-targets the same element after the escape burst and gets
            # stuck again (observed: blue1 lost 55 seconds bouncing between
            # pin 14 in the corner cluster and the goal 8 obstacle).
            if self._target_pin_id is not None:
                self._pin_blacklist.add(self._target_pin_id)
                self._target_pin_id = None
                self._target_pin_steps = 0
            if self._target_cup_id is not None:
                self._cup_blacklist.add(self._target_cup_id)
                self._target_cup_id = None
                self._target_cup_steps = 0
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

        # Track how long we've been carrying each element type.  Used by
        # _try_score_at_range to FORCE a score after CARRY_PATIENCE_STEPS
        # rather than carry an unscorable pin/cup forever.
        self._carry_pin_steps = self._carry_pin_steps + 1 if has_pin else 0
        self._carry_cup_steps = self._carry_cup_steps + 1 if has_cup else 0
        endgame = (self.sim.rules_engine.endgame_active and
                   self.sim.time_remaining is not None)

        # ── 1. ENDGAME PARK ─────────────────────────────────────────── #
        if endgame and self.sim.time_remaining <= self.ENDGAME_PARK_TIME:
            esc = self._maybe_escape_action()
            if esc is not None: return esc
            a["left"], a["right"] = self._drive_to(self._park_target(), full_speed=True)
            self.last_reason = "endgame_park"
            return a

        # ── 2. ENDGAME DUMP ─────────────────────────────────────────── #
        if endgame and self.sim.time_remaining <= self.ENDGAME_DUMP_TIME and (has_pin or has_cup):
            dump = self._try_dump_at_range(has_pin, has_cup)
            if dump is not None:
                return dump
            target = self._nearest_dump_goal(has_pin, has_cup)
            if target is not None:
                self._target_goal_id = target.goal_id
                a["left"], a["right"] = self._drive_to(
                    self._approach_goal_point(target), full_speed=True)
                self.last_reason = f"endgame_dump_to:{target.goal_id}"
                return a
            a["left"], a["right"] = self._drive_to(self._park_target(), full_speed=True)
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

        # Compute target_goal once for use across the remaining priorities.
        target_goal = self._best_scoreable_goal(has_pin, has_cup)

        # ── 5. PRE-FLIP EN ROUTE ───────────────────────────────────────── #
        # Only flip while still driving (more than PRE_FLIP_MIN_DIST from goal).
        # The 2-s cooldown then overlaps with travel time.
        if target_goal is not None:
            d_to_goal = self._dist(self._pos(), (target_goal.x, target_goal.y))
            if d_to_goal > self.PRE_FLIP_MIN_DIST:
                if has_pin and self._flip_pin_lockout == 0:
                    if self._should_flip_pin_for_goal(target_goal):
                        a["flip_pin"] = True
                        self._flip_pin_lockout = self.FLIP_COOLDOWN_STEPS
                        a["left"], a["right"] = self._drive_to(
                            self._approach_goal_point(target_goal), full_speed=True)
                        self.last_reason = f"flip_pin_en_route:{target_goal.goal_id}"
                        return a
                if has_cup and self._flip_cup_lockout == 0:
                    if self._should_flip_cup_for_goal(target_goal):
                        a["flip_cup"] = True
                        self._flip_cup_lockout = self.FLIP_COOLDOWN_STEPS
                        a["left"], a["right"] = self._drive_to(
                            self._approach_goal_point(target_goal), full_speed=True)
                        self.last_reason = f"flip_cup_en_route:{target_goal.goal_id}"
                        return a

        # ── 6. DRIVE TO TARGET GOAL ────────────────────────────────────── #
        # If we have a scoreable goal lined up, commit to it.  This MUST
        # come before fetching — otherwise a bot holding a pin with a
        # goal-needs-pin target would divert to fetch a cup (chain-stack
        # heuristic firing too aggressively) instead of completing the
        # score it could make right now.
        if target_goal is not None:
            d = self._dist(self._pos(), (target_goal.x, target_goal.y))
            self._target_goal_id = target_goal.goal_id
            if d <= self.SCORE_BRAKE_RADIUS:
                # Inside brake radius but score_now didn't fire.  That's
                # because the goal is unservable with our exact inventory
                # right now (e.g. we have pin+cup but goal is full, picked
                # only as fallback).  Drive AWAY toward the next useful
                # task instead of circling in place.
                a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
                self.last_reason = f"abandon_unservable_goal:{target_goal.goal_id}"
                return a
            a["left"], a["right"] = self._drive_to(
                self._approach_goal_point(target_goal), full_speed=True)
            self.last_reason = f"approach_goal:{target_goal.goal_id}"
            return a

        # ── 7. FETCH MISSING ELEMENT ───────────────────────────────────── #
        # No scoreable goal for our current inventory — fetch what's needed.
        # The bias toward the OTHER element type (chain stacking) was moved
        # here from priority 6 in v10.0, so it only fires when we genuinely
        # have nothing to do at any goal right now.
        self._target_goal_id = None

        if not has_pin and not has_cup:
            # Both slots empty.  Decide pin vs cup based on alliance/neutral
            # goal demand: more goals need pin → fetch pin, more need cup →
            # fetch cup.  Fallback: pin (higher solo value).
            need_pin = need_cup = 0
            for g in self.sim.goals:
                if g.alliance not in ("neutral", self.alliance):
                    continue
                nd = self._goal_needs(g)
                if nd == "pin": need_pin += 1
                elif nd == "cup": need_cup += 1
            # Toggle defense BEFORE fetch only if a stolen toggle is
            # genuinely on the way (closer than the nearest valuable
            # pin/cup).  This keeps defense useful but stops the
            # "blue1 crossed the whole field to chase toggle 1 and
            # got stuck against the wall for 40 seconds" failure mode.
            stolen = self._find_stolen_toggle()
            if stolen is not None:
                d_tog = self._dist(self._pos(),
                                   (float(stolen.x), float(stolen.y)))
                if d_tog < self.TOGGLE_DEFENSE_MAX_DIST:
                    nearest_elem_d = self._nearest_valuable_element_dist()
                    if d_tog < nearest_elem_d:
                        tx, ty = float(stolen.x), float(stolen.y)
                        if d_tog <= TOGGLE_INTERACTION_RANGE:
                            a["toggle"] = True
                            self.last_reason = f"reclaim_toggle:{stolen.toggle_id}"
                        else:
                            a["left"], a["right"] = self._drive_to(
                                (tx, ty), full_speed=True)
                            self.last_reason = f"approach_stolen_toggle:{stolen.toggle_id}"
                        return a
            if need_cup > need_pin and self._pick_best_cup() is not None:
                cup = self._pick_best_cup()
                return self._action_get_cup(a, cup)
            return self._action_get_pin(a)

        if has_pin and not has_cup:
            # We have a pin but no goal needs one.  Fetch a cup so we can
            # take advantage of any pin-on-top goal (or set one up).
            cup = self._pick_best_cup()
            if cup is not None:
                return self._action_get_cup(a, cup)
            # No cups available — keep the pin and roam.
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
            self.last_reason = "carry_pin_idle"
            return a

        if not has_pin and has_cup:
            # We have a cup but no goal needs one (i.e. all goals are empty
            # or full).  Fetch a pin to enable the cup somewhere.
            pin = self._pick_best_pin()
            if pin is not None:
                return self._action_get_pin(a)
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
            self.last_reason = "carry_cup_idle"
            return a

        # Both slots full and no scoreable goal — every goal must be full
        # or unmatched.  Roam toward midfield; we'll be in scoring range
        # eventually as opponents change goal state.
        a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
        self.last_reason = "full_inventory_idle"
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
        """Should we flip the cup before dropping it at `goal`?  Uses
        the same delta-aware value model as _try_score_at_range."""
        cup = self.robot.carrying_cup
        if cup is None or self._goal_needs(goal) != "cup":
            return False
        if not goal.stack or not goal.stack[-1][1]:
            return False
        cur_clear = _eff_clear_up(cup)
        v_now  = self._cup_score_value(goal, cur_clear)
        v_flip = self._cup_score_value(goal, not cur_clear)
        return v_flip > v_now + 0.5

    def _should_flip_pin_for_goal(self, goal) -> bool:
        """Should we flip the pin before dropping it at `goal`?  Uses
        the same value model as _try_score_at_range so en-route
        pre-flips and at-range flips agree."""
        pin = self.robot.carrying_pin
        if pin is None or self._goal_needs(goal) != "pin":
            return False
        v_now  = self._pin_score_value(goal, pin.up_half_name, pin.down_half_name)
        v_flip = self._pin_score_value(goal, pin.down_half_name, pin.up_half_name)
        return v_flip > v_now + 0.5

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

    # ── Stolen toggle reclamation ────────────────────────────────────── #

    def _nearest_valuable_element_dist(self) -> float:
        """Distance to the nearest valuable pin/cup we could fetch.
        Returns +inf if nothing is fetchable.  Used by the toggle-defense
        gate to skip detours when a fetch target is closer."""
        my_pos = self._pos()
        best = float("inf")
        for p in self.sim.pins:
            if p.scored or p.carried_by is not None:
                continue
            if getattr(p, 'is_nested', False):
                continue
            if p.pin_id in self._pin_blacklist:
                continue
            if self._pin_value(p) <= 0:
                continue
            px, py = float(p.body.position.x), float(p.body.position.y)
            if self._is_wall_pinned(px, py):
                continue
            d = self._dist(my_pos, (px, py))
            if d < best:
                best = d
        for c in self.sim.cups:
            if c.scored or c.carried_by is not None:
                continue
            if id(c) in self._cup_blacklist:
                continue
            cx, cy = float(c.body.position.x), float(c.body.position.y)
            if self._is_wall_pinned(cx, cy):
                continue
            d = self._dist(my_pos, (cx, cy))
            if d < best:
                best = d
        return best

    def _find_stolen_toggle(self):
        """Find the nearest opp-owned (or yellow) toggle that controls a
        goal we can score in (alliance or neutral, not the centre).
        Returns the toggle object or None.

        Used by priority 6.5 to actively defend toggle ownership when
        the bot has nothing else to do.  Without active reclamation,
        a single opp pass past our toggle can permanently shift the
        +10 pts/yellow scoring balance for the rest of the match.
        """
        my_pos = self._pos()
        best = None
        best_d = float("inf")
        for t in self.sim.toggles:
            if t.owner == self.alliance:
                continue
            # Check this toggle controls at least one of our scoreable
            # (alliance or neutral, non-centre) goals.
            controls_mine = False
            for g in self.sim.goals:
                if g.alliance not in ("neutral", self.alliance):
                    continue
                if g.goal_id == CENTER_GOAL_ID:
                    continue
                if self._toggle_for_goal(g) is t:
                    controls_mine = True
                    break
            if not controls_mine:
                continue
            d = self._dist(my_pos, (float(t.x), float(t.y)))
            if d < best_d:
                best_d = d
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
