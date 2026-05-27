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

    # v15: pickup timeout lengthened from 50→100 steps (2.5s→5s).  Prior
    # value caused pin-target oscillation when the chosen pin was hard to
    # reach (e.g., behind a goal); the timeout fired before the bot could
    # close the distance, blacklisting the pin and swapping to a
    # similarly-hard alternate, which timed out and swapped back, etc.
    # 5 s is enough to traverse the full field at ~30 in/s carrying.
    PICKUP_TIMEOUT     = 100       # ~5 s — chase the same pin/cup
    BLACKLIST_DURATION = 160       # ~8 s — then ignore it

    INTAKE_SLOW_MULT  = 1.6        # below 16" → 70% speed
    INTAKE_CRAWL_MULT = 1.05       # below 10.5" → 35% (don't fully stop)

    SPIN_THRESHOLD_DEG = 60.0
    # v15: reduced SLOW_RADIUS 8 → 5.  Bot was wasting 1-2 s/cycle
    # braking too early on approach; tighter brake gives a faster
    # commit-to-scoring distance.  Risk: occasional overshoot is
    # absorbed by SCORE_BRAKE_RADIUS gate inside try_score_at_range.
    SLOW_RADIUS        = 8.0

    APPROACH_GOAL_INSET  = 9.0     # target this far from goal centre
    SCORE_BRAKE_RADIUS   = SCORING_RADIUS - 1.0   # 11"; brake just inside the boundary

    # Pre-flip only when more than this far from the target goal.
    # Inside this radius we score immediately, regardless of orientation.
    PRE_FLIP_MIN_DIST = 18.0

    # Stuck-escape — escape fires fast as a safety net.  The primary
    # avoidance path is now _deflect_for_obstacles' near-contact
    # sidestep (handles pressed-against-goal cases the old projection
    # check missed), so escape only needs to catch wall corners and
    # other edge cases.
    STUCK_SPEED      = 1.5         # in/s — below this we count as not moving
    STUCK_STEPS      = 18          # 0.9 s of no movement → escape kicks in
    ESCAPE_DURATION  = 16          # 0.8 s of reverse + spin

    # Near-contact sidestep: if a non-destination goal/robot sits within
    # this distance of us AND in our forward hemisphere, route around it
    # via a perpendicular waypoint immediately — independent of the
    # projection-based deflection.  Fixes the "pressed against the goal
    # for several seconds" issue where the obstacle's projection along
    # the path was too small to trigger the old deflection code.
    NEAR_CONTACT_GOAL_DIST  = 15.0
    NEAR_CONTACT_ROBOT_DIST = 19.0

    # Endgame timing (seconds remaining)
    ENDGAME_DUMP_TIME = 8.0
    ENDGAME_PARK_TIME = 6.0   # tuned from sweep (was 3.0; tested 6/8/10 — 6 won)

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

        # Match-load follow-through.  After firing match_load, the loaded
        # cup/pin spawns at a known corner position.  Track it for ~1.5 s
        # so we tight-cycle: drive straight to it and intake before any
        # other consideration (avoids leaving the spawned element behind
        # due to the wall-pin penalty in normal _pick_best_* scans).
        self._just_loaded_steps: int = 0
        self._last_load_spawn: Optional[Tuple[float, float]] = None
        # Index 0/1 picking which of the two loading-zone corners we own
        # this trip (so partners don't crowd the same corner).
        self._target_load_zone: Optional[int] = None
        # Once we commit to a match-load trip, stay committed for ~3 s so
        # noisy _should_match_load flickers don't make us drift halfway to
        # the corner and back repeatedly (observed: 30 s wasted otherwise).
        self._match_load_commit: int = 0
        # Back-off after a failed trip (committed but couldn't reach the
        # zone in time) — don't pound the same path repeatedly.
        self._ml_cooldown_steps: int = 0
        # Track which loading-zone indices have failed this match (via
        # STUCK_ESCAPE).  _pick_loading_zone avoids these on retries.
        self._ml_failed_zones: set = set()
        # Track high-value toggles that proved unreachable (stuck during
        # pursuit).  _find_high_value_toggle_flip skips these.
        self._hv_toggle_blacklist: set = set()
        # Step counter for the current hv-toggle pursuit so we can
        # timeout if the bot makes no progress.
        self._hv_toggle_target_id: Optional[int] = None
        self._hv_toggle_steps: int = 0
        # v14 — Post-score toggle claim.  Set right after a score action
        # that exposes yellow halves on a non-friendly toggle's goal,
        # so the bot's immediate next priority is to claim that toggle.
        self._post_score_toggle_id: Optional[int] = None
        self._post_score_toggle_steps: int = 0

        # v15 — Per-goal progress watchdog.  Tracks how close we've ever
        # gotten to the current target goal.  If we haven't made any new
        # progress (haven't pushed the min distance down) for too many
        # steps, drop the commit-lock and let _best_scoreable_goal re-pick.
        # Without this, the strong commit-lock can hold us against an
        # obstacle for 10+ seconds because the value math still says the
        # locked goal is best.
        self._goal_min_dist_seen: float = float("inf")
        self._goal_no_progress_steps: int = 0
        # Goals abandoned this trip due to no-progress.  Filtered out of
        # _best_scoreable_goal until we score (then list resets).
        self._goal_avoid: set = set()

        # Diagnostic
        self.last_reason: str = "init"

        HeuristicBot._shared_targets[self.robot_id] = {
            "pin": None, "cup": None, "goal": None, "load_zone": None,
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
        self._just_loaded_steps = 0
        self._last_load_spawn   = None
        self._target_load_zone  = None
        self._match_load_commit = 0
        self._ml_cooldown_steps = 0
        self._ml_failed_zones   = set()
        self._hv_toggle_blacklist = set()
        self._hv_toggle_target_id = None
        self._hv_toggle_steps     = 0
        self._post_score_toggle_id    = None
        self._post_score_toggle_steps = 0
        self._goal_min_dist_seen      = float("inf")
        self._goal_no_progress_steps  = 0
        self._goal_avoid              = set()
        self.last_reason        = "init"
        HeuristicBot._shared_targets[self.robot_id] = {
            "pin": None, "cup": None, "goal": None, "load_zone": None,
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

    # ── VEX VRC autonomous wedge helpers ──────────────────────────────── #
    # The four diagonal tape lines divide the field into four wedges, one
    # per starting robot position.  During autonomous, each robot is
    # PHYSICALLY restricted to its own wedge by the auton walls in the
    # simulator.  The bot uses these helpers to avoid driving into the
    # walls (it'd just get stuck and waste auton time).
    _WEDGE_BY_ROBOT = {
        "red1":  "west",   # x small, y around middle
        "red2":  "north",  # x around middle, y small
        "blue1": "south",  # x around middle, y large
        "blue2": "east",   # x large, y around middle
    }

    def _is_in_my_auton_wedge(self, x: float, y: float) -> bool:
        """True if (x,y) is in this robot's autonomous wedge.
        Wedges defined by the two diagonals y=x and y=144-x:
          west:  y > x AND y < 144-x
          north: y < x AND y < 144-x
          east:  y < x AND y > 144-x
          south: y > x AND y > 144-x
        Per VRC Override rules, the midfield diamond is SEALED OFF during
        auton — robots may not cross any white tape line, including the
        diamond perimeter.  No shared zone: each robot is confined to its
        own wedge.
        """
        # Reject anything inside the midfield diamond — it's walled off
        # during auton and not part of any wedge.
        if abs(x - 72.0) + abs(y - 72.0) <= 24.0:
            return False
        diag1 = y < x        # north/east half of the y=x line
        diag2 = y < 144.0 - x  # north/west half of the y=144-x line
        wedge = self._WEDGE_BY_ROBOT.get(self.robot_id, "west")
        if wedge == "west":
            return (not diag1) and diag2
        if wedge == "north":
            return diag1 and diag2
        if wedge == "east":
            return diag1 and (not diag2)
        # south
        return (not diag1) and (not diag2)

    # Per-robot in-wedge anchor used as the fallback when a target lies
    # outside our auton wedge.  Each anchor is inside the robot's wedge,
    # away from the diagonal tape walls and the midfield diamond, near
    # an in-wedge goal — so falling back here keeps the bot productive
    # instead of pushing it into a wall.
    _AUTON_WEDGE_ANCHOR: Dict[str, Tuple[float, float]] = {
        "red1":  (24.0,  72.0),  # west wedge, between R-High and SW
        "red2":  (72.0,  24.0),  # north wedge, between NW and B-Low
        "blue1": (72.0, 120.0),  # south wedge, between R-Low and SE
        "blue2": (120.0, 72.0),  # east wedge, between B-High and NE
    }

    def _clamp_target_to_auton_wedge(self, target_xy: Tuple[float, float]
                                     ) -> Tuple[float, float]:
        """During autonomous, redirect targets outside our wedge to an
        in-wedge anchor (the midfield diamond is walled off, so the old
        MIDFIELD_CENTER fallback would just push the bot into a wall).
        After auton ends, returns the target unchanged.
        """
        if not self._is_in_autonomous():
            return target_xy
        tx, ty = target_xy
        if self._is_in_my_auton_wedge(tx, ty):
            return target_xy
        return self._AUTON_WEDGE_ANCHOR.get(self.robot_id, self._pos())

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

        # ── Near-contact sidestep ─────────────────────────────────────
        # Any non-destination obstacle within close range AND in our
        # forward hemisphere gets routed around immediately, regardless
        # of how the path-projection math classifies it.  This catches
        # the "robot pressed against a goal trying to drive past it"
        # case that the projection-based check below misses (proj is
        # tiny when the obstacle is right next to us, so the old code
        # would skip it and the bot would push into the goal forever).
        near = None
        near_d = float("inf")
        for g in self.sim.goals:
            if math.hypot(g.x - tx, g.y - ty) < 12.0:
                continue   # this goal IS (probably) the destination
            gdx, gdy = g.x - rx, g.y - ry
            d = math.hypot(gdx, gdy)
            if d > self.NEAR_CONTACT_GOAL_DIST:
                continue
            # Forward hemisphere: skip obstacles behind us.
            if ux * gdx + uy * gdy < -1.0:
                continue
            if d < near_d:
                near_d = d
                near = (g.x, g.y, gdx, gdy, self.AVOID_GOAL_RADIUS)
        for r in self.sim.robots:
            if r.robot_id == self.robot_id:
                continue
            ox, oy = float(r.body.position.x), float(r.body.position.y)
            if math.hypot(ox - tx, oy - ty) < 10.0:
                continue
            gdx, gdy = ox - rx, oy - ry
            d = math.hypot(gdx, gdy)
            if d > self.NEAR_CONTACT_ROBOT_DIST:
                continue
            if ux * gdx + uy * gdy < -1.0:
                continue
            if d < near_d:
                near_d = d
                near = (ox, oy, gdx, gdy, self.AVOID_ROBOT_RADIUS)
        if near is not None:
            ox, oy, gdx, gdy, rad = near
            # perp > 0 → obstacle is to my right of path → pass on its left
            perp = rp_x * gdx + rp_y * gdy
            # Pass on the FAR side of the obstacle (opposite to where it
            # currently is relative to the path) so we steer AWAY from it.
            side_sign = -1.0 if perp >= 0.0 else 1.0
            offset = rad + 7.0
            wp = (ox + side_sign * rp_x * offset,
                  oy + side_sign * rp_y * offset)
            # Clamp to robot-reachable area (9" margin = half-robot + buffer).
            return (max(9.0, min(135.0, wp[0])),
                    max(9.0, min(135.0, wp[1])))

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
            # 9" margin accounts for robot half-width (8.5") so deflection
            # waypoints are physically reachable without jamming into the wall.
            return 9.0 < p[0] < 135.0 and 9.0 < p[1] < 135.0

        primary = _wp(side_sign)
        if _in_field(primary):
            return primary
        # Primary waypoint is outside the robot-reachable area — try the
        # opposite side of the obstacle.
        alt = _wp(-side_sign)
        if _in_field(alt):
            return alt
        # Both deflections are out of bounds — clamp to robot-reachable area
        # so the bot drives toward a reachable point and keeps moving.
        return (max(9.0, min(135.0, primary[0])),
                max(9.0, min(135.0, primary[1])))

    def _drive_to(self, target_xy: Tuple[float, float],
                  full_speed: bool = True,
                  avoid_obstacles: bool = True) -> Tuple[float, float]:
        """Wheel commands to steer toward target_xy.

        - During autonomous, targets outside our wedge are clamped to the
          midfield diamond center — the auton walls would block us anyway.
        - If `avoid_obstacles` (default), routes around goals/robots in
          the direct path via `_deflect_for_obstacles`.
        - Turn-in-place only when heading error exceeds SPIN_THRESHOLD_DEG.
        - Within SLOW_RADIUS, throttle is scaled by distance.
        """
        # VRC autonomous: clamp to wedge so we don't waste cycles pushing
        # against the diagonal tape walls.
        target_xy = self._clamp_target_to_auton_wedge(target_xy)
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
        # Skills: no opposing alliance — every goal is scoreable.
        if getattr(self.sim, "skills_mode", False):
            return True
        return goal.alliance in ("neutral", self.alliance)

    # ── Partner coordination ──────────────────────────────────────────── #

    def _publish_targets(self):
        HeuristicBot._shared_targets[self.robot_id] = {
            "pin":       self._target_pin_id,
            "cup":       self._target_cup_id,
            "goal":      self._target_goal_id,
            "load_zone": getattr(self, "_target_load_zone", None),
        }

    def _partner_targets(self) -> Dict[str, set]:
        """Targets owned by allied partners (excluding ourselves)."""
        pins, cups, goals, load_zones = set(), set(), set(), set()
        alliance_of: Dict[str, str] = {r.robot_id: r.alliance for r in self.sim.robots}
        for rid, tgt in HeuristicBot._shared_targets.items():
            if rid == self.robot_id:
                continue
            if alliance_of.get(rid) != self.alliance:
                continue
            if tgt.get("pin")        is not None: pins.add(tgt["pin"])
            if tgt.get("cup")        is not None: cups.add(tgt["cup"])
            if tgt.get("goal")       is not None: goals.add(tgt["goal"])
            if tgt.get("load_zone")  is not None: load_zones.add(tgt["load_zone"])
        return {"pins": pins, "cups": cups, "goals": goals,
                "load_zones": load_zones}

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
        skills = getattr(self.sim, "skills_mode", False)
        if half_name == self.alliance:
            return 5.0
        if half_name in ("red", "blue"):
            # In skills, the other-color half ALSO scores for the player
            # (RulesEngine folds blue into red).  In a match it's an
            # opponent gift.
            return 5.0 if skills else -5.0
        if half_name == "yellow":
            if goal.goal_id == CENTER_GOAL_ID:
                # SC5b: yellow ownership decided at match end by midfield
                # majority.  In skills the only robots in the midfield
                # are the player's, so majority is always the player.
                if skills or self._we_have_midfield_majority():
                    return 10.0
                return 0.0
            tog = self._toggle_for_goal(goal)
            if tog is None:
                return 0.0
            if tog.owner == self.alliance:
                return 10.0
            if skills and tog.owner in ("red", "blue"):
                # In skills, both red and blue toggle states credit the
                # player (folded).  Only "yellow" (neutral) scores 0.
                return 10.0
            if tog.owner == self.opp_alliance:
                return -10.0   # gives them 10 pts (match only)
            return 0.0
        return 0.0

    # ── Score-value functions ───────────────────────────────────────── #
    # v10.1: split into PIN_SCORE_VALUE and CUP_SCORE_VALUE so the bot can
    # compare current-orientation vs. flipped-orientation independently.

    def _pin_score_value(self, goal, up_half: str, down_half: str) -> float:
        """Points scored placing a pin in `goal` with given UP / DOWN halves.
        Accounts for goal post hiding DOWN of bottom-most pin and cup-below
        visibility of stack-position-2 pins.

        v15 — STACK BONUS: the official scoring grants +3 per properly
        nested pin (one with a cup directly below it).  This is a flat
        bonus on top of half-visibility points and was missing from the
        bot's valuation, causing it to value the 2nd pin in a stack
        identically to a fresh 1st pin elsewhere — leaving easy +3 pts
        on the table dozens of times per match.
        """
        n = len(goal.stack)
        if n == 0:
            # First pin: DOWN hidden by post, only UP visible.
            return self._half_pts_for_me(up_half, goal)
        # n == 2: pin on top of cup — visible UP, plus DOWN if cup is
        # clear-side-up.
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

    # v12 — A4 effective-speed estimate for time-based valuation.  Tuned
    # empirically against measured robot movement: ~25 in/s straight-line
    # under load, plus a per-radian spin penalty of ~0.4 s.
    EFFECTIVE_SPEED_IN_PER_S = 25.0
    SPIN_PENALTY_S_PER_RAD   = 0.4

    def _travel_time_to(self, target_xy: Tuple[float, float]) -> float:
        """Estimate seconds to physically reach target_xy from current pose.

        Accounts for both translation distance AND the heading correction
        the bot will have to spin through first.  Used by A3/A4 to compute
        "points per second" rather than naive value-minus-distance.
        """
        rx, ry = self._pos()
        tx, ty = target_xy
        dx, dy = tx - rx, ty - ry
        dist = math.hypot(dx, dy)
        translate_s = dist / self.EFFECTIVE_SPEED_IN_PER_S
        # Heading error penalty.  At 0 err: no penalty.  At π/2: ~0.6s spin.
        desired = math.atan2(dy, dx) if dist > 1e-3 else 0.0
        err = abs(_norm_angle(desired - float(self.robot.body.angle)))
        spin_s = err * self.SPIN_PENALTY_S_PER_RAD
        # Reverse-drive cap: at err > 135°, we'd reverse, so cap effective err.
        if err > math.radians(135.0):
            spin_s = (math.pi - err) * self.SPIN_PENALTY_S_PER_RAD
        return max(0.1, translate_s + spin_s)

    # v13 — points-equivalent cost of 1 second of travel.  Tuned so that
    # a 10-pt score 4s away (cost: 8 pts → net 2) loses to a 5-pt score
    # 0.5s away (cost: 1 pt → net 4).  But a 10-pt score 0.5s away
    # (net 9) beats a 5-pt score 0.1s away (net 4.8).
    TIME_COST_PTS_PER_S = 2.0

    def _best_scoreable_goal(self, has_pin: bool, has_cup: bool):
        """Best goal where I can legally place what I'm holding NOW.

        v13 — Linear time-cost valuation (replaces divisive pps which
        produced extreme values for very close goals and made the bot
        prefer marginal 1-pt scores 0.1s away over 10-pt scores 1s away):
            score = v - TIME_COST_PTS_PER_S * travel_seconds

        Plus a sticky bonus when within 20" of the current target goal
        (G2 commit-lock) so the bot doesn't flip-flop mid-approach.
        """
        my_pos = self._pos()
        partners = self._partner_targets()
        in_auton = self._is_in_autonomous()

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
            # Skip goals we've already given up on this trip (no-progress).
            if g.goal_id in self._goal_avoid:
                continue
            # Auton wedge restriction: white tape walls prevent crossing.
            # During auton, only score on goals inside our wedge.
            if in_auton and not self._is_in_my_auton_wedge(g.x, g.y):
                continue
            v = self._score_value_now(g, has_pin, has_cup)
            # Strategic value bonuses.
            if g.alliance == self.alliance:
                v += 3.0    # alliance goals always preferred (only we can score)
            if g.goal_id == CENTER_GOAL_ID:
                v -= 1.5
            if g.goal_id in partners["goals"]:
                v -= 8.0
            # Linear time-cost: subtract pts-per-second × travel-time.
            travel_s = self._travel_time_to((g.x, g.y))
            sc = v - self.TIME_COST_PTS_PER_S * travel_s
            d = self._dist(my_pos, (g.x, g.y))

            # v15 — 1-step LOOKAHEAD: prefer goals that set up a fast
            # follow-up score.  After scoring here, hands go empty, so
            # the next move is to fetch.  If a loading zone or a partial
            # stack (pin+cup ready for stack-bonus pin) sits close to
            # this goal, the bot can chain it without a long traverse.
            chain = 0.0
            for lz in self._loading_zone_targets():
                if (lz[0] - g.x) ** 2 + (lz[1] - g.y) ** 2 < 35.0 * 35.0:
                    chain += 2.0
                    break
            for g2 in self.sim.goals:
                if g2 is g:
                    continue
                if len(g2.stack) == 2 and not g2.stack[-1][1]:
                    if (g2.x - g.x) ** 2 + (g2.y - g.y) ** 2 < 35.0 * 35.0:
                        chain += 2.0
                        break
            sc += min(chain, 4.0)

            # v15 STRONG commit-lock: once we've picked a goal, stay with it.
            # Previously +1..+4 within 20" was not enough to defeat
            # mid-approach travel-cost noise — the bot oscillated between
            # two near-equal-value goals every ~1 s, wasting 5-10 s/match.
            # New: +5 baseline whenever it's still our target, plus a ramp
            # up to +6 over the final 30" of approach.  Total +11 at 0",
            # which is hard to dislodge without a clearly-better rival.
            # (Empirically: +8 base over-committed and missed the center
            # goal at endgame; +5 base + ramp is the sweet spot.)
            if g.goal_id == self._target_goal_id:
                sc += 5.0
                if d < 30.0:
                    sc += 6.0 * (30.0 - d) / 30.0
            if sc > best_sc:
                best_sc = sc
                best = g
        if best is not None:
            # Reset watchdog if we switched targets (the new target may
            # legitimately be farther; old min-dist is irrelevant).
            if best.goal_id != self._target_goal_id:
                self._goal_min_dist_seen = float("inf")
                self._goal_no_progress_steps = 0
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
        if v < -4.0:   # EXP10: was -1.5 — accept slightly worse scores rather than holding
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
        # v15 — Successful score: clear the per-trip avoid-set and reset
        # the progress watchdog so a previously-blocked goal is reconsidered
        # on the next pickup.  Without this we'd permanently avoid any goal
        # we ever got stuck near.
        self._goal_avoid.clear()
        self._goal_min_dist_seen = float("inf")
        self._goal_no_progress_steps = 0
        # v14 — Post-score toggle claim.  If this score exposes yellow halves
        # on a goal whose controlling toggle is NOT ours, the bot's IMMEDIATE
        # next action should be to claim that toggle.  Each visible yellow
        # half is worth +10 pts when the toggle flips to our alliance.
        tog = self._toggle_for_goal(g)
        if tog is not None and tog.owner != self.alliance:
            # Check if scoring this pin/cup will expose yellow halves.
            yellow_exposed = self._score_will_expose_yellow(g, ntype)
            if yellow_exposed:
                self._post_score_toggle_id = tog.toggle_id
                self._post_score_toggle_steps = 60   # 3-second pursuit window
        return a

    def _score_will_expose_yellow(self, goal, ntype: str) -> bool:
        """Check if placing a pin/cup on `goal` would leave a yellow half
        visible on the top of the resulting stack.
          - Pin score: places carried pin on top → its visible UP half is
            the orientation-current UP (or flipped).  If yellow → True.
          - Cup score: places carried cup on top of an existing pin →
            pin's UP half may become hidden (cup not clear_up) or stay
            visible (cup clear_up).  If pin UP was yellow and stays
            visible → True.
        """
        if ntype == "pin":
            pin = self.robot.carrying_pin
            if pin is None:
                return False
            return (pin.up_half_name == "yellow" or
                    pin.down_half_name == "yellow")
        # cup score: cup goes on top of pin in goal.stack[-1]
        if not goal.stack or not goal.stack[-1][1]:
            return False
        pin_below, _ = goal.stack[-1]
        cup = self.robot.carrying_cup
        if cup is None:
            return False
        # After cup placement, pin_below's UP half visibility depends on
        # cup's clear_on_top (effectively flipped).
        flipped = getattr(cup, 'flipped', False)
        eff_clear_up = (not cup.clear_on_top) if flipped else cup.clear_on_top
        if eff_clear_up:
            # Cup is transparent → pin UP stays visible.
            return pin_below.up_half_name == "yellow"
        # Cup hides pin UP → no exposed yellow from pin.  Cup's DOWN
        # half becomes visible (the half touching the pin) — but cup
        # halves aren't yellow in this game (cups are uniform color).
        return False

    # ── Element pickup helpers ────────────────────────────────────────── #

    def _pin_value(self, pin) -> float:
        """How valuable is this pin?  Negative = avoid.

        v12 — A1 live toggle check:
          Yellow pin halves only score 10 pts when our alliance owns the
          controlling toggle of the goal we'll score at.  If we DON'T own
          any toggle whose region contains a scoreable goal, a yellow pin
          is worth nothing (or negative if opp owns).  Recompute live so
          the bot doesn't carry a yellow pin around looking for a slot
          that doesn't exist.
        """
        c = pin.color
        skills = getattr(self.sim, "skills_mode", False)
        if skills:
            # No opposing alliance — every pin half is the player's.
            own_solid    = c in ("red", "blue")
            opp_solid    = False
            own_yellow   = c in ("red_yellow", "blue_yellow")
            opp_yellow   = False
        else:
            own_solid    = (self.alliance == "red"  and c == "red")  or (self.alliance == "blue" and c == "blue")
            opp_solid    = (self.alliance == "red"  and c == "blue") or (self.alliance == "blue" and c == "red")
            own_yellow   = (self.alliance == "red"  and c == "red_yellow")  or (self.alliance == "blue" and c == "blue_yellow")
            opp_yellow   = (self.alliance == "red"  and c == "blue_yellow") or (self.alliance == "blue" and c == "red_yellow")
        full_yellow  = (c == "yellow" or c == "yellow_yellow")
        # red_blue / blue_red have one alliance half each.  We can orient
        # ours UP so the visible-up half scores for us — equivalent to an
        # own-solid pin for our scoring purposes (but the opponent half
        # is still on the pin, so any DOWN-visible scoring gives them 5).
        mixed_alliance = (c == "red_blue" or c == "blue_red")
        if own_solid:      return 1.0
        if mixed_alliance: return 0.8   # slightly less than own_solid — opp half can be exposed
        if opp_solid:   return -2.0
        # Pin has at least one yellow half — multiplier depends on live toggle
        # ownership of any goal where we can plausibly score it.
        if own_yellow or full_yellow or opp_yellow:
            yellow_mult = self._live_yellow_multiplier()
            base = 1.8 if own_yellow else (1.4 if full_yellow else -1.5)
            # full_yellow value is multiplied by toggle control (2x both halves
            # vs 1x one half).  own_yellow has 1 yellow half.  opp_yellow same.
            if yellow_mult >= 1.0:
                # Friendly toggle control somewhere — yellow is fully worth it.
                return base * yellow_mult
            elif yellow_mult <= -0.5:
                # Opp controls; yellow becomes a liability.
                return base * yellow_mult
            else:
                # Neutral — yellow scores 0 pts, only color base matters.
                # full_yellow=1.4 → 0.4 (cup setup); own_yellow=1.8 → 0.5; opp_yellow=-1.5 → -0.5
                return base * 0.3
        return 0.1

    def _live_yellow_multiplier(self) -> float:
        """Returns a multiplier for yellow-half pin value based on live toggle
        ownership across the field:
          +1.0 → we control at least one toggle whose region has scoreable goals
                 (yellow scores 10 pts at those goals)
           0.0 → all toggles in regions with scoreable goals are NEUTRAL
                (yellow scores 0)
          -1.0 → opp controls all relevant toggles (yellow scores -10 → opp +10)
        Used by _pin_value to keep yellow pin valuations honest.
        """
        skills = getattr(self.sim, "skills_mode", False)
        ours = opps = neut = 0
        for g in self.sim.goals:
            if g.goal_id == CENTER_GOAL_ID:
                continue  # SC5b — decided at end, not via toggles
            needs = self._goal_needs(g)
            if needs == "full":
                continue
            tog = self._toggle_for_goal(g)
            if tog is None:
                continue
            if skills:
                # No opposing alliance — any red/blue toggle credits player.
                if tog.owner in ("red", "blue"):
                    ours += 1
                else:
                    neut += 1
            else:
                if tog.owner == self.alliance:
                    ours += 1
                elif tog.owner == self.opp_alliance:
                    opps += 1
                else:
                    neut += 1
        if ours > 0:
            return 1.0   # at least one ours-controlled scoreable goal
        if opps > 0 and neut == 0:
            return -1.0  # ALL scoreable goals are opp — yellow is bad
        return 0.0       # mixed or neutral — yellow worth its base only

    # Wall threshold (in inches from the field edge).  Anything inside
    # this margin is hard for a 17×15 robot to physically reach because
    # the robot's half-extent (~7.5") plus collision buffer (~2") is
    # larger than the offset.  Wall-pinned objects also tend to clump
    # against each other after being knocked free of their starting
    # cups, making intake unreliable.
    WALL_MARGIN_IN  = 9.0
    WALL_PENALTY_IN = 100.0

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
        in_auton = self._is_in_autonomous()
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
            # Auton wedge restriction: white tape walls prevent us from
            # reaching any pin outside our wedge.  Skip rather than waste
            # the trip pathing toward the anchor fallback.
            if in_auton and not self._is_in_my_auton_wedge(px, py):
                continue
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
        """Pick the closest valuable cup.  v12 — B1 nested-pin bonus:
        cups containing a YELLOW_YELLOW pin are dramatically more valuable
        because intaking the cup ALSO extracts the nested pin (see
        Robot.try_intake nested-pin extraction).  We approximate this by
        subtracting "virtual inches" from the cup's effective distance,
        so a cup with a yellow pin inside is preferred even if 30+ inches
        farther than an empty cup.  Multiplier scales with live yellow
        ownership — if opp owns the toggles, the nested yellow is a
        liability, not a bonus.
        """
        yellow_mult = self._live_yellow_multiplier()
        in_auton = self._is_in_autonomous()
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
            if in_auton and not self._is_in_my_auton_wedge(cx, cy):
                continue
            d = self._dist(my_pos, (cx, cy))
            if self._is_wall_pinned(cx, cy):
                d += self.WALL_PENALTY_IN
            # Nested pin bonus / penalty.
            nested = getattr(c, 'contains_pin', None)
            if nested is not None:
                up = getattr(nested, 'up_half_name', None)
                dn = getattr(nested, 'down_half_name', None)
                if up == "yellow" and dn == "yellow":
                    # Full yellow_yellow — worth +20 pts if we control toggle.
                    # Subtract 40 virtual inches if friendly, add 30 if opp.
                    if yellow_mult >= 1.0:
                        d -= 40.0
                    elif yellow_mult <= -0.5:
                        d += 30.0
                elif up == "yellow" or dn == "yellow":
                    # Half-yellow nested — modest bonus / penalty.
                    if yellow_mult >= 1.0:
                        d -= 15.0
                    elif yellow_mult <= -0.5:
                        d += 10.0
            if d < best_d:
                best_d, best = d, c
        return best, best_d

    def _action_get_pin(self, a: dict) -> dict:
        pin = self._pick_best_pin()
        if pin is None:
            # Field is empty of valuable pins.  Don't drift — go press an
            # un-claimed alliance toggle so we keep producing value.  If all
            # toggles are claimed, position near the highest-value goal so we
            # are first to react when an element appears.
            tog = self._find_useful_toggle()
            if tog is not None:
                tx, ty = float(tog.x), float(tog.y)
                if self._dist(self._pos(), (tx, ty)) <= TOGGLE_INTERACTION_RANGE:
                    a["toggle"] = True
                    self.last_reason = f"claim_toggle:{tog.toggle_id}"
                else:
                    a["left"], a["right"] = self._drive_to((tx, ty), full_speed=True)
                    self.last_reason = f"go_claim_toggle:{tog.toggle_id}"
                return a
            # No toggles left — pre-position near the nearest alliance goal
            # that still needs a pin, so we score immediately when one appears.
            best_g = None
            best_d = float("inf")
            my_pos = self._pos()
            for g in self.sim.goals:
                if not self._valid_goal_for_me(g):
                    continue
                if self._goal_needs(g) != "pin":
                    continue
                d = self._dist(my_pos, (g.x, g.y))
                if d < best_d:
                    best_d, best_g = d, g
            if best_g is not None:
                a["left"], a["right"] = self._drive_to(
                    self._approach_goal_point(best_g), full_speed=True)
                self.last_reason = f"prepos_at_goal:{best_g.goal_id}"
            else:
                a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
                self.last_reason = "roam_midfield"
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
        #
        # v14 — Allow opportunistic intake DURING match-load trips too.
        # If we already have a cup and pick up a pin, both slots fill and
        # _should_match_load returns False (good — score before another
        # ML trip).  If we have a pin and pick up a cup, same.  Either
        # way the bot is more productive than ignoring elements on its
        # path to the loading zone.  But ONLY allow intake of the slot
        # we have OPEN — never pick up the type we already hold (causes
        # an infinite "intake → drop → re-intake" loop with the sim's
        # possession check).
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
            # Abort any in-flight match-load trip.  Cooldown depends on
            # how close we got: if stuck near the zone (within 25") it was
            # probably just slow squeezing into the wall corner — short
            # cooldown so we retry quickly.  If stuck far from zone, the
            # path is genuinely blocked — moderate cooldown to try other
            # tasks.  v15: was 30 s for the far-stuck case which banned
            # match-load for half the skills match after one bump.
            if self._match_load_commit > 0:
                my = self._pos()
                zones = self._loading_zone_targets()
                min_d = min(self._dist(my, z) for z in zones)
                self._match_load_commit = 0
                self._target_load_zone  = None
                if min_d <= 25.0:
                    self._ml_cooldown_steps = 160   # 8 s — just retry approach
                else:
                    self._ml_cooldown_steps = 600   # 30 s — path blocked
            # Also blacklist any in-pursuit high-value toggle.
            if self._hv_toggle_target_id is not None:
                self._hv_toggle_blacklist.add(self._hv_toggle_target_id)
                self._hv_toggle_target_id = None
                self._hv_toggle_steps = 0
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

        # Decrement match-load counters unconditionally every frame so they
        # tick correctly even when other priorities return early.
        if self._just_loaded_steps > 0:
            self._just_loaded_steps -= 1
        if self._match_load_commit > 0:
            self._match_load_commit -= 1
            if self._match_load_commit == 0:
                self._ml_cooldown_steps = 120   # ~6 s back-off on expiry
                self._target_load_zone  = None
        if self._ml_cooldown_steps > 0:
            self._ml_cooldown_steps -= 1

        has_pin = self.robot.carrying_pin is not None
        has_cup = self.robot.carrying_cup is not None

        # Track how long we've been carrying each element type.  Used by
        # _try_score_at_range to FORCE a score after CARRY_PATIENCE_STEPS
        # rather than carry an unscorable pin/cup forever.
        self._carry_pin_steps = self._carry_pin_steps + 1 if has_pin else 0
        self._carry_cup_steps = self._carry_cup_steps + 1 if has_cup else 0
        endgame = (self.sim.rules_engine.endgame_active and
                   self.sim.time_remaining is not None)

        # ── 1. ENDGAME PARK (F1 — park vs score breakpoint) ──────────── #
        # Parking gives a FIXED +8 pts.  Scoring gives variable pts.
        # We park unconditionally inside ENDGAME_PARK_TIME — too risky to
        # score then park in <3s.  BUT in the 3-6s window we compare:
        # if there's a >10pt scoring opportunity within ~3s travel, take
        # it; otherwise start parking early.
        if endgame and self.sim.time_remaining <= self.ENDGAME_PARK_TIME:
            esc = self._maybe_escape_action()
            if esc is not None: return esc
            a["left"], a["right"] = self._drive_to(self._park_target(), full_speed=True)
            self.last_reason = "endgame_park"
            return a
        if (endgame and (has_pin or has_cup) and
                self.sim.time_remaining is not None and
                self.sim.time_remaining <= 6.0 and
                self.sim.time_remaining > self.ENDGAME_PARK_TIME):
            # Inside 3-6s window — compare best score vs park value.
            # If best score is much better than +8 park AND reachable in
            # less than (time_remaining - park_buffer) seconds, score.
            # Otherwise, start parking now so we make it in time.
            best_g = self._best_scoreable_goal(has_pin, has_cup)
            if best_g is not None:
                v = self._score_value_now(best_g, has_pin, has_cup)
                travel_to_score = self._travel_time_to((best_g.x, best_g.y))
                travel_to_park  = self._travel_time_to(self._park_target())
                # We need: travel_to_score + 0.5s (score) + travel_to_park
                # to fit inside time_remaining.  Add 0.5s safety margin.
                total_needed = travel_to_score + 0.5 + travel_to_park + 0.5
                if v >= 10.0 and total_needed < self.sim.time_remaining:
                    # Worth scoring then parking.
                    a["left"], a["right"] = self._drive_to(
                        self._approach_goal_point(best_g), full_speed=True)
                    self.last_reason = f"endgame_score_then_park:{best_g.goal_id}"
                    self._target_goal_id = best_g.goal_id
                    return a
            # Park immediately — value of +8 dominates any small score.
            a["left"], a["right"] = self._drive_to(self._park_target(), full_speed=True)
            self.last_reason = "endgame_park_early"
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

        # ── 3.5. POST-SCORE TOGGLE CLAIM (v14) ─────────────────────────── #
        # Just scored a pin/cup that exposed yellow halves on a non-friendly
        # toggle's goal — IMMEDIATELY claim that toggle so visible yellow
        # halves are credited to us (+10 pts each).
        # Decrement window each step; expire after 60 steps (3s).
        if self._post_score_toggle_steps > 0:
            self._post_score_toggle_steps -= 1
            if self._post_score_toggle_steps == 0:
                self._post_score_toggle_id = None
        if self._post_score_toggle_id is not None:
            tog = next((t for t in self.sim.toggles
                        if t.toggle_id == self._post_score_toggle_id), None)
            if tog is None or tog.owner == self.alliance:
                # Mission accomplished or toggle no longer exists.
                self._post_score_toggle_id = None
                self._post_score_toggle_steps = 0
            else:
                tx, ty = float(tog.x), float(tog.y)
                d_tog = self._dist(self._pos(), (tx, ty))
                if d_tog <= TOGGLE_INTERACTION_RANGE and getattr(self, '_env_cd_toggle', 0) == 0:
                    a["toggle"] = True
                    self.last_reason = f"post_score_toggle:{tog.toggle_id}"
                    return a
                a["left"], a["right"] = self._drive_to((tx, ty), full_speed=True)
                self.last_reason = f"go_post_score_toggle:{tog.toggle_id}"
                return a

        # ── 4. STUCK ESCAPE ────────────────────────────────────────────── #
        esc = self._maybe_escape_action()
        if esc is not None:
            return esc

        # ── 4.4. HIGH-VALUE TOGGLE FLIP (E1) — only as opportunistic ──── #
        # Only fire if we're already WITHIN TOGGLE_INTERACTION_RANGE of an
        # opp-controlled toggle with high swing.  No "drive to it" mode —
        # the opportunistic _finalize_action toggle injection already
        # handles claim-on-passing, and standalone pursuit caused blue2
        # to spend 70s driving to an unreachable toggle.
        if not has_pin and not has_cup:
            hv_tog = self._find_high_value_toggle_flip()
            if hv_tog is not None:
                tx, ty = float(hv_tog.x), float(hv_tog.y)
                d_tog = self._dist(self._pos(), (tx, ty))
                if d_tog <= TOGGLE_INTERACTION_RANGE:
                    a["toggle"] = True
                    self.last_reason = f"hv_flip_toggle:{hv_tog.toggle_id}"
                    return a

        # ── 4.5. MATCH LOAD (red1/blue1 only) ──────────────────────────── #
        # This must come BEFORE goal-approach (priority 6) so a committed
        # match-load trip is not preempted by opportunistic scoring near
        # alliance-side goals en route to the loading zone.
        #
        # Post-load tight intake: drive straight to the spawn and grab the
        # loaded element(s) before the wall-pin heuristic makes the bot
        # drive away.  Selection 1/2 spawns a cup AND a separate nested
        # pin as TWO physics objects at the same corner.  Robot.try_intake
        # grabs the loose pin first; if we exited on the first intake we'd
        # leave the cup behind.  So: keep returning to spawn while EITHER
        # slot is empty AND an uncarried element still exists near spawn.
        if (self._just_loaded_steps > 0 and self._last_load_spawn is not None
                and not (has_pin and has_cup)
                and self._element_loose_near(self._last_load_spawn)):
            spawn = self._last_load_spawn
            d = self._dist(self._pos(), spawn)
            a["intake"] = True
            if d <= INTAKE_RADIUS * 1.5:
                a["left"], a["right"] = self._drive_to(
                    spawn, full_speed=False, avoid_obstacles=False)
                a["left"]  *= 0.4
                a["right"] *= 0.4
            else:
                a["left"], a["right"] = self._drive_to(
                    spawn, full_speed=True, avoid_obstacles=False)
            self.last_reason = "intake_loaded"
            return a
        if self._just_loaded_steps == 0:
            self._last_load_spawn = None

        # Refresh commit if conditions warrant a new trip.
        if self._should_match_load(has_pin, has_cup):
            self._match_load_commit = 60

        if self._match_load_commit > 0 and (not has_pin or not has_cup):
            zone_idx, lz_pt = self._pick_loading_zone()
            self._target_load_zone = zone_idx
            spawn = self._spawn_for_loading_zone(zone_idx)
            if self._in_loading_zone():
                a["match_load"] = True
                a["left"], a["right"] = 0.0, 0.0
                self._last_load_spawn   = spawn
                self._just_loaded_steps = 30   # ~1.5 s window
                self._match_load_commit = 0    # trip done
                self.last_reason = f"match_load_fire:{zone_idx}"
                return a
            a["left"], a["right"] = self._drive_to(
                lz_pt, full_speed=True, avoid_obstacles=True)
            self.last_reason = f"drive_to_match_load:{zone_idx}"
            return a

        if self._match_load_commit == 0:
            self._target_load_zone = None

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
                # Inside brake radius but score_now didn't fire — this goal
                # is unservable with our current inventory.  Pick a concrete
                # next target instead of drifting to midfield.
                #   - If we can dump anywhere else useful: drive there.
                #   - Else if a toggle would help: press it.
                #   - Else go to the SECOND-best goal.
                alt = self._nearest_dump_goal(has_pin, has_cup)
                if alt is not None and alt.goal_id != target_goal.goal_id:
                    self._target_goal_id = alt.goal_id
                    a["left"], a["right"] = self._drive_to(
                        self._approach_goal_point(alt), full_speed=True)
                    self.last_reason = f"redirect_goal:{alt.goal_id}"
                    return a
                tog = self._find_useful_toggle()
                if tog is not None:
                    tx, ty = float(tog.x), float(tog.y)
                    if self._dist(self._pos(), (tx, ty)) <= TOGGLE_INTERACTION_RANGE:
                        a["toggle"] = True
                        self.last_reason = f"flip_toggle_unserve:{tog.toggle_id}"
                    else:
                        a["left"], a["right"] = self._drive_to((tx, ty), full_speed=True)
                        self.last_reason = f"go_toggle_unserve:{tog.toggle_id}"
                    return a
                # Last resort: back off to midfield to find new opportunities.
                a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
                self.last_reason = f"backoff_unservable:{target_goal.goal_id}"
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
            # Pin in hand but no goal needs ONLY a pin.  Fetch a cup so we
            # can complete a stack.
            cup = self._pick_best_cup()
            if cup is not None:
                return self._action_get_cup(a, cup)
            # No cups on field either.  Don't roam — DUMP the pin at the
            # nearest goal that can take it (even at suboptimal value),
            # then go re-acquire.  Sitting on a pin we can't use is waste.
            dump_goal = self._nearest_dump_goal(has_pin=True, has_cup=False)
            if dump_goal is not None:
                self._target_goal_id = dump_goal.goal_id
                a["left"], a["right"] = self._drive_to(
                    self._approach_goal_point(dump_goal), full_speed=True)
                self.last_reason = f"dump_pin_at:{dump_goal.goal_id}"
                return a
            # No goal can take the pin (all full).  Press a toggle to
            # change ownership so the goal becomes scoreable for us.
            tog = self._find_useful_toggle()
            if tog is not None:
                a["left"], a["right"] = self._drive_to(
                    (float(tog.x), float(tog.y)), full_speed=True)
                self.last_reason = f"flip_toggle_for_pin:{tog.toggle_id}"
                return a
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
            self.last_reason = "carry_pin_search"
            return a

        if not has_pin and has_cup:
            # Cup in hand but no goal can take a cup.  Fetch a pin so we
            # can place a pin first (cups need a pin underneath).
            pin = self._pick_best_pin()
            if pin is not None:
                return self._action_get_pin(a)
            # No pins available.  Try to dump the cup at any goal that has
            # a top-pin (needs a cup).
            dump_goal = self._nearest_dump_goal(has_pin=False, has_cup=True)
            if dump_goal is not None:
                self._target_goal_id = dump_goal.goal_id
                a["left"], a["right"] = self._drive_to(
                    self._approach_goal_point(dump_goal), full_speed=True)
                self.last_reason = f"dump_cup_at:{dump_goal.goal_id}"
                return a
            tog = self._find_useful_toggle()
            if tog is not None:
                a["left"], a["right"] = self._drive_to(
                    (float(tog.x), float(tog.y)), full_speed=True)
                self.last_reason = f"flip_toggle_for_cup:{tog.toggle_id}"
                return a
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
            self.last_reason = "carry_cup_search"
            return a

        # Both slots full and no scoreable goal — first try to flip a
        # toggle to enable scoring at SOME goal.  Otherwise patrol midfield
        # (mobile, not idle — we'll spot scoring opportunities as goal
        # state changes from opponent activity).
        tog = self._find_useful_toggle()
        if tog is not None:
            tx, ty = float(tog.x), float(tog.y)
            if self._dist(self._pos(), (tx, ty)) <= TOGGLE_INTERACTION_RANGE:
                a["toggle"] = True
                self.last_reason = f"flip_toggle_full_inv:{tog.toggle_id}"
            else:
                a["left"], a["right"] = self._drive_to((tx, ty), full_speed=True)
                self.last_reason = f"go_toggle_full_inv:{tog.toggle_id}"
            return a
        a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=True)
        self.last_reason = "patrol_midfield"
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

    # v13 — Env-wrapper cooldown mirroring.  The env_wrapper applies
    # cooldowns to discrete actions (score, intake, toggle, match_load,
    # flip) — if the bot fires the action while the env's cooldown is
    # active, the action is silently dropped.  To avoid wasting decision
    # cycles spamming a gated action, the bot mirrors the same cooldowns.
    # Values mirror config/hyperparameters.py.
    _ENV_CD_INTAKE     = 5
    _ENV_CD_SCORE      = 10
    _ENV_CD_TOGGLE     = 100
    _ENV_CD_MATCH_LOAD = 30
    _ENV_CD_FLIP       = 40

    def get_policy_action(self):
        """Return `(cont, disc)` numpy arrays matching env_wrapper format."""
        import numpy as np
        # Initialize env-cooldown counters if missing (lazy init for
        # backward compat with existing instances).
        for attr in ("_env_cd_intake", "_env_cd_score",
                     "_env_cd_toggle", "_env_cd_match_load",
                     "_env_cd_flip"):
            if not hasattr(self, attr):
                setattr(self, attr, 0)
        # Decrement all env cooldowns once per call.
        for attr in ("_env_cd_intake", "_env_cd_score",
                     "_env_cd_toggle", "_env_cd_match_load",
                     "_env_cd_flip"):
            if getattr(self, attr) > 0:
                setattr(self, attr, getattr(self, attr) - 1)
        a = self.get_sim_action()
        # Gate actions that are on cooldown — env would drop them anyway.
        # Saves wasted cycles by letting the bot do something else.
        if self._env_cd_intake > 0 and a.get("intake"):
            a["intake"] = False
        if self._env_cd_score > 0:
            if a.get("score_pin"):  a["score_pin"]  = False
            if a.get("score_cup"):  a["score_cup"]  = False
        if self._env_cd_toggle > 0 and a.get("toggle"):
            a["toggle"] = False
        if self._env_cd_match_load > 0 and a.get("match_load"):
            a["match_load"] = False
        if self._env_cd_flip > 0:
            if a.get("flip_pin"):  a["flip_pin"]  = False
            if a.get("flip_cup"):  a["flip_cup"]  = False
        # Set cooldowns after FIRING (whether or not the env actually
        # accepts; the env's gate matches ours so this stays in sync).
        if a.get("intake"):     self._env_cd_intake     = self._ENV_CD_INTAKE
        if a.get("score_pin"):  self._env_cd_score      = self._ENV_CD_SCORE
        if a.get("score_cup"):  self._env_cd_score      = self._ENV_CD_SCORE
        if a.get("toggle"):     self._env_cd_toggle     = self._ENV_CD_TOGGLE
        if a.get("match_load"): self._env_cd_match_load = self._ENV_CD_MATCH_LOAD
        if a.get("flip_pin"):   self._env_cd_flip       = self._ENV_CD_FLIP
        if a.get("flip_cup"):   self._env_cd_flip       = self._ENV_CD_FLIP
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
            1.0 if a["intake"]     else 0.0,
            1.0 if a["score_pin"]  else 0.0,
            1.0 if a["score_cup"]  else 0.0,
            1.0 if a["toggle"]     else 0.0,
            1.0 if a["flip_pin"]   else 0.0,
            1.0 if a["flip_cup"]   else 0.0,
            1.0 if a["match_load"] else 0.0,
        ], dtype=np.float32)
        return cont, disc

    # ── Match-load helpers ────────────────────────────────────────────── #

    def _can_match_load(self) -> bool:
        """Only red1/blue1 are eligible (matches simulator restriction)."""
        return self.robot_id in ("red1", "blue1")

    def _loading_zone_targets(self) -> List[Tuple[float, float]]:
        """Two corner waypoints inside our alliance's loading zone.
        Index 0 = NORTH corner (y>120), index 1 = SOUTH corner (y<24).

        v13 — Deeper blue target.  Red's target (10, 128) lands at the zone
        boundary (x<12 fires the zone check) and the bot reaches center
        x≈10 facing north — west extent is just 7.5" so center can sit
        right against the wall and still satisfy x<12.  Blue's symmetric
        target (134, 128) DOESN'T work because the bot approaches with
        some south-east heading from its starting position (72,136) and
        its east-extent is >9" at any non-purely-east angle, so center
        stops at ~131.5 — fails the x>132 zone check.  Pushing blue's
        target to (140, 128) keeps the bot driving east through the
        wall-stop, letting the center reach its wall-limited maximum
        which IS inside the zone.
        """
        if self.alliance == "red":
            return [(10.0, 128.0), (10.0, 16.0)]
        return [(140.0, 128.0), (140.0, 16.0)]

    def _pick_loading_zone(self) -> Tuple[int, Tuple[float, float]]:
        """Choose which loading-zone corner to use.

        Logic (in order):
          1. If we've committed to a corner this trip, keep it (sticky).
          2. Prefer un-claimed corners with a clear line-of-sight from us.
          3. Fall back to closest by distance.

        Returns (index, (x, y)).
        """
        zones = self._loading_zone_targets()
        my = self._pos()
        partners = self._partner_targets()
        claimed = partners.get("load_zones", set())

        # Sticky: keep our previous choice while we're still pursuing it.
        if self._target_load_zone in (0, 1):
            return self._target_load_zone, zones[self._target_load_zone]

        # Prefer un-claimed AND line-of-sight clear, ordered by distance.
        candidates_clear = [
            (i, zones[i]) for i in (0, 1)
            if i not in claimed and self._line_clear_of_goals(my, zones[i])
        ]
        if candidates_clear:
            idx, pt = min(candidates_clear,
                          key=lambda ip: self._dist(my, ip[1]))
            return idx, pt

        # Fall back: any un-claimed, by distance.
        candidates = [(i, zones[i]) for i in (0, 1) if i not in claimed]
        if not candidates:
            candidates = [(i, zones[i]) for i in (0, 1)]
        idx, pt = min(candidates, key=lambda ip: self._dist(my, ip[1]))
        return idx, pt

    def _nearest_loading_zone(self) -> Tuple[float, float]:
        """Closest corner (read-only convenience — does NOT commit)."""
        my = self._pos()
        return min(self._loading_zone_targets(),
                   key=lambda t: self._dist(my, t))

    def _spawn_for_loading_zone(self, idx: int) -> Tuple[float, float]:
        """Where the spawned cup/pin lands for this loading zone index.
        Spawn y is derived from the zone corner y-coordinate (not the index)
        so it stays correct regardless of zone ordering in _loading_zone_targets.
        """
        zones = self._loading_zone_targets()
        lz_y = zones[idx][1]
        sx = 6.0 if self.alliance == "red" else 138.0
        sy = 132.0 if lz_y > 72 else 12.0   # north zone → spawn at y=132, south → y=12
        return (sx, sy)

    def _element_loose_near(self, point: Tuple[float, float],
                            radius: float = 14.0) -> bool:
        """True if any uncarried, unscored pin or cup sits within `radius`
        of `point`.  Used by post-load tight-intake to know whether the
        loader still has something to fetch — match-load selections 1/2
        spawn cup+pin as two physics objects, and after grabbing the pin
        first the cup is still loose at the spawn.
        """
        for p in self.sim.pins:
            if p.scored or p.carried_by is not None:
                continue
            if self._dist(point, (float(p.body.position.x),
                                  float(p.body.position.y))) <= radius:
                return True
        for c in self.sim.cups:
            if c.scored or c.carried_by is not None:
                continue
            if self._dist(point, (float(c.body.position.x),
                                  float(c.body.position.y))) <= radius:
                return True
        return False

    def _in_loading_zone(self) -> bool:
        x, y = self._pos()
        if self.alliance == "red":
            return x < 12.0 and (y < 24.0 or y > 120.0)
        return x > 132.0 and (y < 24.0 or y > 120.0)

    def _has_reachable_loading_zone(self, max_d: float = 70.0) -> bool:
        """True if at least one loading-zone corner is within max_d inches.

        Distance-only gate (no line-of-sight check here).  The LoS check is
        used in _pick_loading_zone as a PREFERENCE (prefer clear paths) but
        NOT as a hard requirement for triggering the trip.  Keeping LoS out
        of this gate prevents commit-expiry when the robot drifts slightly
        during its initial spin manoeuvre: even from x≈18 the corner is still
        within 70" so the commit stays alive and the robot continues driving.

        max_d=70" covers both:
          • red1  start (10,72)  → corners (10,16)/(10,128)  = 56"
          • blue1 start (72,136) → corner  (134,128)          = 62.5"
        """
        my = self._pos()
        for pt in self._loading_zone_targets():
            if self._dist(my, pt) <= max_d:
                return True
        return False

    def _line_clear_of_goals(self, a: Tuple[float, float],
                              b: Tuple[float, float]) -> bool:
        """True if the segment a→b doesn't physically collide with any goal.

        Goals in pymunk use `radius * 0.25` as their physics body radius
        (regular goals: 10 * 0.25 = 2.5").  The robot's half-width when
        facing along the path is ~7.5".  Required clearance from goal centre
        to robot path centre ≥ 2.5 + 7.5 = 10".  We use 12" for a small
        safety margin — enough to allow the wall-hugging approaches to the
        loading-zone corners (path is ~14" from nearby goals) while still
        blocking paths that genuinely cut through a goal body.
        """
        ax, ay = a; bx, by = b
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-6:
            return True
        for g in self.sim.goals:
            # Project goal centre onto the segment
            t = ((g.x - ax) * dx + (g.y - ay) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            cx = ax + t * dx
            cy = ay + t * dy
            if math.hypot(g.x - cx, g.y - cy) < 12.0:
                return False
        return True

    def _nearest_useful_element_dist(self) -> float:
        """Closest distance to any non-wall, valuable, fetchable pin or cup.
        Used to compare match-load round-trip vs. normal-fetch round-trip.
        """
        my = self._pos()
        best = float("inf")
        for p in self.sim.pins:
            if p.scored or p.carried_by is not None: continue
            if getattr(p, 'is_nested', False): continue
            if self._pin_value(p) <= 0: continue
            px, py = float(p.body.position.x), float(p.body.position.y)
            if self._is_wall_pinned(px, py): continue
            d = self._dist(my, (px, py))
            if d < best: best = d
        for c in self.sim.cups:
            if c.scored or c.carried_by is not None: continue
            cx, cy = float(c.body.position.x), float(c.body.position.y)
            if self._is_wall_pinned(cx, cy): continue
            d = self._dist(my, (cx, cy))
            if d < best: best = d
        return best

    def _inventory_left(self) -> Tuple[int, int, int]:
        """(cups, alliance_pins, yellow_pins) remaining for our alliance."""
        s = self.sim
        if self.alliance == "red":
            return (int(s.red_cups_left), int(s.red_alliance_pins_left),
                    int(s.red_yellow_pins_left))
        return (int(s.blue_cups_left), int(s.blue_alliance_pins_left),
                int(s.blue_yellow_pins_left))

    def _have_loose_yellow_yellow_on_field(self) -> bool:
        """True if a yellow_yellow pin (both halves yellow) is loose and
        accessible (not nested inside a cup, not already scored/carried,
        not wall-pinned past the reachability margin).

        match_load selection 2 spawns a 'yellow_yellow' pin specifically —
        so checking only for that color matches the value of the load.
        Nested yy pins (inside cups in goals or on the field) don't count
        because they require the cup to be scored/intaked first; until then,
        the match-load route is the only reliable way to get a fresh yy.
        """
        for p in self.sim.pins:
            if getattr(p, 'scored', False):
                continue
            if getattr(p, 'carried_by', None) is not None:
                continue
            if getattr(p, 'is_nested', False):
                continue  # still inside a cup — not directly grababble
            up = getattr(p, 'up_half_name', None)
            dn = getattr(p, 'down_half_name', None)
            if up == "yellow" and dn == "yellow":
                px, py = float(p.body.position.x), float(p.body.position.y)
                if self._is_wall_pinned(px, py):
                    continue  # physically unreachable
                return True
        return False

    def _is_in_autonomous(self) -> bool:
        """True if the match is in the autonomous period — match-loading is
        prohibited, and crossing the diagonal tape walls is physically
        blocked.  The bot must avoid both."""
        return self.sim.match_phase == "autonomous"

    def _should_match_load(self, has_pin: bool, has_cup: bool) -> bool:
        """Decide whether the bot should head to the loading zone.

        v14 — AGGRESSIVE CYCLING for red1/blue1:
          Match-load inventory holds far more elements (10 cups + 12 pins
          + 1 yellow per alliance = 23 each side) than the field starts
          with (~12 pins + 16 cups across the WHOLE field, half ours).
          So red1/blue1 should treat the loading zone as their PRIMARY
          source of elements and cycle (load → score → load → score)
          throughout the match.  Field elements are a side gig picked up
          en route via opportunistic intake.

          Triggers:
            1. Empty hands + any inventory + loading zone reachable.
               → Just go.  Match-load is the dominant strategy.
            2. Partial inventory (cup OR pin, not both) + the OTHER slot
               can be filled by match-load:
                 - has_pin: cup-alone (sel 3) fills cup slot
                 - has_cup: pin-alone (sel 4 or 5) fills pin slot
               → Cycle continues without dropping back to field-fetch.
            3. Both hands full → False (must score first).
        """
        if not self._can_match_load():
            return False
        # VEX VRC rule: no match-loading during autonomous.
        if self._is_in_autonomous():
            return False
        if has_pin and has_cup:
            return False
        if self._ml_cooldown_steps > 0:
            return False
        # Skip in the final ~15 s of driver — load/score/park needs time.
        if (self.sim.match_phase == "driver" and
                self.sim.time_remaining is not None and
                self.sim.time_remaining <= 15.0):
            return False
        cups, ap, yp = self._inventory_left()
        if cups + ap + yp == 0:
            return False
        if not self._has_reachable_loading_zone():
            return False
        my = self._pos()
        loading_d = self._dist(my, self._nearest_loading_zone())

        # ── CASE 1 — Empty hands + ANY inventory: ALWAYS fire ──────────
        # Match-load round-trip ≈ 8-10s with nested extraction giving
        # pin+cup in one intake.  Far more efficient than chasing field
        # elements that are often wall-pinned, nested, or partner-claimed.
        if not has_pin and not has_cup:
            return True

        # ── CASE 2 — Partial inventory + ML can complete the pair ──────
        # has_pin but no cup: ML sel 3 gives cup (then score pin+cup combo)
        # has_cup but no pin: ML sel 4/5 gives pin (then score combo)
        # Distance-gated so the bot doesn't make a 100" detour for a
        # single element — but we're VERY permissive (80") because the
        # combo score after refill is worth ~15-20 pts.
        if has_pin and not has_cup and cups > 0 and loading_d <= 80.0:
            return True
        if has_cup and not has_pin and (ap + yp) > 0 and loading_d <= 80.0:
            return True

        return False

    def _closest_loose_yy_distance(self) -> float:
        """Distance to nearest reachable loose yellow_yellow pin (or inf)."""
        my = self._pos()
        best = float("inf")
        for p in self.sim.pins:
            if getattr(p, 'scored', False): continue
            if getattr(p, 'carried_by', None) is not None: continue
            if getattr(p, 'is_nested', False): continue
            if getattr(p, 'up_half_name', None) != "yellow": continue
            if getattr(p, 'down_half_name', None) != "yellow": continue
            px, py = float(p.body.position.x), float(p.body.position.y)
            if self._is_wall_pinned(px, py): continue
            d = self._dist(my, (px, py))
            if d < best: best = d
        return best

    def _nearest_useful_pin_distance(self) -> float:
        """Distance to nearest pin with positive _pin_value (or inf)."""
        my = self._pos()
        best = float("inf")
        for p in self.sim.pins:
            if p.scored or p.carried_by is not None: continue
            if getattr(p, 'is_nested', False): continue
            if self._pin_value(p) <= 0: continue
            px, py = float(p.body.position.x), float(p.body.position.y)
            if self._is_wall_pinned(px, py): continue
            d = self._dist(my, (px, py))
            if d < best: best = d
        return best

    def _any_useful_pin_on_field(self) -> bool:
        """Read-only: is there at least one valuable, fetchable pin?"""
        for p in self.sim.pins:
            if p.scored or p.carried_by is not None:
                continue
            if getattr(p, 'is_nested', False):
                continue
            if self._pin_value(p) <= 0:
                continue
            px, py = float(p.body.position.x), float(p.body.position.y)
            if self._is_wall_pinned(px, py):
                continue
            return True
        return False

    def _any_useful_cup_on_field(self) -> bool:
        """Read-only: is there at least one fetchable cup?"""
        for c in self.sim.cups:
            if c.scored or c.carried_by is not None:
                continue
            cx, cy = float(c.body.position.x), float(c.body.position.y)
            if self._is_wall_pinned(cx, cy):
                continue
            return True
        return False

    @staticmethod
    def _zero_action() -> Dict[str, Any]:
        return {"left": 0.0, "right": 0.0,
                "intake": False, "score_pin": False, "score_cup": False,
                "toggle": False, "flip_pin": False, "flip_cup": False,
                "match_load": False}

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
        pre-flips and at-range flips agree.

        v15: en-route pre-flip threshold raised from +0.5 to +3 pts.
        Pre-flip eats the 40-step (2s) flip cooldown, blocking any
        at-range flip — so it should only fire when the orientation
        gain is meaningful (a half visibility worth ~5 pts, easily
        worth the 2s).  Tiny +0.5 gains were causing wasted cooldowns
        on goals the bot then bypassed mid-trip."""
        pin = self.robot.carrying_pin
        if pin is None or self._goal_needs(goal) != "pin":
            return False
        v_now  = self._pin_score_value(goal, pin.up_half_name, pin.down_half_name)
        v_flip = self._pin_score_value(goal, pin.down_half_name, pin.up_half_name)
        return v_flip > v_now + 3.0

    # ── Priority 8 — proactive toggle routing ────────────────────────── #

    def _find_useful_toggle(self):
        my_pos = self._pos()
        in_auton = self._is_in_autonomous()
        best = None
        best_sc = -1e9
        for t in self.sim.toggles:
            if t.owner == self.alliance:
                continue
            tx, ty = float(t.x), float(t.y)
            if in_auton and not self._is_in_my_auton_wedge(tx, ty):
                continue
            d = self._dist(my_pos, (tx, ty))
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

    def _find_high_value_toggle_flip(self):
        """E1 — Return a toggle worth interrupting current activity to flip.

        Computes the SCORE SWING (our pts gained + opp pts denied) from
        flipping each non-owned toggle.  Only returns a toggle if the
        swing exceeds 8 pts AND the toggle is within
        TOGGLE_DEFENSE_MAX_DIST (35") AND we have CLEAR LINE OF SIGHT
        AND haven't blacklisted it this match.

        Line-of-sight prevents the bot from pursuing toggles blocked by
        goals/walls — observed: blue2 spent 71s driving toward an
        unreachable toggle, losing 30+ points of normal scoring.

        Blacklist (self._hv_toggle_blacklist) is set when stuck-escape
        fires during a high-value toggle pursuit, so the bot doesn't
        endlessly re-target the same unreachable toggle.
        """
        my_pos = self._pos()
        in_auton = self._is_in_autonomous()
        best = None
        best_swing = 8.0    # threshold — anything ≤ 8 isn't worth a detour
        blacklist = getattr(self, '_hv_toggle_blacklist', set())
        for t in self.sim.toggles:
            if t.owner == self.alliance:
                continue
            if t.toggle_id in blacklist:
                continue
            tx, ty = float(t.x), float(t.y)
            if in_auton and not self._is_in_my_auton_wedge(tx, ty):
                continue
            d = self._dist(my_pos, (tx, ty))
            if d > self.TOGGLE_DEFENSE_MAX_DIST:
                continue
            # Compute swing: sum of yellow half values across goals this
            # toggle controls.  If opp owns it: we GAIN those halves (+10
            # each) and DENY opp (+10 each more), so 2x multiplier.
            # If neutral: we GAIN +10 each (no opp denial).
            swing = 0.0
            for g in self.sim.goals:
                if self._toggle_for_goal(g) is not t:
                    continue
                # Count CURRENTLY VISIBLE yellow halves on this goal.
                n = len(g.stack)
                for i, (obj, is_pin) in enumerate(g.stack):
                    if not is_pin:
                        continue
                    # UP half visible iff stack-top OR cup above is clear_up.
                    up_vis = True
                    if i + 1 < n:
                        nxt_obj, nxt_is_pin = g.stack[i + 1]
                        if not nxt_is_pin:
                            flipped = getattr(nxt_obj, 'flipped', False)
                            eff_clear_up = ((not nxt_obj.clear_on_top)
                                            if flipped else nxt_obj.clear_on_top)
                            up_vis = not eff_clear_up
                    # DOWN half visible iff pin-below or cup-below with clear_up.
                    if i == 0:
                        down_vis = False
                    else:
                        prev_obj, prev_is_pin = g.stack[i - 1]
                        if prev_is_pin:
                            down_vis = True
                        else:
                            flipped = getattr(prev_obj, 'flipped', False)
                            eff_clear_up = ((not prev_obj.clear_on_top)
                                            if flipped else prev_obj.clear_on_top)
                            down_vis = eff_clear_up
                    if up_vis and obj.up_half_name == "yellow":
                        swing += 10.0
                    if down_vis and obj.down_half_name == "yellow":
                        swing += 10.0
            if t.owner == self.opp_alliance:
                swing *= 2.0    # double — gain + deny
            # Travel cost: subtract pts per second of detour.
            t_cost = self._travel_time_to((float(t.x), float(t.y)))
            net = swing - t_cost * 2.0   # ~2 pts/s opportunity cost
            if net > best_swing:
                best_swing = net
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
