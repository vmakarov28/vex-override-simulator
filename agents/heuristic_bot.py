"""
agents/heuristic_bot.py
=======================
Rule-based "perfect-play" bot for VEX Override.

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

Decision priority (highest first)
---------------------------------
1.  EMERGENCY INTERCEPT — an opponent is one step away from scoring a
    valuable element into a goal that hurts me; charge them off course.
2.  ENDGAME PARK — endgame is active AND I'm carrying nothing useful;
    drive to the midfield to claim parking + SC5b yellow majority.
3.  PRE-PLACEMENT CUP FLIP — I'm carrying a cup, approaching a goal
    whose top pin has a known up-colour, and my cup orientation is
    wrong for that goal.  Flip the cup NOW (before arriving) so the
    score action places it correctly.
4.  SCORE — I have a usable element AND I'm near a valid goal.  Pick
    pin-vs-cup based on stack legality and place it.
5.  FLIP PIN — I'm carrying a pin with a yellow half, my alliance owns
    (or can soon own) the toggle for my chosen target goal, and the
    pin's UP face is the wrong colour — flip so YELLOW faces up.
6.  TOGGLE — there's a toggle within range that isn't mine, flip it.
7.  PICKUP CUP — I lack a cup; drive to nearest unscored, uncarried cup.
8.  PICKUP PIN — I lack a pin; drive to nearest helpful pin (own-color
    or yellow > opp_yellow > unscored neutral).
9.  DEFAULT — drive toward midfield centre to stay positionally useful.

Where this bot is intentionally NOT exhaustive
----------------------------------------------
- It does not solve multi-step pathing around obstacles.  If a pin sits
  behind the goal post the bot will try to drive through it.  Practical
  solution: tangent-steering to nearest navigable point (TODO).
- It does not coordinate with its alliance partner.  Two bots may pick
  the same goal/pin simultaneously.  Mitigation: a tiebreak based on
  robot_id (one partner shifts to the second-best target).
- It does not predict the opponent's flip/score timing past one step.
- It treats SC5b center-goal yellows as worth +10 only if my alliance
  has midfield majority RIGHT NOW; it does not strategically dive in to
  break a tie at match end (TODO).
"""

import math
from typing import Optional, Tuple, Dict, Any, List

# Game constants — MUST import from config.game_rules (the authoritative
# version the simulator/rules_engine actually use).  The legacy
# game_rules.py in the project root has stale values (SCORING_RADIUS=16
# vs the live 12, etc.) that would cause the bot to try scoring outside
# the actual valid range.
from config.game_rules import (
    INTAKE_RADIUS, SCORING_RADIUS,
    TOGGLE_INTERACTION_RANGE, MIDFIELD_CENTER, MIDFIELD_HALF,
    FIELD_WIDTH, FIELD_HEIGHT,
)

CENTER_GOAL_ID = 4   # neutral midfield goal
TWO_PI = 2.0 * math.pi


def _norm_angle(a: float) -> float:
    """Normalise angle to (-pi, pi]."""
    while a >  math.pi:  a -= TWO_PI
    while a <= -math.pi: a += TWO_PI
    return a


def _eff_clear_up(cup) -> bool:
    """True iff the CLEAR (white) side of the cup is currently UP.

    Mirrors the same helper in env_wrapper/game_objects so the bot's
    orientation reasoning matches what the scoring engine sees.
    """
    flipped = getattr(cup, "flipped", False)
    return (not cup.clear_on_top) if flipped else cup.clear_on_top


# --------------------------------------------------------------------- #
# HeuristicBot
# --------------------------------------------------------------------- #
class HeuristicBot:
    """Single-robot perfect-play heuristic.

    Construct one per robot you want the bot to control:
        bot = HeuristicBot(robot_id="red1", sim=sim)
        # every control step:
        sim_action = bot.get_sim_action()
        # or, to feed into env_wrapper that expects policy actions:
        cont, disc = bot.get_policy_action()
    """

    def __init__(self, robot_id: str, sim):
        self.robot_id = robot_id
        self.sim = sim
        # Resolve the robot reference lazily so we tolerate sim.reset().
        self._robot_cache = None
        # Sticky targets — once we commit to a pin/cup we keep going for
        # it unless something higher priority happens.  Prevents the
        # "two robots constantly swap targets" failure mode.
        self._target_pin_id: Optional[int] = None
        self._target_cup_id: Optional[int] = None
        self._target_goal_id: Optional[int] = None
        # Per-bot flip cooldown so we don't infinite-loop trying to flip a
        # pin/cup during the env_wrapper's COOLDOWN_FLIP window.  Each
        # successful flip request burns FLIP_COOLDOWN_STEPS before the next
        # flip can be issued.
        self._flip_pin_lockout = 0
        self._flip_cup_lockout = 0
        # Diagnostic: most recent reason string for the chosen action.
        self.last_reason: str = "init"

    # Bot-side cooldown in CONTROL_DT ticks — must be ≥ env's flip cooldown
    # so the bot doesn't keep requesting flip during the gate's lockout.
    FLIP_COOLDOWN_STEPS = 8

    # ------------------------------------------------------------- #
    # Robot / alliance helpers
    # ------------------------------------------------------------- #
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

    # ------------------------------------------------------------- #
    # Geometry helpers
    # ------------------------------------------------------------- #
    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _drive_to(self, target_xy: Tuple[float, float],
                  full_speed: bool = True) -> Tuple[float, float]:
        """Compute (left, right) wheel commands to steer toward target_xy.

        Diff-drive steering: large heading error → spin in place; smaller
        error → forward with proportional yaw bias.
        """
        rx, ry = self._pos()
        tx, ty = target_xy
        desired = math.atan2(ty - ry, tx - rx)
        err = _norm_angle(desired - float(self.robot.body.angle))

        # Spin-in-place threshold (~52°)
        if abs(err) > math.radians(52):
            sgn = 1.0 if err > 0 else -1.0
            return -sgn, sgn  # left/right opposite → rotate

        base = 1.0 if full_speed else 0.7
        bias = max(-0.6, min(0.6, err * 1.2))
        # If err > 0 we want to yaw left (positive z) → right wheel faster
        left  = base - bias
        right = base + bias
        return max(-1.0, min(1.0, left)), max(-1.0, min(1.0, right))

    def _toggle_for_goal(self, goal) -> Optional["Toggle"]:
        """Return the toggle that controls this goal's yellow ownership.

        Mirrors the env_wrapper's _get_toggle_for_goal — quadrant by
        signed dx/dy from field centre.
        """
        if goal.goal_id == CENTER_GOAL_ID:
            return None  # SC5b: center is decided by midfield majority
        dx = goal.x - 72.0
        dy = goal.y - 72.0
        if abs(dx) >= abs(dy):
            tid = 1 if dx <= 0 else 2
        else:
            tid = 3 if dy <= 0 else 4
        for t in self.sim.toggles:
            if t.toggle_id == tid:
                return t
        return None

    def _valid_goal_for_me(self, goal) -> bool:
        """Can I legally score in this goal?  Neutral + own-alliance only."""
        return goal.alliance in ("neutral", self.alliance)

    # ------------------------------------------------------------- #
    # Element value heuristics
    # ------------------------------------------------------------- #
    def _pin_value(self, pin) -> float:
        """Estimate how useful it is for ME to pick up THIS pin.

        Higher = better.  Pure-opponent-color pins are negative because
        scoring them hands the opponent points.
        """
        c = pin.color
        own_solid = (self.alliance == "red" and c == "red") or \
                    (self.alliance == "blue" and c == "blue")
        opp_solid = (self.alliance == "red" and c == "blue") or \
                    (self.alliance == "blue" and c == "red")
        own_yellow = (self.alliance == "red" and c == "red_yellow") or \
                     (self.alliance == "blue" and c == "blue_yellow")
        opp_yellow = (self.alliance == "red" and c == "blue_yellow") or \
                     (self.alliance == "blue" and c == "red_yellow")
        pure_yellow = (c == "yellow")

        if own_solid:    return 1.0    # +5 per visible half, always for me
        if own_yellow:   return 1.4    # +5 own + up to +10 yellow if I own toggle
        if pure_yellow:  return 1.2    # up to +10 if I own toggle
        if opp_yellow:   return 0.5    # +5 to opp BUT yellow can be flipped up
        if opp_solid:    return -1.0   # pure opp — scoring it helps them
        return 0.1

    # ------------------------------------------------------------- #
    # Top-level dispatcher
    # ------------------------------------------------------------- #
    def get_sim_action(self) -> Dict[str, Any]:
        a = self._zero_action()
        # Decrement per-bot flip cooldowns each step
        if self._flip_pin_lockout > 0: self._flip_pin_lockout -= 1
        if self._flip_cup_lockout > 0: self._flip_cup_lockout -= 1

        # 1. EMERGENCY INTERCEPT
        intercept = self._find_intercept_target()
        if intercept is not None:
            opp_robot = intercept
            ox, oy = float(opp_robot.body.position.x), float(opp_robot.body.position.y)
            a["left"], a["right"] = self._drive_to((ox, oy), full_speed=True)
            self.last_reason = f"intercept:{opp_robot.robot_id}"
            return a

        # 2. ENDGAME PARK
        if self._endgame_should_park():
            tx, ty = self._best_park_position()
            a["left"], a["right"] = self._drive_to((tx, ty), full_speed=True)
            self.last_reason = "endgame_park"
            return a

        has_pin = self.robot.carrying_pin is not None
        has_cup = self.robot.carrying_cup is not None

        # 3. PRE-PLACEMENT CUP FLIP (only matters if I have a cup)
        if has_cup and self._flip_cup_lockout == 0:
            flip_cup, target_goal = self._should_flip_cup_now()
            if flip_cup:
                a["flip_cup"] = True
                self._flip_cup_lockout = self.FLIP_COOLDOWN_STEPS
                # Keep driving toward the goal while flipping
                a["left"], a["right"] = self._drive_to((target_goal.x, target_goal.y), full_speed=True)
                self.last_reason = f"flip_cup_for_goal:{target_goal.goal_id}"
                return a

        # 4. SCORE (full or partial inventory near a valid goal)
        if has_pin or has_cup:
            score_action = self._try_score()
            if score_action is not None:
                return score_action

        # 5. PRE-PLACEMENT PIN FLIP (carrying a yellow-half pin to a goal
        #    where my alliance owns the toggle — flip so yellow ends UP)
        if has_pin and self._flip_pin_lockout == 0:
            flip_pin, target_goal = self._should_flip_pin_now()
            if flip_pin:
                a["flip_pin"] = True
                self._flip_pin_lockout = self.FLIP_COOLDOWN_STEPS
                a["left"], a["right"] = self._drive_to((target_goal.x, target_goal.y), full_speed=True)
                self.last_reason = f"flip_pin_for_goal:{target_goal.goal_id}"
                return a

        # 6. TOGGLE (only if not full — toggles are quick detours)
        if not (has_pin and has_cup):
            toggle = self._find_useful_toggle()
            if toggle is not None:
                tx, ty = float(toggle.x), float(toggle.y)
                d = self._dist(self._pos(), (tx, ty))
                if d <= TOGGLE_INTERACTION_RANGE:
                    a["toggle"] = True
                    self.last_reason = f"flip_toggle:{toggle.toggle_id}"
                    return a
                a["left"], a["right"] = self._drive_to((tx, ty))
                self.last_reason = f"approach_toggle:{toggle.toggle_id}"
                return a

        # 7. PICKUP PIN
        if not has_pin:
            pin = self._pick_best_pin()
            if pin is not None:
                self._target_pin_id = pin.pin_id
                px, py = float(pin.body.position.x), float(pin.body.position.y)
                d = self._dist(self._pos(), (px, py))
                if d <= INTAKE_RADIUS:
                    a["intake"] = True
                    self.last_reason = f"intake_pin:{pin.pin_id}"
                    return a
                a["left"], a["right"] = self._drive_to((px, py))
                self.last_reason = f"approach_pin:{pin.pin_id}"
                return a

        # 8. PICKUP CUP
        if not has_cup:
            cup = self._pick_best_cup()
            if cup is not None:
                self._target_cup_id = id(cup)
                cx, cy = float(cup.body.position.x), float(cup.body.position.y)
                d = self._dist(self._pos(), (cx, cy))
                if d <= INTAKE_RADIUS:
                    a["intake"] = True
                    self.last_reason = f"intake_cup:{id(cup)}"
                    return a
                a["left"], a["right"] = self._drive_to((cx, cy))
                self.last_reason = f"approach_cup:{id(cup)}"
                return a

        # 9. DEFAULT
        a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=False)
        self.last_reason = "default_midfield"
        return a

    # ------------------------------------------------------------- #
    # Policy-action API (for use with env_wrapper)
    # ------------------------------------------------------------- #
    def get_policy_action(self):
        """Return `(cont, disc)` numpy arrays matching env_wrapper format.

        cont = [left, right]
        disc = [intake, score_pin, score_cup, toggle, flip_pin, flip_cup, match_load]
                 ↑ same bit order as in env_wrapper.step()
        """
        import numpy as np
        a = self.get_sim_action()
        cont = np.array([a["left"], a["right"]], dtype=np.float32)
        # disc bits are interpreted as ">0.5 == fire" in env_wrapper.
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

    # ------------------------------------------------------------- #
    # Priority 1 — emergency intercept
    # ------------------------------------------------------------- #
    def _find_intercept_target(self):
        """Return an opponent robot we should charge to disrupt.

        Criteria:
          - opponent carrying a pin or cup
          - opponent is within (SCORING_RADIUS * 1.8) of a goal that
            would score against US (their alliance goals OR a center
            goal where they could claim SC5b)
          - opponent is closer to that goal than I am to the opponent
            (so charging makes a difference)
          - I don't have a higher-value commitment right now
        """
        # Don't abandon a near-complete score myself
        if (self.robot.carrying_pin or self.robot.carrying_cup):
            for g in self.sim.goals:
                if self._valid_goal_for_me(g) and \
                   self._dist(self._pos(), (g.x, g.y)) <= SCORING_RADIUS + 4:
                    return None  # I'm about to score myself

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
                    continue  # they can't score in our goals
                d_opp_goal = self._dist((ox, oy), (g.x, g.y))
                if d_opp_goal > SCORING_RADIUS * 1.8:
                    continue
                d_me_opp = self._dist(my_pos, (ox, oy))
                # Can I plausibly reach them before they score?
                if d_me_opp > d_opp_goal + 8.0:
                    continue
                # Heuristic threat = proximity to goal × element value
                # (cup-only is mild, pin or full inventory worse)
                val = 1.0 if opp.carrying_pin else 0.5
                threat = val / (1.0 + d_opp_goal / 5.0)
                if threat > best_threat:
                    best_threat = threat
                    best = opp
        return best

    # ------------------------------------------------------------- #
    # Priority 2 — endgame park
    # ------------------------------------------------------------- #
    def _endgame_should_park(self) -> bool:
        if not self.sim.rules_engine.endgame_active:
            return False
        # If we're holding stuff, finish placing it first
        if self.robot.carrying_pin or self.robot.carrying_cup:
            return False
        # Time guard: only park in last ~6 s of endgame
        if self.sim.time_remaining > 6.0:
            return False
        return True

    def _best_park_position(self) -> Tuple[float, float]:
        """Pick a spot in midfield biased toward where SC5b matters."""
        # Stand near the center goal so we both park and contest midfield
        return (72.0, 72.0)

    # ------------------------------------------------------------- #
    # Priority 3 — pre-placement cup flip
    # ------------------------------------------------------------- #
    def _should_flip_cup_now(self):
        """If I'm carrying a cup, approaching a goal, and my cup orientation
        is wrong for that goal's top pin, return (True, goal).  Else
        (False, None).
        """
        cup = self.robot.carrying_cup
        if cup is None:
            return False, None
        # Pick the goal we're closest to (within reasonable range) that
        # has a pin on top and a known correct orientation.
        my_pos = self._pos()
        best_goal = None
        best_d = float("inf")
        best_correct = None
        for g in self.sim.goals:
            if not self._valid_goal_for_me(g):
                continue
            if not g.stack or not g.stack[-1][1]:
                continue  # top must be a pin
            d = self._dist(my_pos, (g.x, g.y))
            if d > SCORING_RADIUS * 2.5:
                continue
            correct = self._correct_cup_clear_up(g)
            if correct is None:
                continue
            if d < best_d:
                best_d = d
                best_goal = g
                best_correct = correct
        if best_goal is None:
            return False, None
        cup_clear_up = _eff_clear_up(cup)
        if cup_clear_up != best_correct:
            return True, best_goal
        return False, None

    def _correct_cup_clear_up(self, goal) -> Optional[bool]:
        """For a goal whose top is a pin, return the correct cup
        orientation as `clear_up=True/False`, or None if undetermined.

        clear_up=True  → CLEAR side of cup is UP → DARK side is DOWN →
                          BLOCKS the pin below's UP half (denial).
        clear_up=False → CLEAR side DOWN → preserves the pin below's UP half.
        """
        if not goal.stack or not goal.stack[-1][1]:
            return None
        top_pin, _ = goal.stack[-1]
        up = top_pin.get_up_color()
        if up == "red":
            return True if self.alliance == "blue" else False
        if up == "blue":
            return True if self.alliance == "red" else False
        if up == "yellow":
            tog = self._toggle_for_goal(goal)
            if tog is None or tog.owner not in ("red", "blue"):
                return None  # no decision possible yet
            if tog.owner == self.alliance:
                return False  # preserve own yellow
            return True       # deny opponent yellow
        return None

    # ------------------------------------------------------------- #
    # Priority 4 — score
    # ------------------------------------------------------------- #
    def _try_score(self):
        """If I'm near a valid goal and can legally place an element,
        return a sim_action that does so.  Else None.
        """
        goal = self._best_target_goal()
        if goal is None:
            return None
        gx, gy = goal.x, goal.y
        d = self._dist(self._pos(), (gx, gy))
        a = self._zero_action()
        if d > SCORING_RADIUS:
            a["left"], a["right"] = self._drive_to((gx, gy), full_speed=True)
            self.last_reason = f"approach_goal:{goal.goal_id}"
            return a

        # Determine what to place based on stack legality
        # Legal stack from bottom: pin → cup → pin (max 3 elements)
        stack = list(goal.stack)
        n = len(stack)
        if n >= 3:
            # Goal full — abort, find another
            self.last_reason = f"goal_full:{goal.goal_id}"
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=False)
            return a

        if n == 0:
            # Empty → place pin first
            if self.robot.carrying_pin is not None:
                a["score_pin"] = True
                self.last_reason = f"score_pin_empty:{goal.goal_id}"
                return a
            # We have only a cup; cup-first isn't legal — leave it for now
            self.last_reason = f"goal_empty_no_pin:{goal.goal_id}"
            a["left"], a["right"] = self._drive_to(MIDFIELD_CENTER, full_speed=False)
            return a

        top_obj, top_is_pin = stack[-1]
        if top_is_pin:
            # Top is pin → can place cup on top (legal)
            if self.robot.carrying_cup is not None:
                a["score_cup"] = True
                self.last_reason = f"score_cup_on_pin:{goal.goal_id}"
                return a
            # No cup → can't add another pin directly; need to fetch a cup
            self.last_reason = f"top_is_pin_need_cup:{goal.goal_id}"
            return None
        # Top is cup → can place pin (legal stack: pin-cup-pin)
        if self.robot.carrying_pin is not None:
            a["score_pin"] = True
            self.last_reason = f"score_pin_on_cup:{goal.goal_id}"
            return a
        # No pin, top is cup → nothing legal to do
        return None

    def _best_target_goal(self):
        """Pick the highest-value goal to deposit my current inventory into.

        Weighted by:
          - alliance (own/neutral only)
          - distance from me
          - stack depth (deeper = more points already, more risk of full)
          - SC5b consideration for the center goal if I have a yellow pin
        """
        my_pos = self._pos()
        best = None
        best_score = -1e9
        for g in self.sim.goals:
            if not self._valid_goal_for_me(g):
                continue
            n = len(g.stack)
            if n >= 3:
                continue  # full
            d = self._dist(my_pos, (g.x, g.y))
            # Penalty for distance; bonus for goals already started by us
            score = -d
            # Prefer goals whose toggle we own (for yellow scoring)
            tog = self._toggle_for_goal(g)
            if tog is not None and tog.owner == self.alliance:
                score += 20.0
            # Centre goal: only attractive if we have midfield presence
            if g.goal_id == CENTER_GOAL_ID:
                # Conservative: only score there if I think we'll have
                # midfield majority at match end (we have a robot here)
                if not self._we_have_midfield_majority():
                    score -= 15.0
            if score > best_score:
                best_score = score
                best = g
        if best is not None:
            self._target_goal_id = best.goal_id
        return best

    def _we_have_midfield_majority(self) -> bool:
        own = opp = 0
        for r in self.sim.robots:
            if self._in_midfield(float(r.body.position.x),
                                 float(r.body.position.y)):
                if r.alliance == self.alliance:
                    own += 1
                else:
                    opp += 1
        return own > opp

    @staticmethod
    def _in_midfield(x: float, y: float) -> bool:
        mx, my = MIDFIELD_CENTER
        return abs(x - mx) + abs(y - my) <= MIDFIELD_HALF + 10.0

    # ------------------------------------------------------------- #
    # Priority 5 — pre-placement pin flip
    # ------------------------------------------------------------- #
    def _should_flip_pin_now(self):
        """If I'm carrying a yellow-half pin heading to a goal whose
        toggle my alliance owns AND the pin's UP color isn't yellow,
        flip the pin so yellow lands up.
        """
        pin = self.robot.carrying_pin
        if pin is None or not pin.is_yellow:
            return False, None
        goal = self._best_target_goal()
        if goal is None:
            return False, None
        # Only worth flipping if I own the relevant toggle
        tog = self._toggle_for_goal(goal)
        if tog is None or tog.owner != self.alliance:
            # SC5b path: center goal — check midfield majority
            if goal.goal_id != CENTER_GOAL_ID or not self._we_have_midfield_majority():
                return False, None
        if pin.get_up_color() != "yellow":
            return True, goal
        return False, None

    # ------------------------------------------------------------- #
    # Priority 6 — toggle
    # ------------------------------------------------------------- #
    def _find_useful_toggle(self):
        """Return the toggle that, if flipped, helps me most.

        Useful if:
          - it's not currently owned by my alliance, AND
          - it controls a goal we plan to score yellows in, OR is just
            close enough to be a cheap detour.
        """
        my_pos = self._pos()
        best = None
        best_score = -1e9
        for t in self.sim.toggles:
            if t.owner == self.alliance:
                continue  # already mine
            d = self._dist(my_pos, (float(t.x), float(t.y)))
            if d > 36.0:
                continue  # too far to be a cheap detour
            # Score by inverse distance and whether the toggle controls
            # a goal that already has yellow halves visible
            score = -d
            for g in self.sim.goals:
                tog = self._toggle_for_goal(g)
                if tog is not t:
                    continue
                # Goal yellow potential? Just count yellow-half pins in stack
                for obj, is_pin in g.stack:
                    if is_pin and obj.is_yellow:
                        score += 8.0
            if score > best_score:
                best_score = score
                best = t
        return best

    # ------------------------------------------------------------- #
    # Priority 7 — pickup pin
    # ------------------------------------------------------------- #
    def _pick_best_pin(self):
        my_pos = self._pos()
        best = None
        best_score = -1e9
        for p in self.sim.pins:
            if p.scored or p.carried_by is not None:
                continue
            d = self._dist(my_pos, (float(p.body.position.x),
                                    float(p.body.position.y)))
            val = self._pin_value(p)
            if val <= 0:
                continue  # would be net negative for me to score
            # Sticky target bonus
            if self._target_pin_id == p.pin_id:
                d *= 0.7
            score = val * 12.0 - d
            if score > best_score:
                best_score = score
                best = p
        return best

    # ------------------------------------------------------------- #
    # Priority 8 — pickup cup
    # ------------------------------------------------------------- #
    def _pick_best_cup(self):
        my_pos = self._pos()
        best = None
        best_d = float("inf")
        for c in self.sim.cups:
            if c.scored or c.carried_by is not None:
                continue
            d = self._dist(my_pos, (float(c.body.position.x),
                                    float(c.body.position.y)))
            if self._target_cup_id == id(c):
                d *= 0.7
            if d < best_d:
                best_d = d
                best = c
        return best
