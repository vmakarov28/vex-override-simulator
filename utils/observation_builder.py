"""
utils/observation_builder.py
────────────────────────────────────────────────────────────────────────────
Builds the fixed-size 564-dimensional observation vector for one agent.

Layout  (all values normalized to roughly [-1, 1] or [0, 1]):
  [  0: 24]  Self state
  [ 24: 36]  Teammate state
  [ 36: 56]  Opponents (2 × 10)
  [ 56:209]  Goals (9 × 17 = 153)
  [209:409]  K-nearest pins (20 × 10 = 200)
  [409:514]  K-nearest cups (15 × 7 = 105)
  [514:534]  Toggles (4 × 5 = 20)
  [534:554]  Global match state (20)  [v7]
  [554:564]  v8.1 self-awareness + defensive intel (10)
               - own carry_steps / TIME_TO_SCORE_TARGET (1)
               - own holding-overshoot ratio (1)
               - opp1 carrying pin UP colour one-hot (3)
               - opp2 carrying pin UP colour one-hot (3)
               - yellow pins remaining / 12 (1)
               - can-score-anywhere bit (1)
  ──────────
  Total: 564
"""

import math
import numpy as np
from typing import List

from config.game_rules import (
    FIELD_WIDTH, FIELD_HEIGHT, ROBOT_MAX_SPEED,
    SCORING_RADIUS, INTAKE_RADIUS,
    MIDFIELD_CENTER, MIDFIELD_HALF,
    ENDGAME_SECONDS, TOTAL_SECONDS, AUTONOMOUS_SECONDS,
    CENTER_GOAL_ID, ENDGAME_CENTER_MAX_STACK,
)
from config.hyperparameters import (
    ENDGAME_RAMP_SECONDS, TIME_TO_SCORE_TARGET,
    HOLDING_TIMEOUT_STEPS, HOLDING_RAMP_STEPS,
)
from simulation.game_objects import (
    GamePin, GameCup, FieldGoal, FieldToggle,
    C_RED, C_BLUE, C_YELLOW,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FIELD_DIAG  = math.hypot(FIELD_WIDTH, FIELD_HEIGHT)   # ~203.6 in
MAX_SPEED   = float(ROBOT_MAX_SPEED)
MAX_ANG_VEL = 10.0

K_PINS = 20   # nearest pins to encode
K_CUPS = 15   # nearest cups to encode

OBS_DIM = 564

# ─────────────────────────────────────────────────────────────────────────────
# Color → one-hot helper  (3 bits: [red, blue, yellow])
# ─────────────────────────────────────────────────────────────────────────────
def _color_onehot(color_rgb):
    """Convert a pygame color tuple to [is_red, is_blue, is_yellow]."""
    if color_rgb == C_RED:
        return [1.0, 0.0, 0.0]
    elif color_rgb == C_BLUE:
        return [0.0, 1.0, 0.0]
    elif color_rgb == C_YELLOW:
        return [0.0, 0.0, 1.0]
    else:
        return [0.0, 0.0, 0.0]


def _rel(ax, ay, bx, by):
    """Normalised relative position of b relative to a."""
    return (bx - ax) / FIELD_WIDTH, (by - ay) / FIELD_HEIGHT


def _dist(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay) / FIELD_DIAG


def _eff_clear_up(cup) -> bool:
    """True = clear/white half is UP in the stack."""
    flipped = getattr(cup, 'flipped', False)
    return (not cup.clear_on_top) if flipped else cup.clear_on_top


def _is_in_midfield(rx, ry) -> bool:
    mc_x, mc_y = MIDFIELD_CENTER
    return abs(rx - mc_x) + abs(ry - mc_y) <= MIDFIELD_HALF + 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────
def build_observation(
    robot,
    all_robots: list,
    pins: list,
    cups: list,
    goals: list,
    toggles: list,
    rules_engine,
    simulator,
) -> np.ndarray:
    """
    Build the 564-dim observation vector for `robot`.

    Parameters
    ----------
    robot        : Robot — the observing agent
    all_robots   : List[Robot] — all 4 robots [red1, red2, blue1, blue2]
    pins         : List[GamePin]
    cups         : List[GameCup]
    goals        : List[FieldGoal]
    toggles      : List[FieldToggle]
    rules_engine : RulesEngine
    simulator    : OverrideSimulator — for match state (time, phase, inventory)

    Returns
    -------
    np.ndarray of shape (564,), dtype float32
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    ptr = 0

    rx, ry     = float(robot.body.position.x), float(robot.body.position.y)
    ra         = float(robot.body.angle)
    rvx, rvy   = float(robot.body.velocity.x), float(robot.body.velocity.y)
    rang       = float(robot.body.angular_velocity)

    # ── [0:24] Self state ────────────────────────────────────────────────────
    obs[ptr + 0]  = rx / FIELD_WIDTH
    obs[ptr + 1]  = ry / FIELD_HEIGHT
    obs[ptr + 2]  = math.sin(ra)
    obs[ptr + 3]  = math.cos(ra)
    obs[ptr + 4]  = rvx / MAX_SPEED
    obs[ptr + 5]  = rvy / MAX_SPEED
    obs[ptr + 6]  = rang / MAX_ANG_VEL
    obs[ptr + 7]  = 1.0 if robot.carrying_pin else 0.0
    obs[ptr + 8]  = 1.0 if robot.carrying_cup else 0.0

    if robot.carrying_pin:
        up_oh = _color_onehot(robot.carrying_pin.get_up_color())
        dn_oh = _color_onehot(robot.carrying_pin.get_down_color())
        obs[ptr + 9]  = float(robot.carrying_pin.flipped)
    else:
        up_oh = [0.0, 0.0, 0.0]
        dn_oh = [0.0, 0.0, 0.0]
        obs[ptr + 9] = 0.0

    obs[ptr + 10] = up_oh[0]; obs[ptr + 11] = up_oh[1]; obs[ptr + 12] = up_oh[2]
    obs[ptr + 13] = dn_oh[0]; obs[ptr + 14] = dn_oh[1]; obs[ptr + 15] = dn_oh[2]

    if robot.carrying_cup:
        obs[ptr + 16] = 1.0 if _eff_clear_up(robot.carrying_cup) else 0.0
        obs[ptr + 17] = 1.0 if robot.carrying_cup.contains_pin else 0.0
    else:
        obs[ptr + 16] = 0.0
        obs[ptr + 17] = 0.0

    obs[ptr + 18] = min(1.0, robot.intake_cooldown)
    obs[ptr + 19] = 1.0 if robot.is_scoring else 0.0
    obs[ptr + 20] = min(1.0, robot.successful_scores / 10.0)
    obs[ptr + 21] = 1.0 if robot.alliance == "red" else 0.0

    # distance to nearest goal / toggle
    g_dists = [math.hypot(rx - g.x, ry - g.y) for g in goals]
    t_dists = [math.hypot(rx - t.x, ry - t.y) for t in toggles]
    obs[ptr + 22] = min(g_dists) / FIELD_DIAG if g_dists else 0.0
    obs[ptr + 23] = min(t_dists) / FIELD_DIAG if t_dists else 0.0
    ptr += 24

    # ── [24:36] Teammate ─────────────────────────────────────────────────────
    teammates = [r for r in all_robots if r.robot_id != robot.robot_id
                 and r.alliance == robot.alliance]
    if teammates:
        tm = teammates[0]
        tx, ty = float(tm.body.position.x), float(tm.body.position.y)
        ta = float(tm.body.angle)
        tvx, tvy = float(tm.body.velocity.x), float(tm.body.velocity.y)
        rx_, ry_ = _rel(rx, ry, tx, ty)
        obs[ptr + 0]  = rx_
        obs[ptr + 1]  = ry_
        obs[ptr + 2]  = _dist(rx, ry, tx, ty)
        obs[ptr + 3]  = math.sin(ta)
        obs[ptr + 4]  = math.cos(ta)
        obs[ptr + 5]  = tvx / MAX_SPEED
        obs[ptr + 6]  = tvy / MAX_SPEED
        obs[ptr + 7]  = 1.0 if tm.carrying_pin else 0.0
        obs[ptr + 8]  = 1.0 if tm.carrying_cup else 0.0
        obs[ptr + 9]  = 1.0 if tm.is_scoring else 0.0
        obs[ptr + 10] = min(1.0, tm.successful_scores / 10.0)
        obs[ptr + 11] = 1.0 if _is_in_midfield(tx, ty) else 0.0
    ptr += 12

    # ── [36:56] Opponents (2 × 10) ───────────────────────────────────────────
    opponents = sorted(
        [r for r in all_robots if r.alliance != robot.alliance],
        key=lambda r: math.hypot(float(r.body.position.x) - rx,
                                 float(r.body.position.y) - ry)
    )
    for opp in opponents[:2]:
        ox, oy = float(opp.body.position.x), float(opp.body.position.y)
        oa = float(opp.body.angle)
        ovx, ovy = float(opp.body.velocity.x), float(opp.body.velocity.y)
        obs[ptr + 0] = (ox - rx) / FIELD_WIDTH
        obs[ptr + 1] = (oy - ry) / FIELD_HEIGHT
        obs[ptr + 2] = _dist(rx, ry, ox, oy)
        obs[ptr + 3] = math.sin(oa)
        obs[ptr + 4] = math.cos(oa)
        obs[ptr + 5] = ovx / MAX_SPEED
        obs[ptr + 6] = ovy / MAX_SPEED
        obs[ptr + 7] = 1.0 if opp.carrying_pin else 0.0
        obs[ptr + 8] = 1.0 if opp.carrying_cup else 0.0
        obs[ptr + 9] = 1.0 if opp.is_scoring else 0.0
        ptr += 10
    # pad if fewer than 2 opponents visible (shouldn't happen in 2v2)
    remaining_opps = max(0, 2 - len(opponents))
    ptr += remaining_opps * 10

    # ── [56:209] Goals (9 × 17 = 153) ────────────────────────────────────────
    for goal in goals:
        gx, gy = float(goal.x), float(goal.y)
        gdx, gdy = _rel(rx, ry, gx, gy)
        gdist = _dist(rx, ry, gx, gy)

        # Alliance type one-hot [red, blue, neutral]
        is_red_goal  = 1.0 if goal.alliance == "red"     else 0.0
        is_blue_goal = 1.0 if goal.alliance == "blue"    else 0.0
        is_neutral   = 1.0 if goal.alliance == "neutral" else 0.0

        stack_h = len(goal.stack) / 10.0
        top_is_pin = 0.0
        if goal.stack:
            _, top_is_pin_bool = goal.stack[-1]
            top_is_pin = 1.0 if top_is_pin_bool else 0.0

        red_sc  = goal.red_score  / 50.0
        blue_sc = goal.blue_score / 50.0

        # Scoring legality from this robot's perspective
        can_score_here = (goal.alliance == "neutral" or
                          goal.alliance == robot.alliance)
        close = gdist * FIELD_DIAG <= SCORING_RADIUS * 3
        can_score_pin = 1.0 if (
            can_score_here and robot.carrying_pin and
            (not goal.stack or not goal.stack[-1][1])  # top must not be pin
        ) else 0.0
        can_score_cup = 1.0 if (
            can_score_here and robot.carrying_cup and
            goal.stack and goal.stack[-1][1]           # top must be pin
        ) else 0.0

        # Denial potential: top of stack is an opponent's visible pin half
        denial_potential = 0.0
        if goal.stack:
            top_obj, top_is_p = goal.stack[-1]
            if top_is_p:
                up_col = top_obj.get_up_color()
                opp_cols = [C_BLUE if robot.alliance == "red" else C_RED]
                if up_col in opp_cols:
                    denial_potential = 1.0

        # Yellow halves count in this goal's stack
        yellow_halves = 0
        for (obj, is_p) in goal.stack:
            if is_p:
                if obj.get_up_color() == C_YELLOW:
                    yellow_halves += 1
                if obj.get_down_color() == C_YELLOW:
                    yellow_halves += 1
        yellow_halves_norm = min(1.0, yellow_halves / 5.0)

        # Top-pin UP color one-hot [is_red, is_blue, is_yellow].
        # Gives robots direct color information for cup-orientation and
        # flip decisions without requiring them to infer it from denial_potential.
        top_pin_up_red = top_pin_up_blue = top_pin_up_yellow = 0.0
        if goal.stack:
            top_obj, top_is_p = goal.stack[-1]
            if top_is_p:
                up_col = top_obj.get_up_color()
                if up_col == C_RED:
                    top_pin_up_red = 1.0
                elif up_col == C_BLUE:
                    top_pin_up_blue = 1.0
                elif up_col == C_YELLOW:
                    top_pin_up_yellow = 1.0

        obs[ptr + 0]  = gdx
        obs[ptr + 1]  = gdy
        obs[ptr + 2]  = gdist
        obs[ptr + 3]  = is_red_goal
        obs[ptr + 4]  = is_blue_goal
        obs[ptr + 5]  = is_neutral
        obs[ptr + 6]  = stack_h
        obs[ptr + 7]  = top_is_pin
        obs[ptr + 8]  = red_sc
        obs[ptr + 9]  = blue_sc
        obs[ptr + 10] = can_score_pin
        obs[ptr + 11] = can_score_cup
        obs[ptr + 12] = denial_potential
        obs[ptr + 13] = yellow_halves_norm
        obs[ptr + 14] = top_pin_up_red
        obs[ptr + 15] = top_pin_up_blue
        obs[ptr + 16] = top_pin_up_yellow
        ptr += 17

    # ── [209:409] K-nearest pins (20 × 10) ───────────────────────────────────
    live_pins = [p for p in pins if not p.scored]
    live_pins.sort(key=lambda p: math.hypot(float(p.body.position.x) - rx,
                                             float(p.body.position.y) - ry))
    for i in range(K_PINS):
        if i < len(live_pins):
            p = live_pins[i]
            px_, py_ = float(p.body.position.x), float(p.body.position.y)
            up_oh = _color_onehot(p.get_up_color())
            dn_oh = _color_onehot(p.get_down_color())
            obs[ptr + 0] = (px_ - rx) / FIELD_WIDTH
            obs[ptr + 1] = (py_ - ry) / FIELD_HEIGHT
            obs[ptr + 2] = _dist(rx, ry, px_, py_)
            obs[ptr + 3] = 1.0 if p.carried_by is not None else 0.0
            obs[ptr + 4] = up_oh[0]; obs[ptr + 5] = up_oh[1]; obs[ptr + 6] = up_oh[2]
            obs[ptr + 7] = dn_oh[0]; obs[ptr + 8] = dn_oh[1]; obs[ptr + 9] = dn_oh[2]
        ptr += 10

    # ── [409:514] K-nearest cups (15 × 7) ────────────────────────────────────
    live_cups = [c for c in cups if not c.scored]
    live_cups.sort(key=lambda c: math.hypot(float(c.body.position.x) - rx,
                                             float(c.body.position.y) - ry))
    for i in range(K_CUPS):
        if i < len(live_cups):
            c = live_cups[i]
            cx_, cy_ = float(c.body.position.x), float(c.body.position.y)
            # distance to nearest goal for this cup
            cup_g_dists = [math.hypot(cx_ - g.x, cy_ - g.y) for g in goals]
            min_cup_g = min(cup_g_dists) / FIELD_DIAG if cup_g_dists else 0.0
            obs[ptr + 0] = (cx_ - rx) / FIELD_WIDTH
            obs[ptr + 1] = (cy_ - ry) / FIELD_HEIGHT
            obs[ptr + 2] = _dist(rx, ry, cx_, cy_)
            obs[ptr + 3] = 1.0 if c.carried_by is not None else 0.0
            obs[ptr + 4] = 1.0 if _eff_clear_up(c) else 0.0
            obs[ptr + 5] = 1.0 if c.contains_pin is not None else 0.0
            obs[ptr + 6] = min_cup_g
        ptr += 7

    # ── [514:534] Toggles (4 × 5) ────────────────────────────────────────────
    for tog in toggles:
        tgx, tgy = float(tog.x), float(tog.y)
        obs[ptr + 0] = (tgx - rx) / FIELD_WIDTH
        obs[ptr + 1] = (tgy - ry) / FIELD_HEIGHT
        obs[ptr + 2] = _dist(rx, ry, tgx, tgy)
        obs[ptr + 3] = 1.0 if tog.owner == robot.alliance else 0.0
        opp = "blue" if robot.alliance == "red" else "red"
        obs[ptr + 4] = 1.0 if tog.owner == opp else 0.0
        ptr += 5

    # ── [534:554] Global match state (20) ────────────────────────────────────
    tr = float(simulator.time_remaining)
    te = float(simulator.time_elapsed)
    full = float(TOTAL_SECONDS)
    is_auto = 1.0 if simulator.match_phase == "autonomous" else 0.0
    is_drv  = 1.0 if simulator.match_phase == "driver"    else 0.0
    is_end  = 1.0 if rules_engine.endgame_active          else 0.0

    obs[ptr + 0]  = max(0.0, tr) / full
    obs[ptr + 1]  = min(te, full) / full
    obs[ptr + 2]  = is_auto
    obs[ptr + 3]  = is_drv
    obs[ptr + 4]  = is_end
    obs[ptr + 5]  = rules_engine.red_score  / 200.0
    obs[ptr + 6]  = rules_engine.blue_score / 200.0
    obs[ptr + 7]  = (rules_engine.red_score - rules_engine.blue_score) / 200.0
    obs[ptr + 8]  = rules_engine.midfield_red_count  / 2.0
    obs[ptr + 9]  = rules_engine.midfield_blue_count / 2.0
    obs[ptr + 10] = simulator.red_cups_left  / 10.0
    obs[ptr + 11] = simulator.blue_cups_left / 10.0
    obs[ptr + 12] = simulator.red_alliance_pins_left  / 12.0
    obs[ptr + 13] = simulator.blue_alliance_pins_left / 12.0
    obs[ptr + 14] = 1.0 if simulator.match_over else 0.0
    # Time until endgame starts (0.0 once in endgame)
    drv_remaining = tr if simulator.match_phase == "driver" else 0.0
    time_to_endgame = max(0.0, drv_remaining - ENDGAME_SECONDS) / 105.0
    obs[ptr + 15] = time_to_endgame
    obs[ptr + 16] = 1.0 if simulator.timer_started else 0.0

    # ── v7 features (3) ─────────────────────────────────────────────────
    # (a) Alliance-relative score delta in [-1, 1].  80-pt swing saturates.
    my_score  = rules_engine.red_score  if robot.alliance == "red" else rules_engine.blue_score
    opp_score = rules_engine.blue_score if robot.alliance == "red" else rules_engine.red_score
    obs[ptr + 17] = max(-1.0, min(1.0, (my_score - opp_score) / 80.0))

    # (b) Heading-vs-velocity cosine alignment for self in [-1, 1].
    # Robots spinning in place have low |speed|, returning 0 here.  Robots
    # driving forward along their heading return ~1; backing along heading: ~-1.
    speed = math.hypot(rvx, rvy)
    if speed > 1.0:
        # heading unit vector
        hx, hy = math.cos(ra), math.sin(ra)
        obs[ptr + 18] = (rvx * hx + rvy * hy) / speed
    else:
        obs[ptr + 18] = 0.0

    # (c) Endgame urgency ramp: 0 outside endgame; rises 0→1 linearly across
    # the final ENDGAME_RAMP_SECONDS of the match.  Lets the policy condition
    # on "park NOW" vs "park later" without inferring it from raw time_remaining.
    if is_end and tr <= ENDGAME_RAMP_SECONDS:
        obs[ptr + 19] = max(0.0, min(1.0, (ENDGAME_RAMP_SECONDS - tr) / ENDGAME_RAMP_SECONDS))
    else:
        obs[ptr + 19] = 0.0
    ptr += 20

    # ── v8.1 self-awareness + defensive intel (10) ──────────────────────
    # (a) Own carry-step counter normalized to TIME_TO_SCORE_TARGET — lets
    # the policy reason about "I've been carrying too long, score now or
    # cycle".  Reads `robot._carry_steps` set by env_wrapper.step();
    # defaults to 0 when not set (e.g. PettingZoo wrapper).
    own_carry = float(getattr(robot, "_carry_steps", 0))
    obs[ptr + 0] = min(1.0, own_carry / float(TIME_TO_SCORE_TARGET))

    # (b) Holding-penalty overshoot ratio: 0 below HOLDING_TIMEOUT_STEPS,
    # rising linearly to 1 over the next HOLDING_RAMP_STEPS.  Anticipates
    # the quadratic timeout penalty before it bites hard.
    overshoot = max(0.0, own_carry - float(HOLDING_TIMEOUT_STEPS))
    obs[ptr + 1] = min(1.0, overshoot / float(HOLDING_RAMP_STEPS))

    # (c–d) Per-opponent carrying pin UP-colour one-hot.  Defensive intel:
    # if the nearest opponent has a yellow pin, blocking their goal is
    # high-value.  Empty pin slot → all zeros.
    opp_color_slots = [0.0] * 6   # [r1,b1,y1, r2,b2,y2]
    for i, opp in enumerate(opponents[:2]):
        if opp.carrying_pin is not None:
            up_oh = _color_onehot(opp.carrying_pin.get_up_color())
            opp_color_slots[i * 3 + 0] = up_oh[0]
            opp_color_slots[i * 3 + 1] = up_oh[1]
            opp_color_slots[i * 3 + 2] = up_oh[2]
    for j in range(6):
        obs[ptr + 2 + j] = opp_color_slots[j]

    # (e) Yellow pins remaining (unscored, not carried) normalised by /12.
    # Resource-awareness signal for "should I commit to yellow strategy".
    yellow_left = sum(
        1 for p in pins
        if not p.scored and p.carried_by is None and
        (p.get_up_color() == C_YELLOW or p.get_down_color() == C_YELLOW)
    )
    obs[ptr + 8] = min(1.0, yellow_left / 12.0)

    # (f) Can-score-anywhere bit: 1 if this robot can legally extend the
    # stack at any reachable goal right now, else 0.  Makes the legality
    # signal explicit rather than requiring inference across goal slots.
    can_anywhere = 0.0
    if robot.carrying_pin is not None or robot.carrying_cup is not None:
        for g in goals:
            if g.alliance not in ("neutral", robot.alliance):
                continue
            top_is_pin = bool(g.stack) and bool(g.stack[-1][1])
            can_pin = robot.carrying_pin is not None and not top_is_pin
            can_cup = robot.carrying_cup is not None and     top_is_pin
            if can_pin or can_cup:
                can_anywhere = 1.0
                break
    obs[ptr + 9] = can_anywhere
    ptr += 10

    assert ptr == OBS_DIM, f"Obs dim mismatch: {ptr} != {OBS_DIM}"
    return obs


def build_all_observations(simulator) -> dict:
    """
    Build observations for all four robots at once.
    Returns {robot_id: np.ndarray(564,)} dict.
    """
    return {
        robot.robot_id: build_observation(
            robot=robot,
            all_robots=simulator.robots,
            pins=simulator.pins,
            cups=simulator.cups,
            goals=simulator.goals,
            toggles=simulator.toggles,
            rules_engine=simulator.rules_engine,
            simulator=simulator,
        )
        for robot in simulator.robots
    }


def get_action_mask(robot, goals) -> np.ndarray:
    """
    Returns a boolean mask (7,) for the discrete action head.
    True = action is LEGAL (not masked out).

    Actions: [intake, score_pin, score_cup, toggle, flip_pin, flip_cup, match_load]

    Note: the `rules_engine` parameter was removed in v8.2 (PROBLEM 52).
    Legality is determined entirely by robot state and goal stacks; no
    rules-engine state (endgame_active, scores, etc.) gates any action.
    """
    mask = np.ones(7, dtype=bool)

    # intake — only useful if at least one slot is free
    if robot.carrying_pin is not None and robot.carrying_cup is not None:
        mask[0] = False   # both slots full
    if robot.intake_cooldown > 0:
        mask[0] = False

    # score_pin — need a pin and a valid nearby goal with legal top
    if robot.carrying_pin is None:
        mask[1] = False
    else:
        rx, ry = robot.body.position
        can = any(
            math.hypot(rx - g.x, ry - g.y) <= SCORING_RADIUS and
            (g.alliance == "neutral" or g.alliance == robot.alliance) and
            (not g.stack or not g.stack[-1][1])
            for g in goals
        )
        if not can:
            mask[1] = False

    # score_cup — need a cup and a goal whose top is a pin
    if robot.carrying_cup is None:
        mask[2] = False
    else:
        rx, ry = robot.body.position
        can = any(
            math.hypot(rx - g.x, ry - g.y) <= SCORING_RADIUS and
            (g.alliance == "neutral" or g.alliance == robot.alliance) and
            g.stack and g.stack[-1][1]   # top is pin
            for g in goals
        )
        if not can:
            mask[2] = False

    # flip_pin — need a carried pin
    if robot.carrying_pin is None:
        mask[4] = False

    # flip_cup — need a carried cup
    if robot.carrying_cup is None:
        mask[5] = False

    return mask
