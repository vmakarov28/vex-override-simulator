"""
simulation/rules_engine.py
Full VEX Override scoring + rules per the official game manual.
- Cups = 0 points alone
- Pins = 5 points when placed
- Stack bonus only for proper nesting (pin → cup → pin → cup ...)
- Toggle control applies yellow pin bonus to the 2 closest goals
"""

from config.game_rules import (
    FOULS, FOUL_PINNING_SECONDS, FOUL_STANDARD_PTS,
    POINTS_MIDFIELD_PARK, POINTS_AUTONOMOUS_BONUS,
    MIDFIELD_CENTER, MIDFIELD_HALF,
    POINTS_PIN_IN_GOAL, POINTS_CUP_IN_GOAL, POINTS_STACK_BONUS, POINTS_YELLOW_OWNED,
    ENDGAME_SECONDS, CENTER_GOAL_ID, ENDGAME_CENTER_MAX_STACK,
)
from simulation.game_objects import GamePin, GameCup
from typing import Dict, List, Optional
import math


def get_controlling_toggle(x: float, y: float) -> int:
    """
    Returns which toggle controls this position based on the diagonal tape lines.
    1 = Left toggle (controls LEFT area)
    2 = Right toggle (controls RIGHT area)
    3 = Top toggle (controls TOP area)
    4 = Bottom toggle (controls BOTTOM area)
    """
    dx = x - 72
    dy = y - 72

    if dx + dy <= 0 and dx - dy <= 0:
        return 1      # Top triangle → Top toggle (3)
    elif dx + dy <= 0 and dx - dy >= 0:
        return 3      # Right triangle → Right toggle (2)
    elif dx + dy >= 0 and dx - dy >= 0:
        return 2      # Bottom triangle → Bottom toggle (4)
    else:
        return 4      # Left triangle → Left toggle (1)


class RulesEngine:
    def __init__(self):
        self.red_score  = 0
        self.blue_score = 0
        self.fouls: Dict[str, list] = {"red": [], "blue": []}
        self.pin_timers: Dict[str, float] = {}
        self.toggle_state: Dict[int, Optional[str]] = {}
        # Endgame state — updated by simulator.step() every frame
        self.endgame_active       = False
        self.midfield_red_count   = 0    # robots in midfield this frame
        self.midfield_blue_count  = 0
        self.midfield_red_bonus   = 0    # live +8-per-robot parking pts
        self.midfield_blue_bonus  = 0

    def reset(self):
        self.red_score  = 0
        self.blue_score = 0
        self.fouls = {"red": [], "blue": []}
        self.pin_timers = {}
        self.toggle_state = {}
        self.endgame_active      = False
        self.midfield_red_count  = 0
        self.midfield_blue_count = 0
        self.midfield_red_bonus  = 0
        self.midfield_blue_bonus = 0

    def check_possession(self, robot):
        """Enforce 1 Pin + 1 Cup possession limit."""
        pins = 1 if robot.carrying_pin else 0
        cups = 1 if robot.carrying_cup else 0
        if pins > 1 or cups > 1:
            self.apply_foul(robot.alliance, "illegal_possession")

    def apply_foul(self, alliance: str, foul_type: str):
        info = FOULS.get(foul_type, {"points": FOUL_STANDARD_PTS, "description": foul_type})
        self.fouls[alliance].append(foul_type)
        if alliance == "red": self.blue_score += info["points"]
        else:                 self.red_score  += info["points"]

    def update_yellow_pin_ownership(self, pins: list, toggles: list):
        for pin in pins:
            if not pin.is_yellow:
                pin.is_yellow_owned = False
                pin.owned_by = None
                continue
            region = get_controlling_toggle(pin.body.position.x, pin.body.position.y)
            controlling_toggle = None
            for t in toggles:
                if t.toggle_id == region:
                    controlling_toggle = t
                    break
            if controlling_toggle and controlling_toggle.owner in ["red", "blue"]:
                pin.is_yellow_owned = True
                pin.owned_by = controlling_toggle.owner
            else:
                pin.is_yellow_owned = False
                pin.owned_by = None

    # ── Midfield helpers ──────────────────────────────────────────────
    @staticmethod
    def is_robot_in_midfield(robot) -> bool:
        """True if any part of the robot is within the Midfield plane.
        Uses L1 (Manhattan) distance which matches the diamond boundary,
        plus a 10-inch buffer for robot physical size (SC6: 'any part').
        """
        mc_x, mc_y = MIDFIELD_CENTER
        rx, ry = robot.body.position
        return abs(rx - mc_x) + abs(ry - mc_y) <= MIDFIELD_HALF + 10.0

    def _update_midfield_counts(self, robots: list):
        """Count robots in midfield and compute live parking bonus."""
        r = b = 0
        for robot in robots:
            if self.is_robot_in_midfield(robot):
                if robot.alliance == "red": r += 1
                else:                        b += 1
        self.midfield_red_count  = r
        self.midfield_blue_count = b
        self.midfield_red_bonus  = r * POINTS_MIDFIELD_PARK
        self.midfield_blue_bonus = b * POINTS_MIDFIELD_PARK

    # ── Stack placement validation ────────────────────────────────────
    def process_scored_object(self, goal, obj, alliance: str):
        """Validate stack legality and add obj to goal.stack.
        Points are NOT accumulated here — scoring is fully live via
        recompute_all_scores every frame.
        """
        if goal.stack:
            last_obj, last_is_pin = goal.stack[-1]
            if isinstance(obj, GamePin) and last_is_pin:
                return 0   # pin on pin → illegal
            if isinstance(obj, GameCup) and not last_is_pin:
                return 0   # cup on cup → illegal
        else:
            if isinstance(obj, GameCup):
                return 0   # cup on empty goal → illegal
        goal.add_to_stack(obj)
        return 0

    # ── Live score recomputation ──────────────────────────────────────
    def recompute_all_scores(self, goals: list, toggles: list,
                             robots: list = None):
        """Recompute red/blue totals from all goal stacks plus midfield
        parking bonus.  Called every frame.

        IMPORTANT — what changes live vs. what is fixed at match end:
          LIVE:  goal stack scoring (all goals, toggle-based yellow ownership)
          LIVE:  +8 parking per robot in Midfield (endgame only)
          FIXED: SC5b — center goal yellow ownership by robot majority
                 This is evaluated ONLY at match end in calculate_final_score
                 to avoid wild score swings whenever a robot briefly enters
                 or exits the Midfield during play.
        """
        red  = 0
        blue = 0

        # ── Goal stack scoring (toggle-based, stable) ─────────────────
        for goal in goals:
            goal.get_score(toggles)   # updates goal.red_score / goal.blue_score
            red  += goal.red_score
            blue += goal.blue_score

        # ── Live midfield parking bonus (endgame only) ────────────────
        # Only the +8-per-robot bonus updates live.  Yellow pin ownership
        # for the center goal (SC5b) is NOT applied here — it locks in at
        # match end to prevent confusing score swings.
        if self.endgame_active and robots:
            self._update_midfield_counts(robots)
            red  += self.midfield_red_bonus
            blue += self.midfield_blue_bonus
        else:
            self.midfield_red_count  = 0
            self.midfield_blue_count = 0
            self.midfield_red_bonus  = 0
            self.midfield_blue_bonus = 0

        self.red_score  = red
        self.blue_score = blue

    # ── Final score (match end) ───────────────────────────────────────
    def calculate_final_score(self, goals: list, toggles: list,
                              robots: list) -> Dict:
        """Freeze the final score.

        At match end we apply SC5b: yellow pins in the center goal are
        owned by whichever alliance has STRICTLY more robots in the Midfield.
        This is done as a one-time adjustment on top of the already-computed
        toggle-based score, so it can't cause live swings.
        """
        # The live score already includes: goal stacks + parking bonus.
        # Apply SC5b adjustment for center goal yellow pins.
        center_goal = next((g for g in goals if g.goal_id == CENTER_GOAL_ID), None)
        if center_goal:
            r_count = self.midfield_red_count
            b_count = self.midfield_blue_count
            if r_count != b_count:
                majority = "red" if r_count > b_count else "blue"
                # Recompute center goal score with SC5b majority
                old_r = center_goal.red_score
                old_b = center_goal.blue_score
                center_goal.get_score(toggles, midfield_majority=majority)
                # Replace the toggle-based center goal score with SC5b score
                self.red_score  += center_goal.red_score  - old_r
                self.blue_score += center_goal.blue_score - old_b

        return {
            "red":  self.red_score,
            "blue": self.blue_score,
            "winner": "red"  if self.red_score  > self.blue_score else
                      "blue" if self.blue_score > self.red_score  else "tie",
        }