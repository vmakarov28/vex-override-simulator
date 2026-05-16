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
)
from simulation.game_objects import GamePin, GameCup   # ← ADD THIS LINE
from typing import Dict, List, Optional
import math

class RulesEngine:
    def __init__(self):
        self.red_score = 0
        self.blue_score = 0
        self.fouls: Dict[str, list] = {"red": [], "blue": []}
        self.pin_timers: Dict[str, float] = {}
        self.toggle_state: Dict[int, Optional[str]] = {}

    def reset(self):
        self.red_score = 0
        self.blue_score = 0
        self.fouls = {"red": [], "blue": []}
        self.pin_timers = {}
        self.toggle_state = {}

    def check_possession(self, robot):
        """Enforce 1 Pin + 1 Cup possession limit."""
        pins = 1 if robot.carrying_pin else 0
        cups = 1 if robot.carrying_cup else 0
        if pins > 1 or cups > 1:
            self.apply_foul(robot.alliance, "illegal_possession")

    def apply_foul(self, alliance: str, foul_type: str):
        info = FOULS.get(foul_type, {"points": FOUL_STANDARD_PTS, "description": foul_type})
        self.fouls[alliance].append(foul_type)
        if alliance == "red":
            self.blue_score += info["points"]
        else:
            self.red_score += info["points"]

    def update_yellow_pin_ownership(self, pins: list, toggles: list):
        """Toggle controls yellow pins in its quadrant + applies bonus to 2 closest goals."""
        for pin in pins:
            if not pin.is_yellow:
                continue
            pin.is_yellow_owned = False
            for toggle in toggles:
                if toggle.owner is not None and toggle.quadrant == pin.color.split("_")[0]:
                    pin.is_yellow_owned = True
                    break

    def process_scored_object(self, goal, obj, alliance: str):
        """Central scoring logic per official manual.
        - Cups alone = 0 points and are NOT allowed as the first item in a goal.
        - Stack must always start with a pin.
        - Cup is only a container for stacking bonus.
        """
        pts = 0

        if isinstance(obj, GamePin):
            pts += POINTS_PIN_IN_GOAL
            if getattr(obj, 'is_yellow_owned', False):
                pts += POINTS_YELLOW_OWNED

        elif isinstance(obj, GameCup):
            # Empty cup = 0 points and cannot start a stack
            if obj.contains_pin is None:
                return 0   # ← prevents lone cup scoring

            pts += POINTS_CUP_IN_GOAL  # 0
            # Pin inside the cup
            if obj.contains_pin:
                pts += POINTS_PIN_IN_GOAL
                if getattr(obj.contains_pin, 'is_yellow_owned', False):
                    pts += POINTS_YELLOW_OWNED

        # Stack bonus (only when cup contains a pin)
        if isinstance(obj, GameCup) and obj.contains_pin:
            pts += POINTS_STACK_BONUS

        if alliance == "red":
            self.red_score += pts
        else:
            self.blue_score += pts

        return pts

    def calculate_final_score(self, robots) -> Dict:
        """End-game bonuses."""
        mc_x, mc_y = MIDFIELD_CENTER
        for robot in robots:
            rx, ry = robot.body.position
            if abs(rx - mc_x) < MIDFIELD_HALF and abs(ry - mc_y) < MIDFIELD_HALF:
                if robot.alliance == "red":
                    self.red_score += POINTS_MIDFIELD_PARK
                else:
                    self.blue_score += POINTS_MIDFIELD_PARK

        return {
            "red": self.red_score,
            "blue": self.blue_score,
            "winner": "red" if self.red_score > self.blue_score else
                      "blue" if self.blue_score > self.red_score else "tie",
        }