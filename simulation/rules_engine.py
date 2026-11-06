"""
simulation/rules_engine.py
VEX Override scoring + rules per the official game manual v2.0.
- Each visible red/blue pin half = 5, each visible Owned yellow half = 10
- Cups score nothing; there is no stack bonus
- Yellow ownership: Quadrant toggle (SC5a) or Midfield robot majority (SC5b)
- +8 per robot ending the match in the Midfield (SC6), end of match only
- Autonomous Bonus 12 (tie: 6 each), Autonomous Win Point check (SC8)
- Robot Skills scoring restrictions (RSC3)
The per-goal math lives in simulation/scoring.py (shared with strategy/).
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
    VERBOSE = True   # print the [SC5b] diagnostic at match end

    def __init__(self, skills_mode: bool = False):
        # Skills mode = single-robot solo challenge (no alliance opposition).
        # When True, recompute_all_scores folds blue-side scoring into the
        # red total so the entire field counts as the player's score.
        self.skills_mode = skills_mode
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
        # Autonomous bonus (+12 pts) is set once when auton ends and must
        # survive every subsequent recompute_all_scores call — recompute
        # rebuilds totals from goal stacks, so the bonus is stored here
        # and added back in each frame.
        self.auton_bonus_red      = 0
        self.auton_bonus_blue     = 0

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
        self.auton_bonus_red     = 0
        self.auton_bonus_blue    = 0

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
    def robot_outline(robot, n_per_edge: int = 4):
        """Sample points on the robot's rectangular footprint."""
        a = robot.body.angle
        c, s_ = math.cos(a), math.sin(a)
        px, py = robot.body.position
        hl = getattr(robot, "robot_length", 15.0) / 2.0
        hw = getattr(robot, "robot_width", 15.0) / 2.0
        corners = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
        pts = [(px, py)]
        for i in range(4):
            (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 4]
            for k in range(n_per_edge):
                t = k / n_per_edge
                lx, ly = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                pts.append((px + lx * c - ly * s_, py + lx * s_ + ly * c))
        return pts

    @staticmethod
    def is_robot_in_midfield(robot) -> bool:
        """SC6: at least partially within the Midfield (the tape square,
        a diamond |dx|+|dy| <= 24 around the field centre)."""
        mc_x, mc_y = MIDFIELD_CENTER
        for x, y in RulesEngine.robot_outline(robot):
            if abs(x - mc_x) + abs(y - mc_y) <= MIDFIELD_HALF:
                return True
        return False

    @staticmethod
    def robot_touches_perimeter(robot, tol: float = 0.75) -> bool:
        from config.game_rules import FIELD_WIDTH, FIELD_HEIGHT
        for x, y in RulesEngine.robot_outline(robot, 2):
            if x <= tol or y <= tol or x >= FIELD_WIDTH - tol or y >= FIELD_HEIGHT - tol:
                return True
        return False

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
            goal.get_score(toggles, skills=self.skills_mode)
            red  += goal.red_score
            blue += goal.blue_score

        # ── Midfield robots (SC6) ──────────────────────────────────────
        # Counted ONLY at match end (calculate_final_score).  Live counts are
        # kept for the HUD / bots but NOT added to the score.  (The old sim
        # added +8/robot live during the last 20 s.)
        if robots:
            self._update_midfield_counts(robots)

        # Persist the auton bonus (+12 to the auton winner) across frames.
        red  += self.auton_bonus_red
        blue += self.auton_bonus_blue

        # Skills mode: there is no opposing alliance.  Every visible pin half
        # on the field — red, blue, or owned-yellow — counts toward the
        # player's single score.  Fold blue into red and zero blue out.
        if self.skills_mode:
            self.red_score  = red + blue
            self.blue_score = 0
            return

        self.red_score  = red
        self.blue_score = blue

    # ── Final score (match end) ───────────────────────────────────────
    def calculate_final_score(self, goals: list, toggles: list,
                              robots: list) -> Dict:
        """Freeze the final score, applying SC5b at the end-of-match instant.

        SC5b — Center goal yellow halves:
          During live play the center goal's yellow halves contribute 0
          (see FieldGoal.get_score with midfield_majority=None).  Here, at
          match end, we count robots inside the Midfield right now and:
            - strict majority → all visible yellow halves in the center
              goal score 10 pts each to that alliance
            - tie  (0-0, 1-1, 2-2) → yellow halves remain 0 (unclaimed)
          Regular red/blue halves in the center goal were already scored
          live and are not affected.
        """
        # Base: goal stacks with toggle ownership (no Midfield yet).
        self.recompute_all_scores(goals, toggles, robots)
        # Recount robots in midfield at the exact moment the match ends.
        # We do NOT rely on self.midfield_*_count because those are reset to
        # 0 outside the endgame window in recompute_all_scores; reading them
        # here would misclassify the majority if the endgame flag was not
        # active on the final frame.
        self._update_midfield_counts(robots)
        r_count = self.midfield_red_count
        b_count = self.midfield_blue_count
        if self.skills_mode:
            # RSC3a.iii.2: Midfield yellows are Owned if the robot ends there.
            majority = "red" if r_count > 0 else "tie"
        elif r_count > b_count:
            majority = "red"
        elif b_count > r_count:
            majority = "blue"
        else:
            majority = "tie"

        # Recompute the center goal score using the SC5b majority, regardless
        # of whether the live score already had yellows credited (it shouldn't,
        # since live yellows are deferred — but recomputing is the canonical
        # source of truth).
        center_goal = next((g for g in goals if g.goal_id == CENTER_GOAL_ID), None)
        delta_r = delta_b = 0
        n_yellow_visible = 0
        if center_goal:
            old_r = center_goal.red_score
            old_b = center_goal.blue_score
            center_goal.get_score(toggles, midfield_majority=majority,
                                  skills=self.skills_mode)
            delta_r = center_goal.red_score - old_r
            delta_b = center_goal.blue_score - old_b
            if self.skills_mode:
                # No opposing alliance — any SC5b credit goes to the player.
                self.red_score  += delta_r + delta_b
                # blue_score stays 0
            else:
                self.red_score  += delta_r
                self.blue_score += delta_b
            # v9.3 PROBLEM 71: count visible yellow halves in the center goal
            # at match end for the diagnostic line below.
            n = len(center_goal.stack)
            for i, (obj, is_pin) in enumerate(center_goal.stack):
                if not is_pin:
                    continue
                # DOWN visibility
                if i == 0:
                    down_vis = True
                else:
                    prev_obj, prev_is_pin = center_goal.stack[i - 1]
                    if prev_is_pin:
                        down_vis = True
                    else:
                        flipped = getattr(prev_obj, 'flipped', False)
                        eff_clear_up = (not prev_obj.clear_on_top) if flipped else prev_obj.clear_on_top
                        down_vis = eff_clear_up
                # UP visibility
                if i + 1 >= n:
                    up_vis = True
                else:
                    next_obj, next_is_pin = center_goal.stack[i + 1]
                    if next_is_pin:
                        up_vis = True
                    else:
                        flipped = getattr(next_obj, 'flipped', False)
                        eff_clear_up = (not next_obj.clear_on_top) if flipped else next_obj.clear_on_top
                        up_vis = not eff_clear_up
                # Count yellow halves — use name properties (strings), not RGB tuples
                if down_vis and obj.down_half_name == "yellow":
                    n_yellow_visible += 1
                if up_vis and obj.up_half_name == "yellow":
                    n_yellow_visible += 1

        # v9.3 PROBLEM 71: emit a diagnostic line so the SC5b adjustment is
        # visible in match-end output.  This makes future "yellow not credited"
        # bugs immediately obvious without needing to instrument anything.
        # SC6: +8 per robot at least partially in the Midfield at the end.
        self.red_score  += self.midfield_red_bonus
        if self.skills_mode:
            self.red_score += self.midfield_blue_bonus
        else:
            self.blue_score += self.midfield_blue_bonus
        if RulesEngine.VERBOSE:
            print(f"[SC5b] midfield: red={r_count} blue={b_count} -> majority={majority} | "
                  f"center_yellow_visible={n_yellow_visible} | "
                  f"+pts: red={delta_r} blue={delta_b} | "
                  f"park +{self.midfield_red_bonus}/+{self.midfield_blue_bonus}")

        return {
            "red":  self.red_score,
            "blue": self.blue_score,
            "winner": "red"  if self.red_score  > self.blue_score else
                      "blue" if self.blue_score > self.red_score  else "tie",
            "midfield_majority":    majority,
            "midfield_red_count":   r_count,
            "midfield_blue_count":  b_count,
        }

    # ── Autonomous Win Point (SC8) ─────────────────────────────────────
    def autonomous_win_point(self, goals: list, toggles: list, robots: list,
                             alliance: str, worlds: bool = False) -> Dict:
        """Evaluate SC8 for `alliance` at the end of the Autonomous Period.
        "Pins Scored" = Scored pin halves, only counting goals in Quadrants on
        the opposing side of the Autonomous Line are excluded.  Midfield goal
        alliance-colour halves count; Midfield yellows are not Owned in
        auton (SC7a)."""
        from config.game_rules import (AWP_MIN_PINS, AWP_MIN_GOALS_WITH_2,
                                       AWP_MIN_PINS_WORLDS, AWP_MIN_GOALS_WITH_2_WORLDS)
        from simulation.scoring import scored_halves, QUADRANT_COLOR, TOGGLE_OF_QUADRANT
        need_pins = AWP_MIN_PINS_WORLDS if worlds else AWP_MIN_PINS
        need_goals = AWP_MIN_GOALS_WITH_2_WORLDS if worlds else AWP_MIN_GOALS_WITH_2
        owners = {t.toggle_id: t.owner for t in toggles}
        total = 0
        goals_with_2 = 0
        for g in goals:
            if g.region != "midfield" and QUADRANT_COLOR[g.region] != alliance:
                continue          # opposing side of the Autonomous Line
            owner = None
            if g.region != "midfield":
                owner = owners.get(TOGGLE_OF_QUADRANT[g.region])
                owner = owner if owner in ("red", "blue") else None
            n = scored_halves(g.stack_desc(), g.region, owner, alliance)
            total += n
            if n >= 2:
                goals_with_2 += 1
        no_perimeter = not any(self.robot_touches_perimeter(r) for r in robots
                               if r.alliance == alliance)
        return {"pins": total, "goals_with_2": goals_with_2,
                "no_perimeter_contact": no_perimeter,
                "awp": total >= need_pins and goals_with_2 >= need_goals and no_perimeter}
