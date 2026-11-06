"""
simulation/simulator.py
────────────────────────────────────────────────────────────────────────────
Core simulator with PRODUCTION-GRADE Pymunk integration.
"""

import pygame
import pymunk
import numpy as np
import math
import time
from typing import List, Dict, Optional

from config.game_rules import (
    FIELD_WIDTH, FIELD_HEIGHT, RENDER_SCALE,
    SCREEN_W, SCREEN_H,
    AUTONOMOUS_SECONDS, DRIVER_SECONDS, SETTLE_SECONDS, TOTAL_SECONDS,
    SKILLS_SECONDS,
    MAX_ROBOT_SIZE_START, MIDFIELD_CENTER, MIDFIELD_HALF,
    GOALS, TOGGLES, ROBOT_STARTS,
    ENDGAME_SECONDS,
    POINTS_AUTONOMOUS_BONUS, POINTS_AUTONOMOUS_TIE,
    AUTON_LINE_HALF_WIDTH, PREPLACED_GOAL_PINS,
    MATCH_LOADS_CUPS, MATCH_LOADS_ALLIANCE_PINS, MATCH_LOADS_YELLOW_PINS,
    SKILLS_LOADS_CUPS, SKILLS_LOADS_RED_PINS, SKILLS_LOADS_BLUE_PINS,
    SKILLS_LOADS_YELLOW_PINS, MATCH_LOAD_TIME,
)
from config.robot_profile import DEFAULT_PROFILE

from simulation.robot import Robot
from simulation.game_objects import (
    GamePin, GameCup,
    create_field_objects, create_goals, create_toggles,
    FieldGoal, FieldToggle,
    COLL_TYPE_PIN, COLL_TYPE_CUP, COLL_TYPE_WALL, COLL_TYPE_GOAL,
    COLL_TYPE_ROBOT,
    CAT_WALL, CAT_GOAL, CAT_ROBOT, CAT_PIN, CAT_CUP,
    CAT_ROBOT_RED, CAT_ROBOT_BLUE,
    ORIENT_VERTICAL, ORIENT_HORIZONTAL,
    NORMAL_PIN_FILTER,
)

from simulation.rules_engine import RulesEngine

# ════════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
PHYSICS_HZ        = 240
PHYSICS_DT        = 1.0 / PHYSICS_HZ
# Max physics substeps per step() call.  This used to be 4 (= 1/60 s), so a
# bot stepping at CONTROL_DT = 1/20 s only got ONE THIRD of the match time
# to move — the match clock ran 3x faster than the physics.  32 substeps
# covers any dt up to 0.133 s.
MAX_SUBSTEPS      = 32
# Contact impulse needed to tip a standing pin/cup.  0.01 meant ANY touch
# (even a 1 in/s brush) tipped it over.  Measured robot→cup impulses: light
# brush < 1, slow push (~15 in/s) ~12, full-speed hit 100-300.
KNOCK_IMPULSE     = 3.0
SOLVER_ITERATIONS = 10              # 10 iters is stable for 2-D top-down physics

COLOR_BG       = (22, 28, 38)
COLOR_FIELD    = (52, 58, 68)
COLOR_TILE_LINE = (60, 66, 78)
COLOR_TAPE     = (240, 240, 245)
COLOR_MIDFIELD = (35, 90, 155)
COLOR_WALL     = (80, 88, 100)


class OverrideSimulator:
    def __init__(self, headless: bool = False, render_scale: float = RENDER_SCALE,
                 skills_mode: bool = False, robot_profiles: Optional[Dict] = None,
                 preplaced_pins: bool = True):
        self.headless     = headless
        self.render_scale = render_scale
        self.screen_w     = SCREEN_W
        self.screen_h     = SCREEN_H
        # ── Skills mode (VEX Robot Skills Challenge) ──────────────────
        # One robot, 60-second timer, no autonomous period, no opposing
        # alliance.  All on-field scoring counts toward a single score.
        self.skills_mode  = skills_mode
        # robot_id -> RobotProfile (default: config.robot_profile.DEFAULT_PROFILE)
        self.robot_profiles = robot_profiles or {}
        # Head-to-head matches start with a yellow pin on each neutral goal.
        self.preplaced_pins = preplaced_pins and not skills_mode
        self.auton_awp = {}
        self.timer_started = False
        self.time_remaining = SKILLS_SECONDS if skills_mode else AUTONOMOUS_SECONDS

        # Match Load Inventory (Preloads are separate — SG5)
        self._reset_inventory()

        self.match_load_mode = False
        self._last_m_press_red = False
        self._last_m_press_blue = False
        self._m1_pressed_red = False
        self._m2_pressed_red = False
        self._m3_pressed_red = False
        self._m4_pressed_red = False
        self._m5_pressed_red = False

        self._m1_pressed_blue = False
        self._m2_pressed_blue = False
        self._m3_pressed_blue = False
        self._m4_pressed_blue = False
        self._m5_pressed_blue = False

        self.space = pymunk.Space()
        self.space.gravity         = (0, 0)
        self.space.damping         = 0.1
        self.space.iterations      = SOLVER_ITERATIONS
        self.space.collision_slop  = 0.005
        self.space.collision_bias  = (1.0 - 0.05) ** PHYSICS_HZ

        self.robots: List[Robot] = []
        self.pins: List[GamePin] = []
        self.cups: List[GameCup] = []
        self.goals: List[FieldGoal] = []
        self.toggles: List[FieldToggle] = []
        self.rules_engine = RulesEngine(skills_mode=self.skills_mode)

        self.time_elapsed   = 0.0
        # Skills runs are all-driver (no autonomous split).
        self.match_phase    = "driver" if self.skills_mode else "autonomous"
        self.match_over     = False
        self.red_score      = 0
        self.blue_score     = 0
        self._physics_accum = 0.0

        self.screen = None; self.clock = None
        self.font_hud = None; self.font_med = None; self.font_sm = None
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("VEX Override Simulator — Neural Strategy Lab")
            self.clock = pygame.time.Clock()
            self.font_hud = pygame.font.SysFont("monospace", 26, bold=True)
            self.font_med = pygame.font.SysFont("monospace", 16)
            self.font_sm  = pygame.font.SysFont("monospace", 12)

        self._build_walls()
        self._auton_walls: List = []
        self._create_robots()
        # Skills mode has no autonomous phase, so the auton walls are never built.
        if not self.skills_mode:
            self._build_auton_walls()
        self._create_field_objects()
        self._register_collision_handlers()
        self._preload_robots()

        # Autonomous bonus tracking
        self._auton_bonus_applied = False

        self._field_bg = None

    def _build_walls(self):
        sb = self.space.static_body
        W, H, T = FIELD_WIDTH, FIELD_HEIGHT, 2.0
        walls = [
            pymunk.Segment(sb, (0, 0), (W, 0), T),
            pymunk.Segment(sb, (W, 0), (W, H), T),
            pymunk.Segment(sb, (W, H), (0, H), T),
            pymunk.Segment(sb, (0, H), (0, 0), T),
        ]
        wall_filter = pymunk.ShapeFilter(
            categories=CAT_WALL,
            mask=CAT_ROBOT | CAT_PIN | CAT_CUP,
        )
        for wall in walls:
            wall.friction       = 0.55
            wall.elasticity     = 0.10
            wall.collision_type = COLL_TYPE_WALL
            wall.filter         = wall_filter
        self.space.add(*walls)

    def _build_auton_walls(self):
        """SG7: during the Autonomous Period a robot may not touch tiles or
        objects on the OPPOSING side of the Autonomous Line.

        The Autonomous Line is the double tape from the top-left corner to the
        bottom-right corner (y = x) plus the tape square around the Midfield.
        Red side = y > x, blue side = y < x.  The line itself and the Midfield
        may be used by BOTH alliances (SG7b, SG8, glossary "Midfield").

        So each alliance gets its own invisible wall along the FAR edge of the
        tape band, which detours around the far side of the Midfield diamond.
        The wall's shape filter only hits that alliance's robots.  (The old
        sim sealed every robot in its own wedge and kept everyone out of the
        Midfield — far stricter than the manual.)
        """
        sb = self.space.static_body
        T = 0.5
        w = AUTON_LINE_HALF_WIDTH
        # Red robots are kept on y >= x - w.
        red_pts = [(w, 0.0), ((120 + w) / 2, (120 - w) / 2), (72.0, 48.0),
                   (96.0, 72.0), ((168 + w) / 2, (168 - w) / 2), (144.0, 144.0 - w)]
        # Blue robots are kept on y <= x + w (mirror across y = x).
        blue_pts = [(y, x) for (x, y) in red_pts]
        self._auton_walls = []
        for pts, cat in ((red_pts, CAT_ROBOT_RED), (blue_pts, CAT_ROBOT_BLUE)):
            filt = pymunk.ShapeFilter(categories=CAT_WALL, mask=cat)
            for p0, p1 in zip(pts[:-1], pts[1:]):
                seg = pymunk.Segment(sb, p0, p1, T)
                seg.friction = 0.55
                seg.elasticity = 0.05
                seg.collision_type = COLL_TYPE_WALL
                seg.filter = filt
                self.space.add(seg)
                self._auton_walls.append(seg)
        for robot in self.robots:
            robot.touch_allowed = self._auton_touch_fn(robot.alliance)

    @staticmethod
    def auton_side_ok(alliance: str, x: float, y: float) -> bool:
        """True if (x, y) may be touched by `alliance` during auton."""
        w = AUTON_LINE_HALF_WIDTH
        if abs(x - MIDFIELD_CENTER[0]) + abs(y - MIDFIELD_CENTER[1]) <= MIDFIELD_HALF:
            return True
        if alliance == "red":
            return y >= x - w
        return y <= x + w

    def _auton_touch_fn(self, alliance: str):
        def ok(x, y, _a=alliance):
            if self.match_phase != "autonomous":
                return True
            return OverrideSimulator.auton_side_ok(_a, x, y)
        return ok

    def _remove_auton_walls(self):
        """Called when autonomous ends so robots can roam freely in driver."""
        if not hasattr(self, "_auton_walls"):
            return
        for w in self._auton_walls:
            if w in self.space.shapes:
                self.space.remove(w)
        self._auton_walls = []
        for robot in getattr(self, "robots", []):
            robot.touch_allowed = None

    def _create_robots(self):
        def prof(rid):
            return self.robot_profiles.get(rid, DEFAULT_PROFILE)
        if self.skills_mode:
            # VEX Robot Skills: a single robot, starting in the Quadrant next
            # to the red Alliance Station (RSC2a) = red1's start.
            self.robots = [Robot("red1", "red", self.space, prof("red1"))]
            return
        order = ["red1", "red2", "blue1", "blue2"]
        alliances = {"red1": "red", "red2": "red", "blue1": "blue", "blue2": "blue"}
        self.robots = [Robot(rid, alliances[rid], self.space, prof(rid)) for rid in order]

    def _create_field_objects(self):
        self.pins, self.cups = create_field_objects(self.space)
        self.goals   = create_goals(self.space)
        self.toggles = create_toggles()
        self._preplace_goal_pins()

    def _preplace_goal_pins(self):
        """Figure FO-2: every neutral Goal starts with a yellow/yellow Pin
        Placed on it (head-to-head only; Skills goals start empty)."""
        if not self.preplaced_pins:
            return
        for goal in self.goals:
            colour = PREPLACED_GOAL_PINS.get(goal.goal_id)
            if not colour:
                continue
            pin = GamePin(8000 + goal.goal_id, colour, goal.x, goal.y, self.space,
                          orientation=ORIENT_VERTICAL)
            pin.scored = True
            pin.goal_id = goal.goal_id
            try:
                self.space.remove(pin.shape, pin.body)
            except Exception:
                pass
            goal.add_to_stack(pin)
            self.pins.append(pin)

    def _reset_inventory(self):
        if self.skills_mode:
            # RSC4b — everything comes through the red loaders
            self.red_cups_left = SKILLS_LOADS_CUPS
            self.red_alliance_pins_left = SKILLS_LOADS_RED_PINS
            self.red_blue_pins_left = SKILLS_LOADS_BLUE_PINS
            self.red_yellow_pins_left = SKILLS_LOADS_YELLOW_PINS
            self.blue_cups_left = 0
            self.blue_alliance_pins_left = 0
            self.blue_yellow_pins_left = 0
        else:
            self.red_cups_left = MATCH_LOADS_CUPS
            self.red_alliance_pins_left = MATCH_LOADS_ALLIANCE_PINS
            self.red_yellow_pins_left = MATCH_LOADS_YELLOW_PINS
            self.red_blue_pins_left = 0
            self.blue_cups_left = MATCH_LOADS_CUPS
            self.blue_alliance_pins_left = MATCH_LOADS_ALLIANCE_PINS
            self.blue_yellow_pins_left = MATCH_LOADS_YELLOW_PINS

    def _register_collision_handlers(self):
        pairs = [
            (COLL_TYPE_ROBOT, COLL_TYPE_PIN),
            (COLL_TYPE_ROBOT, COLL_TYPE_CUP),
            (COLL_TYPE_PIN,   COLL_TYPE_PIN),
            (COLL_TYPE_PIN,   COLL_TYPE_CUP),
            (COLL_TYPE_CUP,   COLL_TYPE_CUP),
            (COLL_TYPE_WALL,  COLL_TYPE_PIN),
            (COLL_TYPE_WALL,  COLL_TYPE_CUP),
            (COLL_TYPE_GOAL,  COLL_TYPE_PIN),
            (COLL_TYPE_GOAL,  COLL_TYPE_CUP),
        ]
        n_ok = 0
        for ta, tb in pairs:
            try:
                h = self.space.add_collision_handler(ta, tb)
                h.post_solve = self._post_solve_knock
                n_ok += 1
            except Exception:
                try:
                    self.space.on_collision(
                        ta, tb, post_solve=self._post_solve_knock_v7
                    )
                    n_ok += 1
                except Exception as e:
                    print(f"[Simulator] WARN: could not register {ta}↔{tb}: {e}")

    @staticmethod
    def _post_solve_knock(arbiter, space, data):
        OverrideSimulator._handle_arbiter(arbiter, space)

    @staticmethod
    def _post_solve_knock_v7(arbiter, space, data):
        OverrideSimulator._handle_arbiter(arbiter, space)

    @staticmethod
    def _handle_arbiter(arbiter, space):
        impulse = arbiter.total_impulse
        mag = impulse.length
        if mag < KNOCK_IMPULSE:
            return

        a, b = arbiter.shapes
        try:
            normal = arbiter.contact_point_set.normal
        except Exception:
            normal = arbiter.normal

        for shape, n_sign in ((a, -1.0), (b, +1.0)):
            obj = getattr(shape, "game_object", None)
            if obj is None or not hasattr(obj, "knock_over"):
                continue
            if getattr(obj, "carried_by", None) is not None:
                continue
            if getattr(obj, "scored", False):
                continue
            hit_dir = normal * n_sign
            hit_angle = math.atan2(hit_dir.y, hit_dir.x)
            obj.knock_over(hit_angle, mag)

    def step(self, dt: float, actions: Optional[List[Dict]] = None):
        if self.match_over:
            return

        for robot in self.robots:
            robot._dt = dt
        if actions:
            for i, robot in enumerate(self.robots):
                if i < len(actions) and actions[i]:
                    act = actions[i]
                    robot.apply_drive(float(act.get("left", 0)), float(act.get("right", 0)))

                    if act.get("flip_pin", False) and robot.carrying_pin:
                        if robot.profile.can_flip_pin and robot.mech_busy <= 0:
                            robot.carrying_pin.flip()
                            robot.mech_busy = robot.profile.flip_time
                    if act.get("flip_cup", False) and robot.carrying_cup:
                        if robot.profile.can_flip_cup and robot.mech_busy <= 0:
                            robot.carrying_cup.flip()
                            robot.mech_busy = robot.profile.flip_time
                    only = act.get("intake_only")
                    if act.get("intake", False):
                        robot.try_intake(self.pins, self.cups, only)
                    if act.get("intake_rear", False):
                        robot.try_intake_rear(self.pins, self.cups, only)
                    if act.get("score_pin", False):
                        robot.try_score_pin(self.goals, self.rules_engine)
                    if act.get("score_cup", False):
                        robot.try_score_cup(self.goals, self.rules_engine)
                    if act.get("toggle", False):
                        robot.try_toggle(self.toggles, skills=self.skills_mode)
                    else:
                        robot.toggle_progress = 0.0
                    if act.get("match_load", False):
                        self._try_bot_match_load(robot, act.get("match_load"), dt)
                    else:
                        robot._load_progress = 0.0
                else:
                    robot.apply_drive(0, 0)
        else:
            for robot in self.robots:
                robot.apply_drive(0, 0)

        self._physics_accum += dt
        steps = 0
        while self._physics_accum >= PHYSICS_DT and steps < MAX_SUBSTEPS:
            for robot in self.robots:
                robot.physics_substep()
                robot.update_carried_positions()
            self.space.step(PHYSICS_DT)
            self._physics_accum -= PHYSICS_DT
            steps += 1
        if self._physics_accum > PHYSICS_DT * MAX_SUBSTEPS:
            self._physics_accum = 0.0
        self.time_elapsed += dt

        for pin in self.pins: pin.update(dt)
        for cup in self.cups: cup.update(dt)

        for robot in self.robots:
            self.rules_engine.check_possession(robot)
        self.rules_engine.update_yellow_pin_ownership(self.pins, self.toggles)

        # Set endgame flag before scoring so the +8 bonus and central goal
        # height restriction apply immediately when time crosses 20s.
        # Endgame is a Head-to-Head concept (glossary); Skills has none.
        self.rules_engine.endgame_active = (
            not self.skills_mode and
            self.match_phase == "driver" and
            self.timer_started and
            self.time_remaining <= ENDGAME_SECONDS
        )

        # Live recomputation every frame: goal stacks (toggle-based, stable)
        # + live +8 parking bonus per robot in midfield (endgame only).
        # SC5b (center goal yellow ownership from robot majority) is NOT
        # applied here — it only locks in at match end to prevent swings.
        self.rules_engine.recompute_all_scores(self.goals, self.toggles, self.robots)

        self.red_score  = self.rules_engine.red_score
        self.blue_score = self.rules_engine.blue_score

        # === TIMER LOGIC (starts only on first input) ===
        if not self.timer_started:
            # Check if any player is giving input
            if actions:
                for act in actions:
                    if act and (act.get("left", 0) != 0 or act.get("right", 0) != 0 or
                                act.get("intake", False) or act.get("score_pin", False) or
                                act.get("score_cup", False) or act.get("toggle", False) or
                                act.get("flip_pin", False) or act.get("flip_cup", False)):
                        self.timer_started = True
                        break

        if self.timer_started:
            self.time_remaining -= dt

        self._update_phase()

    def _update_phase(self):
        if self.time_remaining <= 0:
            if self.match_phase == "autonomous":
                # Skills mode never enters this branch (it starts in "driver").
                self._end_autonomous()
                self.match_phase = "driver"
                self.time_remaining = DRIVER_SECONDS
            elif self.match_phase == "driver":
                if not self.match_over:
                    self.match_over = True
                    self.match_phase = "ended"
                    # Pass goals + toggles so calculate_final_score can apply
                    # SC5b (center goal yellow ownership from robot majority).
                    final = self.rules_engine.calculate_final_score(
                        self.goals, self.toggles, self.robots)
                    self.red_score  = final["red"]
                    self.blue_score = final["blue"]

    def _end_autonomous(self):
        """VEX V5 VRC end-of-autonomous routine:
          1. Remove the autonomous tape-line walls (driver-control is unrestricted).
          2. Determine the autonomous winner based on current scores.
          3. Award POINTS_AUTONOMOUS_BONUS (12 pts) to the winning alliance.
             Tie: no bonus awarded (matches official VRC ruling).
        """
        self._remove_auton_walls()
        if self._auton_bonus_applied:
            return
        self._auton_bonus_applied = True
        # Snapshot the current "auton score" so we can show the breakdown.
        self._auton_score_red  = self.rules_engine.red_score
        self._auton_score_blue = self.rules_engine.blue_score
        # Persist the bonus on the rules engine so recompute_all_scores
        # (which rebuilds totals every frame) keeps adding it in.
        if self._auton_score_red > self._auton_score_blue:
            self.rules_engine.auton_bonus_red = POINTS_AUTONOMOUS_BONUS
            self._auton_winner = "red"
        elif self._auton_score_blue > self._auton_score_red:
            self.rules_engine.auton_bonus_blue = POINTS_AUTONOMOUS_BONUS
            self._auton_winner = "blue"
        else:
            # SC7b: a tie (including 0-0) gives each alliance 6 points.
            self.rules_engine.auton_bonus_red = POINTS_AUTONOMOUS_TIE
            self.rules_engine.auton_bonus_blue = POINTS_AUTONOMOUS_TIE
            self._auton_winner = "tie"
        # SC8: Autonomous Win Point status for each alliance
        self.auton_awp = {
            a: self.rules_engine.autonomous_win_point(
                self.goals, self.toggles, self.robots, a)
            for a in ("red", "blue")}
        # Surface immediately so observers reading red_score/blue_score
        # between this call and the next recompute see the right value.
        self.rules_engine.red_score  += self.rules_engine.auton_bonus_red
        self.rules_engine.blue_score += self.rules_engine.auton_bonus_blue
        self.red_score  = self.rules_engine.red_score
        self.blue_score = self.rules_engine.blue_score
        if not RulesEngine.VERBOSE:
            return
        print(f"[AUTON] Autonomous ended: red={self._auton_score_red} "
              f"blue={self._auton_score_blue} winner={self._auton_winner} "
              f"(+{POINTS_AUTONOMOUS_BONUS} bonus to "
              f"{self._auton_winner if self._auton_winner != 'tie' else 'none'})")

    def _try_flip_toggle(self, robot):
        for toggle in self.toggles:
            if toggle.try_interact(robot):
                return True
        return False

    def render(self):
        if self.headless or self.screen is None:
            return
        if self._field_bg is None:
            self._field_bg = self._build_field_bg()
        self.screen.blit(self._field_bg, (0, 0))

        for toggle in self.toggles: 
            toggle.draw(self.screen)

        # === LOOSE FIELD ELEMENTS (pins & cups) ===
        for cup in self.cups:
            if cup.carried_by is None and not cup.scored: 
                cup.draw(self.screen)
        for pin in self.pins:
            if getattr(pin, 'is_nested', False):
                continue
            if pin.carried_by is None and not pin.scored: 
                pin.draw(self.screen)

        # === ROBOTS (now drawn on top of loose pins/cups) ===
        for robot in self.robots:  
            robot.draw(self.screen, self.font_sm)

        # === GOALS (drawn last so stacks are on top) ===
        for goal in self.goals:   
            goal.draw(self.screen, self.font_sm)

        self._draw_hud()
        pygame.display.flip()

    def _build_field_bg(self) -> pygame.Surface:
        sc = self.render_scale
        W = int(FIELD_WIDTH * sc); H = int(FIELD_HEIGHT * sc)
        surf = pygame.Surface((W, H))
        surf.fill(COLOR_FIELD)
        from config.game_rules import TILE_SIZE
        tile_px = int(TILE_SIZE * sc)
        for gx in range(0, W, tile_px):
            pygame.draw.line(surf, COLOR_TILE_LINE, (gx, 0), (gx, H), 1)
        for gy in range(0, H, tile_px):
            pygame.draw.line(surf, COLOR_TILE_LINE, (0, gy), (W, gy), 1)

        # Midfield diamond - ROTATED 90° COUNTER-CLOCKWISE
        mc_x, mc_y = MIDFIELD_CENTER
        mf_px = int(mc_x * sc); mf_py = int(mc_y * sc)
        mf_half = int(MIDFIELD_HALF * sc)

        # 90° CCW rotated points
        mf_top    = (mf_px - mf_half, mf_py)        # was left
        mf_right  = (mf_px, mf_py - mf_half)        # was top
        mf_bottom = (mf_px + mf_half, mf_py)        # was right
        mf_left   = (mf_px, mf_py + mf_half)        # was bottom

        # Centers of each side (for correct 45° lines)
        mf_top_center    = ((mf_top[0] + mf_right[0]) // 2, (mf_top[1] + mf_right[1]) // 2)
        mf_right_center  = ((mf_right[0] + mf_bottom[0]) // 2, (mf_right[1] + mf_bottom[1]) // 2)
        mf_bottom_center = ((mf_bottom[0] + mf_left[0]) // 2, (mf_bottom[1] + mf_left[1]) // 2)
        mf_left_center   = ((mf_left[0] + mf_top[0]) // 2, (mf_left[1] + mf_top[1]) // 2)

        pygame.draw.polygon(surf, COLOR_TAPE, [mf_top, mf_right, mf_bottom, mf_left], 3)

        tape_c = COLOR_TAPE
        fc_tl, fc_tr = (0, 0), (W, 0)
        fc_bl, fc_br = (0, H), (W, H)

        # Double white line (top-left ↔ bottom-right) — now correct sides
        pygame.draw.line(surf, tape_c, fc_tl, mf_top_center, 4)
        pygame.draw.line(surf, tape_c, mf_bottom_center, fc_br, 4)

        # Single white line (bottom-left ↔ top-right) — now correct sides
        pygame.draw.line(surf, tape_c, fc_bl, mf_left_center, 2)
        pygame.draw.line(surf, tape_c, mf_right_center, fc_tr, 2)

        # Alliance brackets
        bracket_size = int(12 * sc)
        def draw_bracket(s, x, y, flip_x, flip_y, color):
            dx = -1 if flip_x else 1; dy = -1 if flip_y else 1
            pygame.draw.line(s, color, (x, y), (x + dx * bracket_size, y), 2)
            pygame.draw.line(s, color, (x, y), (x, y + dy * bracket_size), 2)
        draw_bracket(surf, 0, int(72 * sc), False, False, (210, 80, 80))
        draw_bracket(surf, 0, int(72 * sc), False, True,  (210, 80, 80))
        draw_bracket(surf, W, int(72 * sc), True,  False, (80, 130, 230))
        draw_bracket(surf, W, int(72 * sc), True,  True,  (80, 130, 230))

        # === MATCH LOADING AREAS (Red = left, Blue = right) ===
        from config.game_rules import LOADING_ZONE_WIDTH, LOADING_ZONE_HEIGHT, RED_LOADING_ZONES, BLUE_LOADING_ZONES

        zone_w = int(LOADING_ZONE_WIDTH * sc)
        zone_h = int(LOADING_ZONE_HEIGHT * sc)

        red_color   = (210, 50, 50)
        blue_color  = (50, 100, 210)
        tape_width  = 3

        # First: Fill loading zones with field color to cover diagonal lines
        for zone in RED_LOADING_ZONES + BLUE_LOADING_ZONES:
            zx = int(zone["x"] * sc)
            zy = int(zone["y"] * sc)
            pygame.draw.rect(surf, COLOR_FIELD, (zx, zy, zone_w, zone_h))

        # Then: Draw red tape outlines (left side)
        for zone in RED_LOADING_ZONES:
            zx = int(zone["x"] * sc)
            zy = int(zone["y"] * sc)
            pygame.draw.rect(surf, red_color, (zx, zy, zone_w, zone_h), tape_width)

        # Then: Draw blue tape outlines (right side)
        for zone in BLUE_LOADING_ZONES:
            zx = int(zone["x"] * sc)
            zy = int(zone["y"] * sc)
            pygame.draw.rect(surf, blue_color, (zx, zy, zone_w, zone_h), tape_width)


        pygame.draw.rect(surf, COLOR_WALL, (0, 0, W, H), 4)
        return surf

    def _draw_hud(self):
        # === SCOREBOARD + TIMER (Top Right) ===
        if self.rules_engine.endgame_active:
            t_color  = (255, 210, 50)
            t_label  = f"ENDGAME  {max(0, int(self.time_remaining))}s"
        else:
            t_color = (200, 200, 210)
            t_label = f"{max(0, int(self.time_remaining))}s"
        timer_txt = self.font_sm.render(t_label, True, t_color)

        if self.skills_mode:
            # Single combined score — no opposing alliance.
            ss_label = self.font_hud.render(f"SCORE {self.red_score}",
                                            True, (255, 220, 90))
            self.screen.blit(ss_label,
                             (self.screen_w - ss_label.get_width() - 12, 8))
            self.screen.blit(timer_txt,
                             (self.screen_w - timer_txt.get_width() - 12,
                              8 + ss_label.get_height() + 4))
        else:
            rs = self.font_sm.render(f"RED {self.red_score}", True, (255, 100, 100))
            bs = self.font_sm.render(f"BLUE {self.blue_score}", True, (100, 170, 255))
            self.screen.blit(rs,        (self.screen_w - rs.get_width()        - 12, 8))
            self.screen.blit(bs,        (self.screen_w - bs.get_width()        - 12, 26))
            self.screen.blit(timer_txt, (self.screen_w - timer_txt.get_width() - 12, 44))

        # === MIDFIELD PARKING (endgame only) ===
        if self.rules_engine.endgame_active:
            re = self.rules_engine
            mf = self.font_sm.render(
                f"Midfield  R:{re.midfield_red_count}(+{re.midfield_red_bonus})"
                f"  B:{re.midfield_blue_count}(+{re.midfield_blue_bonus})",
                True, (255, 210, 50))
            self.screen.blit(mf, (self.screen_w - mf.get_width() - 12, 62))

        # === MATCH LOAD INVENTORY (Bottom) ===
        # Red (left side) — always shown; in skills mode this is the only one
        extra = f" {self.red_blue_pins_left}B" if self.skills_mode else ""
        red_inv = self.font_sm.render(
            f"Red: {self.red_cups_left}C {self.red_alliance_pins_left}P {self.red_yellow_pins_left}Y{extra}",
            True, (255, 150, 150)
        )
        self.screen.blit(red_inv, (8, self.screen_h - 20))

        # Blue (right side) — hidden in skills mode (no blue alliance)
        if not self.skills_mode:
            blue_inv = self.font_sm.render(
                f"Blue: {self.blue_cups_left}C {self.blue_alliance_pins_left}P {self.blue_yellow_pins_left}Y",
                True, (150, 180, 255)
            )
            self.screen.blit(blue_inv, (self.screen_w - blue_inv.get_width() - 8, self.screen_h - 20))

        # === GAME OVER OVERLAY ===
        if self.match_over:
            ov = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 140))
            self.screen.blit(ov, (0, 0))

            if self.skills_mode:
                # Skills score screen (no winner — solo challenge).
                title_surf = self.font_hud.render("SKILLS RUN COMPLETE",
                                                  True, (255, 220, 90))
                score_surf = self.font_hud.render(f"Score: {self.red_score}",
                                                  True, (255, 255, 255))
                tx = self.screen_w // 2 - title_surf.get_width() // 2
                ty = self.screen_h // 2 - 56
                sx = self.screen_w // 2 - score_surf.get_width() // 2
                sy = ty + title_surf.get_height() + 10
                box_w = max(title_surf.get_width(), score_surf.get_width()) + 64
                box_h = title_surf.get_height() + score_surf.get_height() + 36
                bx = self.screen_w // 2 - box_w // 2
                by = ty - 14
                pygame.draw.rect(self.screen, (18, 24, 35),
                                 (bx, by, box_w, box_h), border_radius=10)
                pygame.draw.rect(self.screen, (255, 220, 90),
                                 (bx, by, box_w, box_h), 2, border_radius=10)
                self.screen.blit(title_surf, (tx, ty))
                self.screen.blit(score_surf, (sx, sy))

                reset_txt = self.font_med.render("Press R to run again",
                                                 True, (200, 200, 210))
                self.screen.blit(reset_txt,
                                 (self.screen_w // 2 - reset_txt.get_width() // 2,
                                  sy + score_surf.get_height() + 14))
            else:
                winner = ("RED" if self.red_score > self.blue_score else
                          "BLUE" if self.blue_score > self.red_score else "TIE")
                w_color = (255, 100, 100) if winner == "RED" else (
                          (100, 170, 255) if winner == "BLUE" else (200, 200, 200))

                end_surf = self.font_hud.render(
                    f"{winner} WINS! {self.red_score} – {self.blue_score}",
                    True, w_color)
                ex = self.screen_w // 2 - end_surf.get_width() // 2
                ey = self.screen_h // 2 - 20

                pygame.draw.rect(self.screen, (18, 24, 35),
                                 (ex - 16, ey - 10, end_surf.get_width() + 32, 48),
                                 border_radius=8)
                self.screen.blit(end_surf, (ex, ey))

                reset_txt = self.font_med.render("Press R to reset",
                                                 True, (200, 200, 210))
                self.screen.blit(reset_txt,
                                 (self.screen_w // 2 - reset_txt.get_width() // 2,
                                  ey + 32))

    def handle_keyboard(self, keys) -> List[Optional[Dict]]:
        # Slot count matches len(self.robots): 1 in skills, 4 in a match.
        actions = [None] * len(self.robots)
        left = right = 0.0
        if keys[pygame.K_w]: left = right = 1.0
        if keys[pygame.K_s]: left = right = -1.0
        if keys[pygame.K_a]: left, right = -0.85, 0.85
        if keys[pygame.K_d]: left, right =  0.85, -0.85
        actions[0] = {
            "left": left, "right": right,
            "intake": bool(keys[pygame.K_e]),
            "intake_rear": bool(keys[pygame.K_g]),     # back intake (vertical pins)
            "score_pin": bool(keys[pygame.K_q] and not keys[pygame.K_LSHIFT]),
            "score_cup": bool(keys[pygame.K_q] and keys[pygame.K_LSHIFT]),
            "toggle": bool(keys[pygame.K_t]),
            "flip_pin": bool(keys[pygame.K_f] and not keys[pygame.K_LSHIFT]),
            "flip_cup": bool(keys[pygame.K_f] and keys[pygame.K_LSHIFT]),
        }
        # Blue1 arrow-key controls only exist in a 2v2 match.
        if not self.skills_mode and len(self.robots) > 2:
            left = right = 0.0
            if keys[pygame.K_UP]:    left = right = 1.0
            if keys[pygame.K_DOWN]:  left = right = -1.0
            if keys[pygame.K_LEFT]:  left, right = -0.85, 0.85
            if keys[pygame.K_RIGHT]: left, right =  0.85, -0.85
            actions[2] = {
                "left": left, "right": right,
                "intake": bool(keys[pygame.K_RSHIFT]),
                "score_pin": bool(keys[pygame.K_RCTRL]),
                "score_cup": bool(keys[pygame.K_RCTRL] and keys[pygame.K_LSHIFT]),
                "toggle": False,
                "flip_pin": bool(keys[pygame.K_LEFTBRACKET] and not keys[pygame.K_LSHIFT]),
                "flip_cup": bool(keys[pygame.K_LEFTBRACKET] and keys[pygame.K_LSHIFT]),
            }

        # === MATCH LOAD SYSTEM (M + Number) ===
        red1 = self.robots[0]
        blue1 = self.robots[2] if len(self.robots) > 2 else None

        # Red Match Load (DISABLED during autonomous per VRC rules)
        if self.match_phase == "autonomous":
            return actions
        if self._is_in_loading_zone(red1, "red"):
            if keys[pygame.K_m] and keys[pygame.K_1]:
                if not getattr(self, '_m1_pressed_red', False):
                    self._perform_match_load(red1, 1)
                    self._m1_pressed_red = True
            else:
                self._m1_pressed_red = False

            if keys[pygame.K_m] and keys[pygame.K_2]:
                if not getattr(self, '_m2_pressed_red', False):
                    self._perform_match_load(red1, 2)
                    self._m2_pressed_red = True
            else:
                self._m2_pressed_red = False

            if keys[pygame.K_m] and keys[pygame.K_3]:
                if not getattr(self, '_m3_pressed_red', False):
                    self._perform_match_load(red1, 3)
                    self._m3_pressed_red = True
            else:
                self._m3_pressed_red = False

            if keys[pygame.K_m] and keys[pygame.K_4]:
                if not getattr(self, '_m4_pressed_red', False):
                    self._perform_match_load(red1, 4)
                    self._m4_pressed_red = True
            else:
                self._m4_pressed_red = False

            if keys[pygame.K_m] and keys[pygame.K_5]:
                if not getattr(self, '_m5_pressed_red', False):
                    self._perform_match_load(red1, 5)
                    self._m5_pressed_red = True
            else:
                self._m5_pressed_red = False

        # Blue Match Load — only in 2v2 match mode (blue1 exists).
        if blue1 is not None and self._is_in_loading_zone(blue1, "blue"):
            if keys[pygame.K_m] and keys[pygame.K_1]:
                if not getattr(self, '_m1_pressed_blue', False):
                    self._perform_match_load(blue1, 1)
                    self._m1_pressed_blue = True
            else:
                self._m1_pressed_blue = False

            if keys[pygame.K_m] and keys[pygame.K_2]:
                if not getattr(self, '_m2_pressed_blue', False):
                    self._perform_match_load(blue1, 2)
                    self._m2_pressed_blue = True
            else:
                self._m2_pressed_blue = False

            if keys[pygame.K_m] and keys[pygame.K_3]:
                if not getattr(self, '_m3_pressed_blue', False):
                    self._perform_match_load(blue1, 3)
                    self._m3_pressed_blue = True
            else:
                self._m3_pressed_blue = False

            if keys[pygame.K_m] and keys[pygame.K_4]:
                if not getattr(self, '_m4_pressed_blue', False):
                    self._perform_match_load(blue1, 4)
                    self._m4_pressed_blue = True
            else:
                self._m4_pressed_blue = False

            if keys[pygame.K_m] and keys[pygame.K_5]:
                if not getattr(self, '_m5_pressed_blue', False):
                    self._perform_match_load(blue1, 5)
                    self._m5_pressed_blue = True
            else:
                self._m5_pressed_blue = False

        return actions

    def reset(self):
        self.time_elapsed   = 0.0
        # Skills runs are all-driver; matches start in autonomous.
        self.match_phase    = "driver" if self.skills_mode else "autonomous"
        self.match_over     = False
        self.red_score      = 0
        self.blue_score     = 0
        self.rules_engine   = RulesEngine(skills_mode=self.skills_mode)
        self._physics_accum = 0.0

        # === RESET TIMER ===
        self.timer_started = False
        self.time_remaining = SKILLS_SECONDS if self.skills_mode else AUTONOMOUS_SECONDS

        self._auton_bonus_applied = False
        self._auton_score_red  = 0
        self._auton_score_blue = 0
        self._auton_winner = None

        self._reset_inventory()
        self.auton_awp = {}
        self.match_load_mode = False

        for obj in self.pins + self.cups:
            try:
                if obj.shape and obj.shape in self.space.shapes:
                    self.space.remove(obj.shape)
                if obj.body and obj.body in self.space.bodies:
                    self.space.remove(obj.body)
            except Exception:
                pass

        for goal in self.goals:
            try:
                if hasattr(goal, 'phys_shape') and goal.phys_shape in self.space.shapes:
                    self.space.remove(goal.phys_shape)
                if hasattr(goal, 'phys_body') and goal.phys_body in self.space.bodies:
                    self.space.remove(goal.phys_body)
            except Exception:
                pass

        self.pins, self.cups = create_field_objects(self.space)
        self.goals   = create_goals(self.space)
        self.toggles = create_toggles()
        self._preplace_goal_pins()

        for robot in self.robots:
            robot.reset()
        if not self.skills_mode:
            self._remove_auton_walls()
            self._build_auton_walls()

        # Re-issue starting preloads so every robot enters auton with
        # the same pin + cup loadout each match.
        self._preload_robots()

        #print("[Simulator] Match reset.")
    def _is_in_loading_zone(self, robot, alliance):
        pos = robot.body.position
        if alliance == "red":
            return (pos.x < 12 and pos.y < 24) or (pos.x < 12 and pos.y > 120)
        else:
            return (pos.x > 132 and pos.y < 24) or (pos.x > 132 and pos.y > 120)

    def _try_bot_match_load(self, robot, choice=True, dt: float = 0.05) -> bool:
        """Bot-driven Match Load (SG11).

        Either robot of an alliance may take Match Loads from its own
        alliance's Loaders, during the Driver Controlled Period only (Skills:
        any time, red Loaders only).  The robot must sit in one of its Load
        Zones and hold the action for the robot profile's match_load_time;
        the element(s) are then handed straight into the robot's intake (the
        drive-team member drops them in the Loader, the robot pulls them out
        of the bottom opening).

        choice: True/"auto" or one of
          "cup+pin" (loaded cup with alliance pin), "cup+yellow", "cup+blue"
          (Skills), "cup", "pin", "yellow", "blue" (Skills).
        Returns True when a load completed this step.
        """
        if self.match_phase == "autonomous":
            return False
        alliance = robot.alliance
        if not self._is_in_loading_zone(robot, alliance):
            robot._load_progress = 0.0
            return False
        inv_alliance = alliance
        need_pin = robot.carrying_pin is None
        need_cup = robot.carrying_cup is None
        if not (need_pin or need_cup):
            return False

        def have(kind):
            if kind == "cup":
                return self._has_inventory(inv_alliance, "cup")
            if kind == "pin":
                return self._has_inventory(inv_alliance, "alliance_pin")
            if kind == "yellow":
                return self._has_inventory(inv_alliance, "yellow_pin")
            if kind == "blue":
                return self._has_inventory(inv_alliance, "blue_pin")
            return False

        if choice in (True, "auto", None):
            order = ["cup+yellow", "cup+pin", "cup+blue", "pin", "blue", "yellow", "cup"]
        else:
            order = [choice]
        pick = None
        for c in order:
            parts = c.split("+")
            ok = True
            for part in parts:
                if part == "cup" and not need_cup:
                    ok = False
                if part != "cup" and not need_pin:
                    ok = False
                if not have(part):
                    ok = False
            if ok:
                pick = parts
                break
        if pick is None:
            return False

        robot._load_progress = getattr(robot, "_load_progress", 0.0) + dt
        if robot._load_progress < robot.profile.match_load_time:
            return False
        robot._load_progress = 0.0
        x, y = robot.body.position
        for part in pick:
            if part == "cup":
                obj = GameCup(9500 + len(self.cups), "gray", x, y, self.space)
                self.cups.append(obj)
                self._use_inventory(inv_alliance, "cup")
                robot._grab(obj, "cup")
            else:
                colour = {"pin": "red_yellow" if alliance == "red" else "blue_yellow",
                          "yellow": "yellow_yellow", "blue": "blue_yellow"}[part]
                obj = GamePin(9500 + len(self.pins), colour, x, y, self.space)
                self.pins.append(obj)
                self._use_inventory(inv_alliance,
                                    {"pin": "alliance_pin", "yellow": "yellow_pin",
                                     "blue": "blue_pin"}[part])
                robot._grab(obj, "pin")
        return True

    def _perform_match_load(self, robot, selection):
        alliance = robot.alliance
        zone_x = 6 if alliance == "red" else 138
        zone_y = 12 if robot.body.position.y < 72 else 132

        if selection == 1 and self._has_inventory(alliance, "cup") and self._has_inventory(alliance, "alliance_pin"):
            self._spawn_loaded_cup(zone_x, zone_y, alliance, "alliance")
            self._use_inventory(alliance, "cup")
            self._use_inventory(alliance, "alliance_pin")
        elif selection == 2 and self._has_inventory(alliance, "cup") and self._has_inventory(alliance, "yellow_pin"):
            self._spawn_loaded_cup(zone_x, zone_y, alliance, "yellow")
            self._use_inventory(alliance, "cup")
            self._use_inventory(alliance, "yellow_pin")
        elif selection == 3 and self._has_inventory(alliance, "cup"):
            self._spawn_individual_cup(zone_x, zone_y)
            self._use_inventory(alliance, "cup")
        elif selection == 4 and self._has_inventory(alliance, "alliance_pin"):
            self._spawn_individual_pin(zone_x, zone_y, alliance)
            self._use_inventory(alliance, "alliance_pin")
        elif selection == 5 and self._has_inventory(alliance, "yellow_pin"):
            self._spawn_individual_pin(zone_x, zone_y, "yellow")
            self._use_inventory(alliance, "yellow_pin")

    def _has_inventory(self, alliance, item_type):
        if alliance == "red":
            if item_type == "cup": return self.red_cups_left > 0
            if item_type == "alliance_pin": return self.red_alliance_pins_left > 0
            if item_type == "yellow_pin": return self.red_yellow_pins_left > 0
            if item_type == "blue_pin": return getattr(self, "red_blue_pins_left", 0) > 0
        else:
            if item_type == "cup": return self.blue_cups_left > 0
            if item_type == "alliance_pin": return self.blue_alliance_pins_left > 0
            if item_type == "yellow_pin": return self.blue_yellow_pins_left > 0
        return False

    def _use_inventory(self, alliance, item_type):
        if alliance == "red":
            if item_type == "cup": self.red_cups_left -= 1
            if item_type == "alliance_pin": self.red_alliance_pins_left -= 1
            if item_type == "yellow_pin": self.red_yellow_pins_left -= 1
            if item_type == "blue_pin": self.red_blue_pins_left -= 1
        else:
            if item_type == "cup": self.blue_cups_left -= 1
            if item_type == "alliance_pin": self.blue_alliance_pins_left -= 1
            if item_type == "yellow_pin": self.blue_yellow_pins_left -= 1

    def _spawn_loaded_cup(self, x, y, alliance, pin_type):
        color = "red_yellow" if alliance == "red" else "blue_yellow"
        if pin_type == "yellow":
            color = "yellow_yellow"
        cup = GameCup(999, "gray", x, y, self.space)
        pin = GamePin(999, color, x, y, self.space)
        cup.contains_pin = pin
        pin.is_nested = True
        self.cups.append(cup)
        self.pins.append(pin)

    def _spawn_individual_cup(self, x, y):
        cup = GameCup(999, "gray", x, y, self.space)
        self.cups.append(cup)

    def _spawn_individual_pin(self, x, y, pin_type):
        color = "yellow_yellow" if pin_type == "yellow" else ("red_yellow" if pin_type == "red" else "blue_yellow")
        pin = GamePin(999, color, x, y, self.space)
        self.pins.append(pin)

    def _preload_robots(self):
        """SG5: each robot starts holding ONE Preload Pin (red/yellow for red,
        blue/yellow for blue) and no Cup.  Preloads are separate from the
        Match Load supply.  Skills (RSC2c): one red/yellow Preload.
        (The old sim also gave every robot a cup and took both from the
        Match Loads.)"""
        for i, robot in enumerate(self.robots):
            color = "red_yellow" if robot.alliance == "red" else "blue_yellow"
            rx, ry = float(robot.body.position.x), float(robot.body.position.y)
            rangle = float(robot.body.angle)
            pin = GamePin(9000 + i, color, rx, ry, self.space,
                          orientation=ORIENT_HORIZONTAL, angle=rangle)
            pin.carried_by  = robot.robot_id
            pin.orientation = ORIENT_HORIZONTAL
            pin.body.angle  = rangle
            pin.set_carried()
            pin._build_shape()
            robot.carrying_pin = pin
            self.pins.append(pin)
            robot.lift_state  = 1.0
            robot.lift_target = 1.0

    def run_interactive(self):
        if self.headless:
            print("[Simulator] Cannot run interactive in headless mode.")
            return
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            dt = min(dt, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:        self.reset()
                    if event.key == pygame.K_ESCAPE:   running = False
            keys = pygame.key.get_pressed()
            actions = self.handle_keyboard(keys)
            self.step(dt, actions)
            self.render()
        pygame.quit()
        print("[Simulator] Closed.")