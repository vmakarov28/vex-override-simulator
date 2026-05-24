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
    MAX_ROBOT_SIZE_START, MIDFIELD_CENTER, MIDFIELD_HALF,
    GOALS, TOGGLES, ROBOT_STARTS,
    ENDGAME_SECONDS,
)

from simulation.robot import Robot
from simulation.game_objects import (
    GamePin, GameCup,
    create_field_objects, create_goals, create_toggles,
    FieldGoal, FieldToggle,
    COLL_TYPE_PIN, COLL_TYPE_CUP, COLL_TYPE_WALL, COLL_TYPE_GOAL,
    COLL_TYPE_ROBOT,
    CAT_WALL, CAT_GOAL, CAT_ROBOT, CAT_PIN, CAT_CUP,
    ORIENT_VERTICAL, ORIENT_HORIZONTAL,
    NORMAL_PIN_FILTER,          # ← ADD THIS
)

from simulation.rules_engine import RulesEngine

# ════════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
PHYSICS_HZ        = 240
PHYSICS_DT        = 1.0 / PHYSICS_HZ
MAX_SUBSTEPS      = 4               # 4 × 1/240 s ≈ 67 ms sim time per control step
KNOCK_IMPULSE     = 0.01            # ← Lowered for reliable knock-over
SOLVER_ITERATIONS = 10              # 10 iters is stable for 2-D top-down physics

COLOR_BG       = (22, 28, 38)
COLOR_FIELD    = (52, 58, 68)
COLOR_TILE_LINE = (60, 66, 78)
COLOR_TAPE     = (240, 240, 245)
COLOR_MIDFIELD = (35, 90, 155)
COLOR_WALL     = (80, 88, 100)


class OverrideSimulator:
    def __init__(self, headless: bool = False, render_scale: float = RENDER_SCALE):
        self.headless     = headless
        self.render_scale = render_scale
        self.screen_w     = SCREEN_W
        self.screen_h     = SCREEN_H
        self.timer_started = False
        self.time_remaining = AUTONOMOUS_SECONDS

        # Match Load Inventory
        self.red_cups_left = 10
        self.red_alliance_pins_left = 12
        self.red_yellow_pins_left = 1

        self.blue_cups_left = 10
        self.blue_alliance_pins_left = 12
        self.blue_yellow_pins_left = 1

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
        self.rules_engine = RulesEngine()

        self.time_elapsed   = 0.0
        self.match_phase    = "autonomous"
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
        self._create_robots()
        self._create_field_objects()
        self._register_collision_handlers()

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

    def _create_robots(self):
        order = ["red1", "red2", "blue1", "blue2"]
        alliances = {"red1": "red", "red2": "red", "blue1": "blue", "blue2": "blue"}
        self.robots = [Robot(rid, alliances[rid], self.space) for rid in order]

    def _create_field_objects(self):
        self.pins, self.cups = create_field_objects(self.space)
        self.goals   = create_goals(self.space)
        self.toggles = create_toggles()

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

        if actions:
            for i, robot in enumerate(self.robots):
                if i < len(actions) and actions[i]:
                    act = actions[i]
                    robot.apply_drive(float(act.get("left", 0)), float(act.get("right", 0)))

                    if act.get("flip_pin", False) and robot.carrying_pin:
                        robot.carrying_pin.flip()
                    if act.get("flip_cup", False) and robot.carrying_cup:
                        robot.carrying_cup.flip()
                    if act.get("intake", False):
                        robot.try_intake(self.pins, self.cups)
                    if act.get("score_pin", False):
                        robot.try_score_pin(self.goals, self.rules_engine)
                    if act.get("score_cup", False):
                        robot.try_score_cup(self.goals, self.rules_engine)
                    if act.get("toggle", False):
                        robot.try_toggle(self.toggles)
                    if act.get("match_load", False):
                        self._try_bot_match_load(robot)
                else:
                    robot.apply_drive(0, 0)

        self._physics_accum += dt
        steps = 0
        while self._physics_accum >= PHYSICS_DT and steps < MAX_SUBSTEPS:
            for robot in self.robots:
                robot.update_carried_positions()
            self.space.step(PHYSICS_DT)
            self._physics_accum -= PHYSICS_DT
            steps += 1
        if self._physics_accum > PHYSICS_DT * MAX_SUBSTEPS:
            self._physics_accum = 0.0

        for pin in self.pins: pin.update(dt)
        for cup in self.cups: cup.update(dt)

        for robot in self.robots:
            self.rules_engine.check_possession(robot)
        self.rules_engine.update_yellow_pin_ownership(self.pins, self.toggles)

        # Set endgame flag before scoring so the +8 bonus and central goal
        # height restriction apply immediately when time crosses 20s.
        self.rules_engine.endgame_active = (
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
        elif self.time_remaining <= 0 and self.match_phase == "autonomous":
            self.match_phase = "driver"
            self.time_remaining = DRIVER_SECONDS

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
        rs = self.font_sm.render(f"RED {self.red_score}", True, (255, 100, 100))
        bs = self.font_sm.render(f"BLUE {self.blue_score}", True, (100, 170, 255))

        if self.rules_engine.endgame_active:
            t_color  = (255, 210, 50)
            t_label  = f"ENDGAME  {max(0, int(self.time_remaining))}s"
        else:
            t_color = (200, 200, 210)
            t_label = f"{max(0, int(self.time_remaining))}s"
        timer_txt = self.font_sm.render(t_label, True, t_color)

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
        # Red (left side)
        red_inv = self.font_sm.render(
            f"Red: {self.red_cups_left}C {self.red_alliance_pins_left}P {self.red_yellow_pins_left}Y",
            True, (255, 150, 150)
        )
        self.screen.blit(red_inv, (8, self.screen_h - 20))

        # Blue (right side)
        blue_inv = self.font_sm.render(
            f"Blue: {self.blue_cups_left}C {self.blue_alliance_pins_left}P {self.blue_yellow_pins_left}Y",
            True, (150, 180, 255)
        )
        self.screen.blit(blue_inv, (self.screen_w - blue_inv.get_width() - 8, self.screen_h - 20))

        # === GAME OVER OVERLAY ===
        if self.match_over:
            winner = ("RED" if self.red_score > self.blue_score else
                      "BLUE" if self.blue_score > self.red_score else "TIE")
            w_color = (255, 100, 100) if winner == "RED" else (
                      (100, 170, 255) if winner == "BLUE" else (200, 200, 200))

            ov = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 100))
            self.screen.blit(ov, (0, 0))

            end_surf = self.font_hud.render(
                f"{winner} WINS! {self.red_score} – {self.blue_score}", True, w_color)
            ex = self.screen_w // 2 - end_surf.get_width() // 2
            ey = self.screen_h // 2 - 20

            pygame.draw.rect(self.screen, (18, 24, 35),
                             (ex - 16, ey - 10, end_surf.get_width() + 32, 48),
                             border_radius=8)
            self.screen.blit(end_surf, (ex, ey))

            reset_txt = self.font_med.render("Press R to reset", True, (200, 200, 210))
            self.screen.blit(reset_txt,
                             (self.screen_w // 2 - reset_txt.get_width() // 2, ey + 32))

    def handle_keyboard(self, keys) -> List[Optional[Dict]]:
        actions = [None, None, None, None]
        left = right = 0.0
        if keys[pygame.K_w]: left = right = 1.0
        if keys[pygame.K_s]: left = right = -1.0
        if keys[pygame.K_a]: left, right = -0.85, 0.85
        if keys[pygame.K_d]: left, right =  0.85, -0.85
        actions[0] = {
            "left": left, "right": right,
            "intake": bool(keys[pygame.K_e]),
            "score_pin": bool(keys[pygame.K_q] and not keys[pygame.K_LSHIFT]),
            "score_cup": bool(keys[pygame.K_q] and keys[pygame.K_LSHIFT]),
            "toggle": bool(keys[pygame.K_t]),
            "flip_pin": bool(keys[pygame.K_f] and not keys[pygame.K_LSHIFT]),
            "flip_cup": bool(keys[pygame.K_f] and keys[pygame.K_LSHIFT]),
        }
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
        blue1 = self.robots[2]

        # Red Match Load
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

        # Blue Match Load
        if self._is_in_loading_zone(blue1, "blue"):
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
        self.match_phase    = "autonomous"
        self.match_over     = False
        self.red_score      = 0
        self.blue_score     = 0
        self.rules_engine   = RulesEngine()
        self._physics_accum = 0.0

        # === RESET TIMER ===
        self.timer_started = False
        self.time_remaining = AUTONOMOUS_SECONDS

        self.red_cups_left = 10
        self.red_alliance_pins_left = 12
        self.red_yellow_pins_left = 1
        self.blue_cups_left = 10
        self.blue_alliance_pins_left = 12
        self.blue_yellow_pins_left = 1
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

        for robot in self.robots:
            robot.reset()

        #print("[Simulator] Match reset.")
    def _is_in_loading_zone(self, robot, alliance):
        pos = robot.body.position
        if alliance == "red":
            return (pos.x < 12 and pos.y < 24) or (pos.x < 12 and pos.y > 120)
        else:
            return (pos.x > 132 and pos.y < 24) or (pos.x > 132 and pos.y > 120)

    def _try_bot_match_load(self, robot) -> bool:
        """Bot-driven match load.  Only red1/blue1 are eligible (matches the
        keyboard handler's restriction).  Robot must be in its alliance's
        loading zone.  Picks the best available selection automatically:

            1. Loaded cup + yellow pin (selection 2)   — highest value
            2. Loaded cup + alliance pin (selection 1) — pin+cup at once
            3. Individual alliance pin   (selection 4)
            4. Individual cup            (selection 3)
            5. Individual yellow pin     (selection 5) — fallback (no cup left)

        Returns True if a match load actually fired.
        """
        if robot.robot_id not in ("red1", "blue1"):
            return False
        alliance = robot.alliance
        if not self._is_in_loading_zone(robot, alliance):
            return False
        if self._has_inventory(alliance, "cup") and self._has_inventory(alliance, "yellow_pin"):
            self._perform_match_load(robot, 2)
            return True
        if self._has_inventory(alliance, "cup") and self._has_inventory(alliance, "alliance_pin"):
            self._perform_match_load(robot, 1)
            return True
        if self._has_inventory(alliance, "alliance_pin"):
            self._perform_match_load(robot, 4)
            return True
        if self._has_inventory(alliance, "cup"):
            self._perform_match_load(robot, 3)
            return True
        if self._has_inventory(alliance, "yellow_pin"):
            self._perform_match_load(robot, 5)
            return True
        return False

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