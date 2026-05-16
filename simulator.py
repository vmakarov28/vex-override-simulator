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
)

from simulation.rules_engine import RulesEngine

# ════════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
PHYSICS_HZ        = 240
PHYSICS_DT        = 1.0 / PHYSICS_HZ
MAX_SUBSTEPS      = 8
KNOCK_IMPULSE     = 0.01            # ← Lowered for reliable knock-over
SOLVER_ITERATIONS = 30

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

        self.space = pymunk.Space()
        self.space.gravity         = (0, 0)
        self.space.damping         = 0.25
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
        print(f"[Simulator] Override Simulator initialized "
              f"(physics @ {PHYSICS_HZ} Hz, iterations={SOLVER_ITERATIONS}).")

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
        print(f"[Simulator] ✓ Registered {n_ok}/{len(pairs)} collision handlers.")

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
                    if act.get("intake", False):
                        robot.try_intake(self.pins, self.cups)
                    if act.get("score", False):
                        robot.try_score(self.goals, self.rules_engine)
                    if act.get("toggle", False):
                        self._try_flip_toggle(robot)
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

        self.red_score  = self.rules_engine.red_score
        self.blue_score = self.rules_engine.blue_score

        self.time_elapsed += dt
        self._update_phase()

    def _update_phase(self):
        end_time = AUTONOMOUS_SECONDS + DRIVER_SECONDS + SETTLE_SECONDS
        if self.time_elapsed >= end_time:
            if not self.match_over:
                self.match_over = True
                self.match_phase = "ended"
                final = self.rules_engine.calculate_final_score(self.robots)
                self.red_score  = final["red"]
                self.blue_score = final["blue"]
        elif self.time_elapsed >= AUTONOMOUS_SECONDS + DRIVER_SECONDS:
            self.match_phase = "settle"
        elif self.time_elapsed >= AUTONOMOUS_SECONDS:
            self.match_phase = "driver"
        else:
            self.match_phase = "autonomous"

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
        for toggle in self.toggles: toggle.draw(self.screen)
        for goal   in self.goals:   goal.draw(self.screen, self.font_sm)
        for cup    in self.cups:
            if cup.carried_by is None and not cup.scored: cup.draw(self.screen)
        for pin    in self.pins:
            if pin.carried_by is None and not pin.scored: pin.draw(self.screen)
        for robot  in self.robots:  robot.draw(self.screen, self.font_sm)
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

        pygame.draw.rect(surf, COLOR_WALL, (0, 0, W, H), 4)
        return surf

    def _draw_hud(self):
        W = self.screen_w
        sb_w, sb_h = 320, 44
        sb_x = W // 2 - sb_w // 2; sb_y = 8
        pygame.draw.rect(self.screen, (18, 24, 35), (sb_x, sb_y, sb_w, sb_h), border_radius=7)
        pygame.draw.rect(self.screen, (70, 88, 115), (sb_x, sb_y, sb_w, sb_h), 2, border_radius=7)
        rs = self.font_hud.render(f"RED {self.red_score}",  True, (255, 100, 100))
        bs = self.font_hud.render(f"BLUE {self.blue_score}", True, (100, 170, 255))
        self.screen.blit(rs, (sb_x + 14,  sb_y + 7))
        self.screen.blit(bs, (sb_x + 175, sb_y + 7))
        remaining = max(0, TOTAL_SECONDS - self.time_elapsed)
        t_color = (100, 220, 255) if self.match_phase == "autonomous" else (255, 200, 100)
        if remaining < 10:
            t_color = (255, 80, 80)
        phase_txt = self.font_med.render(
            f"{self.match_phase.upper()} • {remaining:.1f}s", True, t_color)
        self.screen.blit(phase_txt,
                         (W // 2 - phase_txt.get_width() // 2, sb_y + sb_h + 3))
        hints = [
            "WASD = Drive Red1",
            "E = Intake  Q = Score  T = Toggle",
            "Arrows = Drive Blue1   Shift/Ctrl",
            "R = Reset",
        ]
        for i, hint in enumerate(hints):
            ht = self.font_sm.render(hint, True, (160, 165, 180))
            self.screen.blit(ht, (8, 8 + i * 15))
        if self.match_over:
            winner = ("RED"  if self.red_score  > self.blue_score else
                      "BLUE" if self.blue_score > self.red_score  else "TIE")
            w_color = (255, 100, 100) if winner == "RED" else (
                      (100, 170, 255) if winner == "BLUE" else (200, 200, 200))
            ov = pygame.Surface((W, self.screen_h), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 100)); self.screen.blit(ov, (0, 0))
            end_surf = self.font_hud.render(
                f"{winner} WINS! {self.red_score} – {self.blue_score}", True, w_color)
            ex = W // 2 - end_surf.get_width() // 2
            ey = self.screen_h // 2 - 20
            pygame.draw.rect(self.screen, (18, 24, 35),
                             (ex - 16, ey - 10, end_surf.get_width() + 32, 48),
                             border_radius=8)
            self.screen.blit(end_surf, (ex, ey))
            reset_txt = self.font_med.render("Press R to reset", True, (200, 200, 210))
            self.screen.blit(reset_txt,
                             (W // 2 - reset_txt.get_width() // 2, ey + 32))

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
            "score":  bool(keys[pygame.K_q]),
            "toggle": bool(keys[pygame.K_t]),
        }
        left = right = 0.0
        if keys[pygame.K_UP]:    left = right = 1.0
        if keys[pygame.K_DOWN]:  left = right = -1.0
        if keys[pygame.K_LEFT]:  left, right = -0.85, 0.85
        if keys[pygame.K_RIGHT]: left, right =  0.85, -0.85
        actions[2] = {
            "left": left, "right": right,
            "intake": bool(keys[pygame.K_RSHIFT]),
            "score":  bool(keys[pygame.K_RCTRL]),
            "toggle": False,
        }
        return actions

    def reset(self):
        self.time_elapsed   = 0.0
        self.match_phase    = "autonomous"
        self.match_over     = False
        self.red_score      = 0
        self.blue_score     = 0
        self.rules_engine   = RulesEngine()
        self._physics_accum = 0.0

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

        print("[Simulator] Match reset.")

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