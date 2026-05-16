"""
simulation/robot.py
────────────────────────────────────────────────────────────────────────────
Robot class with PRODUCTION-GRADE Pymunk integration.

Major physics changes vs. the old version
─────────────────────────────────────────
1.  Carried objects are NO LONGER removed from the Pymunk space. Instead,
    their shape.filter is flipped to CARRIED_FILTER (mask=0, collides with
    nothing). When dropped or knocked free, the filter is restored.
2.  update_carried_positions() no longer rebuilds shapes. It only sets
    body.position / body.angle / zeroes velocity.
3.  The robot's own shape now uses a proper ShapeFilter.
4.  Drive forces and braking are unchanged.
"""

import pymunk
from pymunk import Vec2d
import pygame
import math
import numpy as np
from typing import Optional
from config.game_rules import (
    MAX_ROBOT_SIZE_START, RENDER_SCALE,
    ROBOT_DRIVE_FORCE, ROBOT_TURN_TORQUE, ROBOT_MAX_SPEED,
    INTAKE_RADIUS, SCORING_RADIUS,
    MAX_PINS_HELD, MAX_CUPS_HELD,
    POINTS_PIN_IN_GOAL, POINTS_CUP_IN_GOAL, POINTS_STACK_BONUS,
    ROBOT_STARTS,
)
from simulation.game_objects import (
    ORIENT_HORIZONTAL,
    COLL_TYPE_ROBOT,
    CAT_WALL, CAT_GOAL, CAT_ROBOT, CAT_PIN, CAT_CUP,
)

ROBOT_COLORS = {"red": (200, 45, 50), "blue": (45, 95, 210)}
ROBOT_BORDER = {"red": (140, 25, 30), "blue": (25, 60, 155)}
ROBOT_ACCENT = {"red": (255, 120, 120), "blue": (120, 170, 255)}

ROBOT_FILTER = pymunk.ShapeFilter(
    categories=CAT_ROBOT,
    mask=CAT_WALL | CAT_GOAL | CAT_ROBOT | CAT_PIN | CAT_CUP,
)


class Robot:
    """One VEX Override robot with differential drive, intake, and scoring."""

    def __init__(self, robot_id: str, alliance: str, space: pymunk.Space):
        self.robot_id = robot_id
        self.alliance = alliance
        self.space = space

        start = ROBOT_STARTS[robot_id]
        self.start_pos   = start["pos"]
        self.start_angle = start["angle"]

        mass = 15.0
        self.robot_width  = 17.0
        self.robot_length = 15.0

        moment = pymunk.moment_for_box(mass, (self.robot_width, self.robot_length))
        self.body = pymunk.Body(mass, moment)
        self.body.position = self.start_pos
        self.body.angle    = self.start_angle

        self.shape = pymunk.Poly.create_box(
            self.body, (self.robot_width, self.robot_length))
        self.shape.friction       = 0.35          # realistic plastic/metal
        self.shape.elasticity     = 0.05
        self.shape.collision_type = COLL_TYPE_ROBOT
        self.shape.filter         = ROBOT_FILTER
        self.shape.game_object    = self
        space.add(self.body, self.shape)

        self.carrying_pin: Optional[object]  = None
        self.carrying_cup: Optional[object]  = None
        self.successful_scores: int          = 0
        self.foul_count: int                 = 0
        self.intake_cooldown: float          = 0.0
        self.intake_in_progress: bool        = False
        self.pending_intake_object           = None
        self.is_scoring: bool                = False
        self.scoring_timer: float            = 0.0
        self.scoring_objects                 = []
        self.lift_state: float               = 0.0
        self.lift_target: float              = 0.0

        print(f"[Robot] {robot_id} ({alliance}) spawned at {self.start_pos}")

    def apply_drive(self, left: float, right: float):
        thrust = (left + right) * ROBOT_DRIVE_FORCE
        torque = (right - left) * ROBOT_TURN_TORQUE

        angle = self.body.angle
        fx = math.cos(angle) * thrust
        fy = math.sin(angle) * thrust
        self.body.apply_force_at_world_point((fx, fy), self.body.position)
        self.body.torque += torque

        if self.body.velocity.length > ROBOT_MAX_SPEED:
            self.body.velocity = self.body.velocity.normalized() * ROBOT_MAX_SPEED

        if abs(left) < 0.1 and abs(right) < 0.1:
            brake = 0.35 if (self.carrying_pin or self.carrying_cup) else 0.55
            self.body.velocity *= brake
            self.body.angular_velocity *= 0.32

        self.lift_target = 1.0 if (self.carrying_pin or self.carrying_cup) else 0.0
        if self.lift_target > self.lift_state:
            self.lift_state = min(self.lift_state + 0.12, self.lift_target)
        else:
            self.lift_state = max(self.lift_state - 0.10, self.lift_target)

        if self.intake_cooldown > 0:
            self.intake_cooldown -= 1 / 60.0
        if self.intake_in_progress and self.intake_cooldown <= 0:
            obj = self.pending_intake_object
            if obj is not None:
                from simulation.game_objects import GamePin, GameCup
                if isinstance(obj, GamePin) and self.carrying_pin is None:
                    self.carrying_pin = obj
                elif isinstance(obj, GameCup) and self.carrying_cup is None:
                    self.carrying_cup = obj
            self.pending_intake_object = None
            self.intake_in_progress = False

        if self.is_scoring and self.scoring_timer > 0:
            self.scoring_timer -= 1 / 60.0
            if self.scoring_timer <= 0:
                self.is_scoring = False
                self.scoring_objects = []

    def try_intake(self, pins: list, cups: list) -> bool:
        if self.carrying_pin and self.carrying_cup:
            return False
        if self.intake_cooldown > 0 or self.intake_in_progress:
            return False

        pos   = self.body.position
        angle = self.body.angle
        front_x = pos.x + math.cos(angle) * (self.robot_width / 2 + INTAKE_RADIUS * 0.6)
        front_y = pos.y + math.sin(angle) * (self.robot_length / 2 + INTAKE_RADIUS * 0.6)

        if not self.carrying_pin:
            best_pin, best_dist = None, INTAKE_RADIUS
            for pin in pins:
                if pin.carried_by is not None or pin.scored:
                    continue
                d = math.hypot(pin.body.position.x - front_x,
                               pin.body.position.y - front_y)
                if d < best_dist:
                    best_dist, best_pin = d, pin
            if best_pin:
                if self.carrying_cup and self.carrying_cup.contains_pin is None:
                    self.carrying_cup.contains_pin = best_pin
                    best_pin.carried_by  = self.robot_id
                    best_pin.orientation = ORIENT_HORIZONTAL
                    best_pin.body.angle  = angle
                    best_pin.set_carried()
                    best_pin._build_shape()
                    return True

                best_pin.carried_by  = self.robot_id
                best_pin.orientation = ORIENT_HORIZONTAL
                best_pin.body.angle  = angle
                best_pin.set_carried()
                best_pin._build_shape()
                self.pending_intake_object = best_pin
                self.intake_in_progress    = True
                self.intake_cooldown       = 0.4
                return True

        if not self.carrying_cup:
            best_cup, best_dist = None, INTAKE_RADIUS
            for cup in cups:
                if cup.carried_by is not None or cup.scored:
                    continue
                d = math.hypot(cup.body.position.x - front_x,
                               cup.body.position.y - front_y)
                if d < best_dist:
                    best_dist, best_cup = d, cup
            if best_cup:
                if self.carrying_pin and best_cup.contains_pin is None:
                    best_cup.contains_pin = self.carrying_pin
                    self.carrying_pin.carried_by  = self.robot_id
                    self.carrying_pin.orientation = ORIENT_HORIZONTAL
                    self.carrying_pin.body.angle  = angle
                    self.carrying_pin.set_carried()
                    self.carrying_pin._build_shape()

                best_cup.carried_by  = self.robot_id
                best_cup.orientation = ORIENT_HORIZONTAL
                best_cup.body.angle  = angle
                best_cup.set_carried()
                best_cup._build_shape()
                self.pending_intake_object = best_cup
                self.intake_in_progress    = True
                self.intake_cooldown       = 0.4
                return True

        return False

    def try_score(self, goals: list, rules_engine) -> bool:
        if not (self.carrying_pin or self.carrying_cup):
            return False
        pos = self.body.position
        for goal in goals:
            if math.hypot(pos.x - goal.x, pos.y - goal.y) > SCORING_RADIUS:
                continue

            if self.carrying_cup and self.carrying_cup.contains_pin is None and not self.carrying_pin:
                continue

            pts = 0
            obj_list = []

            if self.carrying_pin:
                pts += rules_engine.process_scored_object(goal, self.carrying_pin, self.alliance)
                obj_list.append(self.carrying_pin)
                self.carrying_pin.scored  = True
                self.carrying_pin.goal_id = goal.goal_id
                goal.scored_pins.append(self.carrying_pin)
                goal.stack_count += 1
                try:
                    if self.carrying_pin.shape and self.carrying_pin.shape in self.space.shapes:
                        self.space.remove(self.carrying_pin.shape)
                        self.carrying_pin.shape = None
                    if self.carrying_pin.body in self.space.bodies:
                        self.space.remove(self.carrying_pin.body)
                except Exception:
                    pass
                self.carrying_pin = None

            if self.carrying_cup:
                pts += rules_engine.process_scored_object(goal, self.carrying_cup, self.alliance)
                obj_list.append(self.carrying_cup)
                self.carrying_cup.scored  = True
                self.carrying_cup.goal_id = goal.goal_id
                goal.scored_cups.append(self.carrying_cup)
                goal.stack_count += 1
                try:
                    if self.carrying_cup.shape and self.carrying_cup.shape in self.space.shapes:
                        self.space.remove(self.carrying_cup.shape)
                        self.carrying_cup.shape = None
                    if self.carrying_cup.body in self.space.bodies:
                        self.space.remove(self.carrying_cup.body)
                except Exception:
                    pass
                self.carrying_cup = None

            self.successful_scores += 1
            self.is_scoring     = True
            self.scoring_timer  = 0.5
            self.scoring_objects = obj_list
            self.lift_target    = 1.0
            return True

        return False

    def update_carried_positions(self):
        pos   = self.body.position
        angle = self.body.angle
        half_l = self.robot_length / 2.0

        if self.carrying_pin:
            px = pos.x + math.cos(angle) * (half_l * 0.65)
            py = pos.y + math.sin(angle) * (half_l * 0.65)
            self.carrying_pin.body.position         = (px, py)
            self.carrying_pin.body.angle            = angle
            self.carrying_pin.body.velocity         = (0, 0)
            self.carrying_pin.body.angular_velocity = 0
            self.carrying_pin.body.force            = (0, 0)
            self.carrying_pin.body.torque           = 0
            self.carrying_pin.angle                 = angle
            self.carrying_pin.set_carried()

        if self.carrying_cup:
            cx = pos.x + math.cos(angle) * (half_l * 0.3)
            cy = pos.y + math.sin(angle) * (half_l * 0.3)
            self.carrying_cup.body.position         = (cx, cy)
            self.carrying_cup.body.angle            = angle
            self.carrying_cup.body.velocity         = (0, 0)
            self.carrying_cup.body.angular_velocity = 0
            self.carrying_cup.body.force            = (0, 0)
            self.carrying_cup.body.torque           = 0
            self.carrying_cup.angle                 = angle
            self.carrying_cup.set_carried()

            if self.carrying_cup.contains_pin:
                cp = self.carrying_cup.contains_pin
                cp.body.position         = (cx, cy)
                cp.body.angle            = angle
                cp.body.velocity         = (0, 0)
                cp.body.angular_velocity = 0
                cp.body.force            = (0, 0)
                cp.body.torque           = 0
                cp.angle                 = angle
                cp.set_carried()

        if self.pending_intake_object and hasattr(self.pending_intake_object, 'body'):
            ox = pos.x + math.cos(angle) * (half_l + 8)
            oy = pos.y + math.sin(angle) * (half_l + 8)
            self.pending_intake_object.body.position = (ox, oy)
            self.pending_intake_object.body.velocity = (0, 0)
            self.pending_intake_object.body.angular_velocity = 0

    def reset(self):
        self.body.position         = self.start_pos
        self.body.angle            = self.start_angle
        self.body.velocity         = (0, 0)
        self.body.angular_velocity = 0.0
        self.carrying_pin          = None
        self.carrying_cup          = None
        self.successful_scores     = 0
        self.foul_count            = 0
        self.intake_cooldown       = 0.0
        self.intake_in_progress    = False
        self.pending_intake_object = None
        self.is_scoring            = False
        self.scoring_timer         = 0.0
        self.scoring_objects       = []
        self.lift_state            = 0.0
        self.lift_target           = 0.0

    def draw(self, surface: pygame.Surface, font_small=None):
        pos = self.body.position
        angle = self.body.angle
        half_w = self.robot_width  / 2.0
        half_l = self.robot_length / 2.0
        sc = RENDER_SCALE

        body_color   = ROBOT_COLORS[self.alliance]
        border_color = ROBOT_BORDER[self.alliance]
        accent_color = ROBOT_ACCENT[self.alliance]

        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = []
        for dx, dy in [(-half_w, -half_l), (half_w, -half_l),
                       (half_w,  half_l), (-half_w,  half_l)]:
            wx = pos.x + dx * cos_a - dy * sin_a
            wy = pos.y + dx * sin_a + dy * cos_a
            corners.append((wx * sc, wy * sc))

        pygame.draw.polygon(surface, body_color, corners)
        pygame.draw.polygon(surface, border_color, corners, 2)

        hl_pts = [corners[0], corners[1],
                  ((corners[1][0] + corners[2][0]) / 2,
                   (corners[1][1] + corners[2][1]) / 2),
                  ((corners[0][0] + corners[3][0]) / 2,
                   (corners[0][1] + corners[3][1]) / 2)]
        hl_surf = pygame.Surface((surface.get_width(), surface.get_height()),
                                 pygame.SRCALPHA)
        pygame.draw.polygon(hl_surf, (*accent_color, 50), hl_pts)
        surface.blit(hl_surf, (0, 0))

        cx, cy = pos.x * sc, pos.y * sc

        fwd_len = half_l * sc * 1.5
        fx, fy = cx + cos_a * fwd_len, cy + sin_a * fwd_len
        pygame.draw.line(surface, (255, 255, 255), (cx, cy), (fx, fy), 2)
        for offset in [2.4, -2.4]:
            ax = fx + math.cos(angle + offset) * 5
            ay = fy + math.sin(angle + offset) * 5
            pygame.draw.line(surface, (255, 255, 255), (fx, fy), (ax, ay), 2)

        perp_angle = angle + math.pi / 2
        lift_base = half_w * 0.3 * sc
        lift_max  = half_w * 1.1 * sc
        l_len = lift_base + (lift_max - lift_base) * self.lift_state
        lx = cx + math.cos(perp_angle) * l_len
        ly = cy + math.sin(perp_angle) * l_len
        lift_color = (255, 215, 60) if (self.carrying_pin or self.carrying_cup) else (180, 160, 50)
        pygame.draw.line(surface, lift_color, (int(cx), int(cy)), (int(lx), int(ly)), 4)
        pygame.draw.circle(surface, lift_color, (int(lx), int(ly)), 4)

        if self.carrying_pin:  self.carrying_pin.draw(surface)
        if self.carrying_cup:  self.carrying_cup.draw(surface)

        if font_small:
            label = font_small.render(self.robot_id[-1], True, (255, 255, 255))
            surface.blit(label,
                         (int(cx) - label.get_width() // 2,
                          int(cy) - label.get_height() // 2))
