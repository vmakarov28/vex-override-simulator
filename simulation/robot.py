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
    GamePin,
    CARRIED_FILTER,          # ← ADD THIS LINE
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

        if self.is_scoring and self.scoring_timer > 0:
            self.scoring_timer -= 1 / 60.0
            if self.scoring_timer <= 0:
                self.is_scoring = False
                self.scoring_objects = []

    def try_intake(self, pins: list, cups: list) -> bool:
        """Attempt to intake the nearest eligible object in front of the robot.

        Possession rules (strictly enforced):
          - At most 1 pin  held at any time.
          - At most 1 cup  held at any time.
          - Having both is legal; the robot is simply full when both slots are
            occupied and will refuse further intakes until one is scored.

        Intake geometry:
          - The search origin is the robot's front face centre.
          - Objects within INTAKE_RADIUS inches of that origin are eligible.
          - We also accept objects that are within INTAKE_RADIUS of the robot's
            body centre (catches wall-pinned pieces that slide behind the nose).
          - Closest eligible object wins (separate closest for pin vs cup so
            that spamming E while carrying 1 cup will grab the nearest pin
            regardless of nearby cups, and vice-versa).
        """
        if self.intake_cooldown > 0:
            return False

        has_pin = self.carrying_pin is not None
        has_cup = self.carrying_cup is not None

        # Both slots full — nothing to do.
        if has_pin and has_cup:
            return False

        pos   = self.body.position
        angle = self.body.angle

        # Two candidate origins: front-face centre and body centre.
        # Using both catches objects pinned against the wall that have slid
        # slightly behind the robot's geometric front point.
        half_len = self.robot_length / 2.0
        front_x  = pos.x + math.cos(angle) * (half_len + 2.0)
        front_y  = pos.y + math.sin(angle) * (half_len + 2.0)

        def in_range(obj_pos):
            """True if obj_pos is within INTAKE_RADIUS of front OR body."""
            d_front = math.hypot(obj_pos.x - front_x, obj_pos.y - front_y)
            d_body  = math.hypot(obj_pos.x - pos.x,   obj_pos.y - pos.y)
            return min(d_front, d_body), d_front <= INTAKE_RADIUS or d_body <= INTAKE_RADIUS

        best_pin  = None;  best_pin_d  = 999.0
        best_cup  = None;  best_cup_d  = 999.0

        # ── Pin search (only if we don't already have one) ──────────────
        if not has_pin:
            for pin in pins:
                if pin.carried_by is not None or getattr(pin, 'scored', False):
                    continue
                if getattr(pin, 'is_nested', False):
                    continue          # nested pins are picked up with their cup
                d, ok = in_range(pin.body.position)
                if ok and d < best_pin_d:
                    best_pin_d = d
                    best_pin   = pin

        # ── Cup search (only if we don't already have one) ──────────────
        if not has_cup:
            for cup in cups:
                if cup.carried_by is not None or getattr(cup, 'scored', False):
                    continue
                d, ok = in_range(cup.body.position)
                if ok and d < best_cup_d:
                    best_cup_d = d
                    best_cup   = cup

        # ── Choose: prefer whichever is closer, subject to what we need ─
        # If we need both, grab the closer one first.
        # If we only need one type, grab that type.
        target_obj  = None
        target_type = None

        if best_pin is not None and best_cup is not None:
            # Need both — take the closer one.
            if best_pin_d <= best_cup_d:
                target_obj, target_type = best_pin, "pin"
            else:
                target_obj, target_type = best_cup, "cup"
        elif best_pin is not None:
            target_obj, target_type = best_pin, "pin"
        elif best_cup is not None:
            target_obj, target_type = best_cup, "cup"

        if target_obj is None:
            return False

        # ── Perform intake ───────────────────────────────────────────────
        target_obj.carried_by  = self.robot_id
        target_obj.orientation = ORIENT_HORIZONTAL
        target_obj.body.angle  = angle
        target_obj.body.velocity         = (0, 0)
        target_obj.body.angular_velocity = 0
        target_obj.set_carried()
        target_obj._build_shape()

        if target_type == "pin":
            self.carrying_pin = target_obj

        else:  # cup
            self.carrying_cup = target_obj

            # If the cup has a nested pin AND we don't already have a pin,
            # extract it as a separately carried pin.
            # If we already carry a pin, leave the nested pin inside the cup
            # (the robot takes the cup-with-pin as one unit).
            nested = target_obj.contains_pin
            if nested is not None and self.carrying_pin is None:
                nested.carried_by          = self.robot_id
                nested.orientation         = ORIENT_HORIZONTAL
                nested.body.angle          = angle
                nested.body.velocity       = (0, 0)
                nested.body.angular_velocity = 0
                nested.set_carried()
                if nested.shape:
                    nested.shape.filter = CARRIED_FILTER
                self.carrying_pin          = nested
                target_obj.contains_pin    = None
                if hasattr(nested, 'is_nested'):
                    nested.is_nested = False   # allow it to render normally

        # Short cooldown prevents a single keypress registering as two intakes.
        self.intake_cooldown = 0.15
        return True

    def try_score_pin(self, goals: list, rules_engine) -> bool:
        if not self.carrying_pin: return False
        pos = self.body.position
        for goal in goals:
            if math.hypot(pos.x - goal.x, pos.y - goal.y) > SCORING_RADIUS: continue

            # Alliance goals: only the matching alliance may score here.
            if goal.alliance != "neutral" and goal.alliance != self.alliance:
                continue

            # Endgame: central goal is locked once the stack reaches
            # Pin + Cup + Pin (3 items). Locked in if already at/above that
            # height before endgame started.
            if getattr(rules_engine, 'endgame_active', False):
                from config.game_rules import CENTER_GOAL_ID, ENDGAME_CENTER_MAX_STACK
                if goal.goal_id == CENTER_GOAL_ID and len(goal.stack) >= ENDGAME_CENTER_MAX_STACK:
                    continue

            # Prevent pin on pin (illegal stack)
            if goal.stack:
                last_obj, last_is_pin = goal.stack[-1]
                if last_is_pin:
                    return False  # Cannot score pin directly on another pin

            rules_engine.process_scored_object(goal, self.carrying_pin, self.alliance)
            self.carrying_pin.scored = True
            self.carrying_pin.goal_id = goal.goal_id
            try:
                if self.carrying_pin.shape and self.carrying_pin.shape in self.space.shapes:
                    self.space.remove(self.carrying_pin.shape)
                if self.carrying_pin.body in self.space.bodies:
                    self.space.remove(self.carrying_pin.body)
            except: pass
            self.carrying_pin = None
            self.successful_scores += 1
            self.is_scoring = True
            self.scoring_timer = 0.5
            return True
        return False

    def try_toggle(self, toggles: list) -> bool:
        """Try to flip the nearest toggle."""
        pos = self.body.position
        for toggle in toggles:
            dist = math.hypot(pos.x - toggle.x, pos.y - toggle.y)
            if dist <= 18:  # interaction range
                if toggle.owner == "red":
                    toggle.owner = "blue"
                elif toggle.owner == "blue":
                    toggle.owner = "yellow"
                else:
                    toggle.owner = self.alliance
                return True
        return False

    def try_score_cup(self, goals: list, rules_engine) -> bool:
        if not self.carrying_cup: 
            return False
        pos = self.body.position
        for goal in goals:
            if math.hypot(pos.x - goal.x, pos.y - goal.y) > SCORING_RADIUS: 
                continue

            # Alliance goals: only the matching alliance may score here.
            if goal.alliance != "neutral" and goal.alliance != self.alliance:
                continue

            # Endgame: central goal locked once stack reaches Pin+Cup+Pin (3 items).
            if getattr(rules_engine, 'endgame_active', False):
                from config.game_rules import CENTER_GOAL_ID, ENDGAME_CENTER_MAX_STACK
                if goal.goal_id == CENTER_GOAL_ID and len(goal.stack) >= ENDGAME_CENTER_MAX_STACK:
                    continue

            # Prevent illegal stacks (cup must go on a pin)
            if not goal.stack:
                return False
            last_obj, last_is_pin = goal.stack[-1]
            if not last_is_pin:
                return False

            # The side facing the FRONT of the robot goes DOWN on the goal.
            rules_engine.process_scored_object(goal, self.carrying_cup, self.alliance)
            self.carrying_cup.scored = True
            self.carrying_cup.goal_id = goal.goal_id
            try:
                if self.carrying_cup.shape and self.carrying_cup.shape in self.space.shapes:
                    self.space.remove(self.carrying_cup.shape)
                if self.carrying_cup.body in self.space.bodies:
                    self.space.remove(self.carrying_cup.body)
            except: 
                pass
            self.carrying_cup = None
            self.successful_scores += 1
            self.is_scoring = True
            self.scoring_timer = 0.5
            return True
        return False

    def try_score(self, goals: list, rules_engine) -> bool:
        if self.try_score_pin(goals, rules_engine): return True
        return self.try_score_cup(goals, rules_engine)

    def update_carried_positions(self):
        """Pin on LEFT side, Cup on RIGHT side — closer to center."""
        pos   = self.body.position
        angle = self.body.angle
        half_l = self.robot_length / 2.0

        left_x,  left_y  = -math.sin(angle),  math.cos(angle)
        right_x, right_y =  math.sin(angle), -math.cos(angle)

        if self.carrying_pin:
            px = pos.x + left_x * 5.0 + math.cos(angle) * (half_l * 0.25)
            py = pos.y + left_y * 5.0 + math.sin(angle) * (half_l * 0.25)
            pin_angle = angle + (math.pi if getattr(self.carrying_pin, 'flipped', False) else 0.0)
            self.carrying_pin.body.position         = (px, py)
            self.carrying_pin.body.angle            = pin_angle
            self.carrying_pin.body.velocity         = (0, 0)
            self.carrying_pin.body.angular_velocity = 0
            self.carrying_pin.body.force            = (0, 0)
            self.carrying_pin.body.torque           = 0
            self.carrying_pin.angle                 = pin_angle
            self.carrying_pin.set_carried()

        if self.carrying_cup:
            cx = pos.x + right_x * 5.0 + math.cos(angle) * (half_l * 0.25)
            cy = pos.y + right_y * 5.0 + math.sin(angle) * (half_l * 0.25)
            self.carrying_cup.body.position         = (cx, cy)
            self.carrying_cup.body.angle            = angle
            self.carrying_cup.body.velocity         = (0, 0)
            self.carrying_cup.body.angular_velocity = 0
            self.carrying_cup.body.force            = (0, 0)
            self.carrying_cup.body.torque           = 0
            self.carrying_cup.angle                 = angle
            self.carrying_cup.set_carried()

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
        half_w = self.robot_width / 2.0
        half_l = self.robot_length / 2.0
        sc = RENDER_SCALE

        body_color   = ROBOT_COLORS[self.alliance]
        border_color = ROBOT_BORDER[self.alliance]
        accent_color = ROBOT_ACCENT[self.alliance]

        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners = []
        for dx, dy in [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (-half_w, half_l)]:
            wx = pos.x + dx * cos_a - dy * sin_a
            wy = pos.y + dx * sin_a + dy * cos_a
            corners.append((wx * sc, wy * sc))

        pygame.draw.polygon(surface, body_color, corners)
        pygame.draw.polygon(surface, border_color, corners, 2)

        # Lift + front arrow (unchanged)
        cx, cy = pos.x * sc, pos.y * sc
        fwd_len = half_l * sc * 1.5
        fx, fy = cx + cos_a * fwd_len, cy + sin_a * fwd_len
        pygame.draw.line(surface, (255, 255, 255), (cx, cy), (fx, fy), 2)

        perp_angle = angle + math.pi / 2
        lift_base = half_w * 0.3 * sc
        lift_max  = half_w * 1.1 * sc
        l_len = lift_base + (lift_max - lift_base) * self.lift_state
        lx = cx + math.cos(perp_angle) * l_len
        ly = cy + math.sin(perp_angle) * l_len
        lift_color = (255, 215, 60) if (self.carrying_pin or self.carrying_cup) else (180, 160, 50)
        pygame.draw.line(surface, lift_color, (int(cx), int(cy)), (int(lx), int(ly)), 4)
        pygame.draw.circle(surface, lift_color, (int(lx), int(ly)), 4)
        # Draw carried pin (LEFT) and cup (RIGHT) independently
        if self.carrying_pin:
            self.carrying_pin.draw(surface)
        if self.carrying_cup:
            self.carrying_cup.draw(surface)

        if font_small:
            label = font_small.render(self.robot_id[-1], True, (255, 255, 255))
            surface.blit(label, (int(cx) - label.get_width() // 2, int(cy) - label.get_height() // 2))
