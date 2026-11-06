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
from config.robot_profile import RobotProfile, DEFAULT_PROFILE
from simulation.game_objects import (
    ORIENT_HORIZONTAL, ORIENT_VERTICAL,
    CAT_ROBOT_RED, CAT_ROBOT_BLUE,
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

    def __init__(self, robot_id: str, alliance: str, space: pymunk.Space,
                 profile: RobotProfile = None):
        self.robot_id = robot_id
        self.alliance = alliance
        self.space = space
        self.profile = profile or DEFAULT_PROFILE

        start = ROBOT_STARTS[robot_id]
        self.start_pos   = start["pos"]
        self.start_angle = start["angle"]

        mass = 15.0
        # NOTE: pymunk box (w, h) puts the first size along the body x-axis,
        # which is the robot's FORWARD direction.  So robot_length goes first.
        self.robot_width  = float(self.profile.width)
        self.robot_length = float(self.profile.length)

        moment = pymunk.moment_for_box(mass, (self.robot_length, self.robot_width))
        self.body = pymunk.Body(mass, moment)
        self.body.position = self.start_pos
        self.body.angle    = self.start_angle

        self.shape = pymunk.Poly.create_box(
            self.body, (self.robot_length, self.robot_width))
        self.shape.friction       = 0.35          # realistic plastic/metal
        self.shape.elasticity     = 0.05
        self.shape.collision_type = COLL_TYPE_ROBOT
        alliance_cat = CAT_ROBOT_RED if alliance == "red" else CAT_ROBOT_BLUE
        self.shape.filter = pymunk.ShapeFilter(
            categories=CAT_ROBOT | alliance_cat,
            mask=CAT_WALL | CAT_GOAL | CAT_ROBOT | CAT_PIN | CAT_CUP,
        )
        self.shape.game_object    = self
        space.add(self.body, self.shape)

        self.carrying_pin: Optional[object]  = None
        self.carrying_cup: Optional[object]  = None
        self.successful_scores: int          = 0
        self.foul_count: int                 = 0
        self.intake_cooldown: float          = 0.0
        self.mech_busy: float                = 0.0   # scoring / flipping lockout (s)
        self.toggle_progress: float          = 0.0
        self.intake_in_progress: bool        = False
        self.pending_intake_object           = None
        self.is_scoring: bool                = False
        self.scoring_timer: float            = 0.0
        self.scoring_objects                 = []
        self.lift_state: float               = 0.0
        self.lift_target: float              = 0.0
        self._dt: float                      = 1 / 60.0
        # Set by the simulator: callable(x, y) -> bool, False when an object
        # at (x, y) may not be touched right now (opponent side in auton).
        self.touch_allowed = None

    # ── geometry helpers ─────────────────────────────────────────────────
    def _point_along(self, dist: float):
        a = self.body.angle
        p = self.body.position
        return p.x + math.cos(a) * dist, p.y + math.sin(a) * dist

    def front_point(self):
        return self._point_along(self.robot_length / 2.0 + 2.0)

    def rear_point(self):
        return self._point_along(-(self.robot_length / 2.0 + 2.0))

    def score_point(self):
        side = self.profile.score_side
        if side == "rear":
            return self.rear_point()
        if side == "front":
            return self.front_point()
        p = self.body.position
        return p.x, p.y

    # Drive force is applied on EVERY physics substep (physics_substep).  The
    # old code applied it once per step() call, so only the first of N
    # substeps was driven and the robot's speed depended on the step size.
    # DRIVE_SCALE reproduces the original interactive feel (1 driven substep
    # in 4 at 60 fps): ~60 in/s top speed, ~190 deg/s turning.
    DRIVE_SCALE = 0.25

    def physics_substep(self):
        left, right = getattr(self, "_cmd", (0.0, 0.0))
        thrust = (left + right) * ROBOT_DRIVE_FORCE * self.DRIVE_SCALE
        torque = (right - left) * ROBOT_TURN_TORQUE * self.DRIVE_SCALE
        angle = self.body.angle
        self.body.apply_force_at_world_point(
            (math.cos(angle) * thrust, math.sin(angle) * thrust), self.body.position)
        self.body.torque += torque
        vmax = min(ROBOT_MAX_SPEED, self.profile.max_speed * 1.15)
        if self.body.velocity.length > vmax:
            self.body.velocity = self.body.velocity.normalized() * vmax

    def apply_drive(self, left: float, right: float):
        self._cmd = (float(left), float(right))
        dt = self._dt
        if abs(left) < 0.1 and abs(right) < 0.1:
            brake = 0.35 if (self.carrying_pin or self.carrying_cup) else 0.55
            k = dt * 60.0                       # tuned per 1/60 s frame
            self.body.velocity *= brake ** k
            self.body.angular_velocity *= 0.32 ** k

        self.lift_target = 1.0 if (self.carrying_pin or self.carrying_cup) else 0.0
        if self.lift_target > self.lift_state:
            self.lift_state = min(self.lift_state + 0.12, self.lift_target)
        else:
            self.lift_state = max(self.lift_state - 0.10, self.lift_target)

        if self.intake_cooldown > 0:
            self.intake_cooldown -= dt
        if self.mech_busy > 0:
            self.mech_busy -= dt

        if self.is_scoring and self.scoring_timer > 0:
            self.scoring_timer -= dt
            if self.scoring_timer <= 0:
                self.is_scoring = False
                self.scoring_objects = []

    # ── intake ───────────────────────────────────────────────────────────
    def _touch_ok(self, obj) -> bool:
        if self.touch_allowed is None:
            return True
        if getattr(obj, "on_auton_line", False):
            return True
        p = obj.body.position
        return self.touch_allowed(p.x, p.y)

    def _grab(self, obj, kind: str):
        angle = self.body.angle
        obj.carried_by  = self.robot_id
        obj.orientation = ORIENT_HORIZONTAL
        obj.body.angle  = angle
        obj.body.velocity         = (0, 0)
        obj.body.angular_velocity = 0
        if hasattr(obj, "is_nested"):
            obj.is_nested = False
        obj.set_carried()
        obj._build_shape()
        if kind == "pin":
            self.carrying_pin = obj
        else:
            self.carrying_cup = obj

    def try_intake(self, pins: list, cups: list, only=None) -> bool:
        """FRONT intake: nearest loose pin or cup in front of the robot.

        Possession (SG6): at most 1 pin + 1 cup.  A cup with a nested pin
        is taken as both at once when the pin slot is free; if the robot
        already holds a pin, a loaded cup cannot be taken (would be 2 pins).
        """
        if self.intake_cooldown > 0:
            return False
        has_pin = self.carrying_pin is not None
        has_cup = self.carrying_cup is not None
        if has_pin and has_cup:
            return False
        fx, fy = self.front_point()

        best, best_d, best_kind = None, INTAKE_RADIUS, None
        if not has_pin and self.profile.front_intake_pins:
            for pin in pins:
                if pin.carried_by is not None or getattr(pin, 'scored', False):
                    continue
                if getattr(pin, 'is_nested', False):
                    continue          # nested pins come with their cup
                if only is not None and pin not in only:
                    continue
                d = math.hypot(pin.body.position.x - fx, pin.body.position.y - fy)
                if d <= best_d and self._touch_ok(pin):
                    best, best_d, best_kind = pin, d, "pin"
        if not has_cup and self.profile.front_intake_cups:
            for cup in cups:
                if cup.carried_by is not None or getattr(cup, 'scored', False):
                    continue
                if cup.contains_pin is not None and has_pin:
                    continue          # would exceed 1 pin
                if only is not None and cup not in only:
                    continue
                d = math.hypot(cup.body.position.x - fx, cup.body.position.y - fy)
                if d <= best_d and self._touch_ok(cup):
                    best, best_d, best_kind = cup, d, "cup"
        if best is None:
            return False

        if best_kind == "pin":
            self._grab(best, "pin")
        else:
            nested = best.contains_pin
            self._grab(best, "cup")
            if nested is not None:
                best.contains_pin = None
                self._grab(nested, "pin")
        self.intake_cooldown = self.profile.intake_time
        return True

    def try_intake_rear(self, pins: list, cups: list, only=None) -> bool:
        """BACK intake: VERTICAL pins only (standing alone, or standing up in
        a vertical cup when rear_pulls_nested_pin)."""
        if not self.profile.rear_intake_vertical_pins:
            return False
        if self.intake_cooldown > 0 or self.carrying_pin is not None:
            return False
        rx, ry = self.rear_point()
        best, best_d, best_cup = None, INTAKE_RADIUS, None
        for pin in pins:
            if pin.carried_by is not None or getattr(pin, 'scored', False):
                continue
            if getattr(pin, 'is_nested', False) or pin.orientation != ORIENT_VERTICAL:
                continue
            if only is not None and pin not in only:
                continue
            d = math.hypot(pin.body.position.x - rx, pin.body.position.y - ry)
            if d <= best_d and self._touch_ok(pin):
                best, best_d, best_cup = pin, d, None
        if self.profile.rear_pulls_nested_pin:
            for cup in cups:
                if cup.contains_pin is None or cup.carried_by is not None:
                    continue
                if getattr(cup, 'scored', False) or cup.orientation != ORIENT_VERTICAL:
                    continue
                if only is not None and cup not in only:
                    continue
                d = math.hypot(cup.body.position.x - rx, cup.body.position.y - ry)
                if d <= best_d and self._touch_ok(cup):
                    best, best_d, best_cup = cup.contains_pin, d, cup
        if best is None:
            return False
        if best_cup is not None:
            best_cup.contains_pin = None
            best.body.position = best_cup.body.position
        self._grab(best, "pin")
        self.intake_cooldown = self.profile.intake_time
        return True

    # ── scoring ──────────────────────────────────────────────────────────
    def _goal_accepts(self, goal, kind: str, rules_engine) -> bool:
        from config.game_rules import (CENTER_GOAL_ID, GOAL_HEIGHT,
                                       PIN_STACK_HEIGHT)
        skills = getattr(rules_engine, "skills_mode", False)
        # SG9: never touch the opponent's alliance goals (Skills: RSC2e
        # allows every goal).
        if not skills and goal.alliance != "neutral" and goal.alliance != self.alliance:
            return False
        # SG12 (v2.0): no Placing on the Midfield Goal during the Endgame.
        if getattr(rules_engine, 'endgame_active', False) and goal.goal_id == CENTER_GOAL_ID:
            return False
        if self.touch_allowed is not None and not self.touch_allowed(goal.x, goal.y):
            return False
        if goal.stack:
            last_is_pin = goal.stack[-1][1]
        else:
            last_is_pin = None
        if kind == "pin" and last_is_pin is True:
            return False
        if kind == "cup" and last_is_pin is not True:
            return False
        n_pins = sum(1 for _, ip in goal.stack if ip)
        top = GOAL_HEIGHT[goal.kind] + PIN_STACK_HEIGHT * n_pins
        if top > self.profile.max_place_height:
            return False
        return True

    def _in_score_range(self, goal) -> bool:
        sx, sy = self.score_point()
        return math.hypot(sx - goal.x, sy - goal.y) <= SCORING_RADIUS

    def _place(self, goal, obj, rules_engine, attr: str):
        rules_engine.process_scored_object(goal, obj, self.alliance)
        obj.scored = True
        obj.goal_id = goal.goal_id
        try:
            if obj.shape and obj.shape in self.space.shapes:
                self.space.remove(obj.shape)
            if obj.body in self.space.bodies:
                self.space.remove(obj.body)
        except Exception:
            pass
        setattr(self, attr, None)
        self.successful_scores += 1
        self.is_scoring = True
        self.scoring_timer = 0.5

    def try_score_pin(self, goals: list, rules_engine) -> bool:
        if not self.carrying_pin or self.mech_busy > 0:
            return False
        for goal in goals:
            if self._in_score_range(goal) and self._goal_accepts(goal, "pin", rules_engine):
                self._place(goal, self.carrying_pin, rules_engine, "carrying_pin")
                self.mech_busy = self.profile.score_pin_time
                return True
        return False

    def try_score_cup(self, goals: list, rules_engine) -> bool:
        if not self.carrying_cup or self.mech_busy > 0:
            return False
        for goal in goals:
            if self._in_score_range(goal) and self._goal_accepts(goal, "cup", rules_engine):
                # The cup half facing the FRONT of the robot goes DOWN.
                self._place(goal, self.carrying_cup, rules_engine, "carrying_cup")
                self.mech_busy = self.profile.score_cup_time
                return True
        return False

    def try_toggle(self, toggles: list, skills: bool = False) -> bool:
        """Push on the nearest toggle.  It only changes colour after
        TOGGLE_SET_TIME seconds of continuous contact (the old sim flipped it
        instantly).  Head-to-head: sets it to our alliance.  Skills (RSC3):
        sets it to the colour of its own Quadrant, the only setting that
        makes yellow pins there Owned."""
        from config.game_rules import TOGGLE_SET_TIME, TOGGLE_INTERACTION_RANGE
        pos = self.body.position
        for toggle in toggles:
            if math.hypot(pos.x - toggle.x, pos.y - toggle.y) <= TOGGLE_INTERACTION_RANGE:
                target = toggle.quadrant if skills else self.alliance
                if toggle.owner == target:
                    self.toggle_progress = 0.0
                    return False
                if self.touch_allowed is not None and not self.touch_allowed(toggle.x, toggle.y):
                    return False
                self.toggle_progress += self._dt
                if self.toggle_progress >= TOGGLE_SET_TIME:
                    toggle.owner = target
                    self.toggle_progress = 0.0
                    return True
                return False
        self.toggle_progress = 0.0
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
        self.mech_busy             = 0.0
        self.toggle_progress       = 0.0
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
        for dx, dy in [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)]:
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
