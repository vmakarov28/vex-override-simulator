"""
simulation/game_objects.py
────────────────────────────────────────────────────────────────────────────
Game objects with PRODUCTION-GRADE 2-D rigid-body physics (Pymunk/Chipmunk2D).

Key architectural changes vs. the old version
─────────────────────────────────────────────
1. Rounded convex polygons (pymunk.Poly with radius=...) replace the old
   pointy diamonds. Rounding is THE fix for clipping into robot corners
   and "perpendicular-launch" on glancing blows. (Minkowski sum trick.)
2. pymunk.ShapeFilter categories carry carried-state info. Held objects
   flip their filter mask to 0 (collides with nothing) — no shape add/
   remove churn during play.
3. body.velocity_func enforces per-object speed and angular-speed caps.
   A pinched pin can NEVER fling at 800 in/s anymore — physically clamped.
4. body.angle is the single source of truth for orientation. self.angle
   is kept in sync purely for the existing drawing code.
5. knock_over() now uses the REAL contact normal coming from Pymunk's
   post-solve callback (see simulator.py). The math bug
   (target = hit + π/2) is fixed: pins fall PARALLEL to plow direction.
"""

import pymunk
import pygame
import math
from typing import Optional, Tuple
from config.game_rules import (
    RENDER_SCALE,
    PIN_STARTS, CUP_STARTS, GOALS, TOGGLES,
    POINTS_PIN_IN_GOAL, POINTS_CUP_IN_GOAL, POINTS_STACK_BONUS,
    POINTS_YELLOW_OWNED, TOGGLE_INTERACTION_RANGE,
    FOUL_STANDARD_PTS,
    ORIENT_VERTICAL, ORIENT_HORIZONTAL,
)

# ════════════════════════════════════════════════════════════════════════════
# COLORS (unchanged)
# ════════════════════════════════════════════════════════════════════════════
C_RED        = (210, 45, 50)
C_RED_LIGHT  = (255, 110, 110)
C_BLUE       = (40, 90, 200)
C_BLUE_LIGHT = (100, 160, 255)
C_YELLOW     = (240, 195, 20)
C_YELLOW_LT  = (255, 230, 100)
C_WHITE      = (240, 240, 245)
C_GRAY_DARK  = (90, 90, 100)
C_GRAY_MID   = (155, 155, 165)
C_GRAY_LIGHT = (200, 200, 210)
C_OUTLINE    = (20, 20, 25)
C_SHADOW     = (15, 15, 20)

# ════════════════════════════════════════════════════════════════════════════
# PHYSICS SIZES (inches) and MASSES
# ────────────────────────────────────────────────────────────────────────────
PIN_PHYS_HALF_LEN = 4.10
PIN_PHYS_HALF_WID = 1.05
PIN_PHYS_RADIUS_V = 1.05
PIN_CORNER_R      = 0.35
PIN_MASS          = 0.55

CUP_PHYS_HALF_LEN = 4.20
CUP_PHYS_HALF_WID = 2.10
CUP_PHYS_RADIUS_V = 2.40
CUP_CORNER_R      = 0.45
CUP_MASS          = 0.80

# Draw sizes (pixels)
PIN_DRAW_RADIUS_V = 12
PIN_DRAW_HALF_LEN = 26
PIN_DRAW_WIDTH_H  = 10
CUP_DRAW_RADIUS_V = 14
CUP_DRAW_HALF_LEN = 32
CUP_DRAW_WIDTH_H  = 14

# ════════════════════════════════════════════════════════════════════════════
# COLLISION TYPES & SHAPE-FILTER CATEGORIES
# ════════════════════════════════════════════════════════════════════════════
COLL_TYPE_ROBOT = 1
COLL_TYPE_PIN   = 2
COLL_TYPE_CUP   = 3
COLL_TYPE_GOAL  = 5
COLL_TYPE_WALL  = 9

CAT_WALL    = 1 << 0
CAT_GOAL    = 1 << 1
CAT_ROBOT   = 1 << 2
CAT_PIN     = 1 << 3
CAT_CUP     = 1 << 4
CAT_CARRIED = 1 << 5

NORMAL_PIN_FILTER = pymunk.ShapeFilter(
    categories=CAT_PIN,
    mask=CAT_WALL | CAT_GOAL | CAT_ROBOT | CAT_PIN | CAT_CUP,
)
NORMAL_CUP_FILTER = pymunk.ShapeFilter(
    categories=CAT_CUP,
    mask=CAT_WALL | CAT_GOAL | CAT_ROBOT | CAT_PIN | CAT_CUP,
)
CARRIED_FILTER = pymunk.ShapeFilter(categories=CAT_CARRIED, mask=0)

# ════════════════════════════════════════════════════════════════════════════
# VELOCITY-LIMITER (anti-launch guarantee)
# ════════════════════════════════════════════════════════════════════════════
MAX_OBJ_LINEAR_SPEED  = 70.0
MAX_OBJ_ANGULAR_SPEED = 14.0

def _limit_velocity(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, damping, dt)
    v = body.velocity
    speed = v.length
    if speed > MAX_OBJ_LINEAR_SPEED:
        body.velocity = v * (MAX_OBJ_LINEAR_SPEED / speed)
    av = body.angular_velocity
    if av > MAX_OBJ_ANGULAR_SPEED:
        body.angular_velocity = MAX_OBJ_ANGULAR_SPEED
    elif av < -MAX_OBJ_ANGULAR_SPEED:
        body.angular_velocity = -MAX_OBJ_ANGULAR_SPEED


# ════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _hex_points(cx, cy, r, angle_offset=0.0):
    return [
        (cx + r * math.cos(math.radians(60 * i + angle_offset)),
         cy + r * math.sin(math.radians(60 * i + angle_offset)))
        for i in range(6)
    ]

def _draw_bicolor_hex(surface, cx, cy, r, color_top, color_bot, outline=C_OUTLINE):
    pts = _hex_points(cx, cy, r, angle_offset=30)
    pygame.draw.polygon(surface, color_bot, pts)
    top_pts = [p for p in pts if p[1] < cy]
    if top_pts:
        left_x = min(p[0] for p in pts)
        right_x = max(p[0] for p in pts)
        clip = [(left_x, cy)] + top_pts + [(right_x, cy)]
        if len(clip) >= 3:
            pygame.draw.polygon(surface, color_top, clip)
    pygame.draw.polygon(surface, outline, pts, 2)

def _highlight_hex(surface, cx, cy, r, angle_offset=30):
    pts = _hex_points(cx, cy, r, angle_offset)
    gloss_pts = [p for p in pts if p[1] < cy - r * 0.1]
    if len(gloss_pts) >= 2:
        gloss_pts = [(cx, cy - r * 0.8)] + gloss_pts
        gloss_surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
        offset_pts = [(p[0] - cx + r * 2, p[1] - cy + r * 2) for p in gloss_pts]
        if len(offset_pts) >= 3:
            pygame.draw.polygon(gloss_surf, (255, 255, 255, 40), offset_pts)
        surface.blit(gloss_surf, (int(cx - r * 2), int(cy - r * 2)))


# ════════════════════════════════════════════════════════════════════════════
# GAME PIN
# ════════════════════════════════════════════════════════════════════════════
class GamePin:
    def __init__(self, pin_id: int, color: str, x: float, y: float,
                 space: pymunk.Space,
                 orientation: str = ORIENT_VERTICAL, angle: float = 0.0):
        self.pin_id      = pin_id
        self.color       = color
        self.is_yellow   = 'yellow' in color
        self.space       = space
        self.orientation = orientation
        self.angle       = angle

        # Game state MUST be initialised BEFORE _build_shape()
        self.scored          = False
        self.goal_id         = None
        self.carried_by      = None
        self.is_yellow_owned = False
        self._knock_cooldown = 0.0

        moment = pymunk.moment_for_circle(PIN_MASS, 0, PIN_PHYS_RADIUS_V)
        self.body = pymunk.Body(PIN_MASS, moment)
        self.body.position      = (x, y)
        self.body.angle         = angle
        self.body.velocity_func = _limit_velocity
        space.add(self.body)

        self.shape: Optional[pymunk.Shape] = None
        self._build_shape()

    def _build_shape(self):
        if self.shape is not None and self.shape in self.space.shapes:
            self.space.remove(self.shape)
        self.shape = None

        if self.orientation == ORIENT_VERTICAL:
            shape = pymunk.Circle(self.body, PIN_PHYS_RADIUS_V)
            self.body.moment = pymunk.moment_for_circle(
                PIN_MASS, 0, PIN_PHYS_RADIUS_V)
            shape.friction   = 0.45
            shape.elasticity = 0.04
        else:
            hl = PIN_PHYS_HALF_LEN - PIN_CORNER_R
            hw = PIN_PHYS_HALF_WID - PIN_CORNER_R
            verts = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
            shape = pymunk.Poly(self.body, verts, radius=PIN_CORNER_R)
            self.body.moment = pymunk.moment_for_poly(
                PIN_MASS, verts, (0, 0), radius=PIN_CORNER_R)
            shape.friction   = 0.40
            shape.elasticity = 0.04

        shape.collision_type = COLL_TYPE_PIN
        shape.game_object    = self
        shape.filter = CARRIED_FILTER if self.carried_by else NORMAL_PIN_FILTER
        self.space.add(shape)
        self.shape = shape

    def knock_over(self, hit_angle: float, impulse_mag: float = 0.0):
        if self.orientation == ORIENT_HORIZONTAL:
            return
        # Always flip on any registered collision
        self.orientation = ORIENT_HORIZONTAL
        self.body.angle = hit_angle
        self.body.angular_velocity = min(4.5, max(1.2, impulse_mag * 0.028))
        self._build_shape()
        self._knock_cooldown = 0.12

    def stand_up(self):
        self.orientation = ORIENT_VERTICAL
        self.body.angle  = 0.0
        self._build_shape()

    def update(self, dt: float):
        if self.carried_by is not None:
            if self.shape and self.shape.filter is not CARRIED_FILTER:
                self.shape.filter = CARRIED_FILTER
            return
        if self.shape and self.shape.filter is CARRIED_FILTER:
            self.shape.filter = NORMAL_PIN_FILTER
        if self._knock_cooldown > 0:
            self._knock_cooldown = max(0.0, self._knock_cooldown - dt)
        self.angle = self.body.angle

    def set_carried(self):
        if self.shape:
            self.shape.filter = CARRIED_FILTER

    def set_released(self):
        if self.shape:
            self.shape.filter = NORMAL_PIN_FILTER

    def get_colors(self):
        return {
            "red":         (C_RED, C_RED),
            "blue":        (C_BLUE, C_BLUE),
            "yellow":      (C_YELLOW, C_YELLOW),
            "red_yellow":  (C_RED, C_YELLOW),
            "blue_yellow": (C_BLUE, C_YELLOW),
        }.get(self.color, (C_GRAY_MID, C_GRAY_MID))

    def draw(self, surface):
        sc = RENDER_SCALE
        px = self.body.position.x * sc
        py = self.body.position.y * sc
        c1, c2 = self.get_colors()
        if self.orientation == ORIENT_VERTICAL:
            self._draw_vertical(surface, px, py, c1, c2)
        else:
            self._draw_horizontal(surface, px, py, c1, c2)
        if self.is_yellow_owned:
            r = PIN_DRAW_RADIUS_V + 4 if self.orientation == ORIENT_VERTICAL else PIN_DRAW_WIDTH_H + 4
            glow_surf = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255, 235, 40, 70), (r * 2, r * 2), r * 2)
            surface.blit(glow_surf, (int(px - r * 2), int(py - r * 2)))

    def _draw_vertical(self, surface, px, py, c1, c2):
        r = PIN_DRAW_RADIUS_V
        shadow_pts = _hex_points(px + 1, py + 2, r, angle_offset=30)
        pygame.draw.polygon(surface, C_SHADOW, shadow_pts)
        pts = _hex_points(px, py, r, angle_offset=30)
        if c1 == c2:
            pygame.draw.polygon(surface, c1, pts)
        else:
            _draw_bicolor_hex(surface, px, py, r, c1, c2)
        _highlight_hex(surface, px, py, r)
        pygame.draw.polygon(surface, C_OUTLINE, pts, 2)
        pygame.draw.circle(surface, (255, 255, 255, 100), (int(px), int(py)), 2)

    def _draw_horizontal(self, surface, px, py, c1, c2):
        hl = PIN_DRAW_HALF_LEN
        hw = PIN_DRAW_WIDTH_H * 0.85
        a = self.angle
        cos_a, sin_a = math.cos(a), math.sin(a)
        cx, cy = px, py
        lx = cx - hl * cos_a; ly = cy - hl * sin_a
        rx = cx + hl * cos_a; ry = cy + hl * sin_a
        tx = cx + hw * sin_a; ty = cy - hw * cos_a
        bx = cx - hw * sin_a; by = cy + hw * cos_a
        pygame.draw.polygon(surface, c1, [(lx, ly), (tx, ty), (bx, by)])
        pygame.draw.polygon(surface, c2, [(rx, ry), (tx, ty), (bx, by)])
        pygame.draw.polygon(surface, C_OUTLINE,
                            [(lx, ly), (tx, ty), (rx, ry), (bx, by)], 3)
        pygame.draw.line(surface, (255, 255, 255, 90), (cx, cy), (tx, ty), 2)
        pygame.draw.line(surface, (255, 255, 255, 90), (cx, cy), (bx, by), 2)
        pygame.draw.circle(surface, C_OUTLINE, (int(cx), int(cy)), 3)

    def get_points(self, toggle_owner=None):
        if self.is_yellow and toggle_owner:
            return POINTS_YELLOW_OWNED
        return POINTS_PIN_IN_GOAL

    def __repr__(self):
        return f"Pin({self.pin_id}, {self.color}, {self.orientation})"


# ════════════════════════════════════════════════════════════════════════════
# GAME CUP
# ════════════════════════════════════════════════════════════════════════════
class GameCup:
    def __init__(self, cup_id: int, side: str, x: float, y: float,
                 space: pymunk.Space,
                 orientation: str = ORIENT_VERTICAL, angle: float = 0.0):
        self.cup_id      = cup_id
        self.side        = side
        self.space       = space
        self.orientation = orientation
        self.angle       = angle

        # Game state MUST be initialised BEFORE _build_shape()
        self.contains_pin    = None
        self.scored          = False
        self.goal_id         = None
        self.carried_by      = None
        self._knock_cooldown = 0.0

        moment = pymunk.moment_for_circle(CUP_MASS, 0, CUP_PHYS_RADIUS_V)
        self.body = pymunk.Body(CUP_MASS, moment)
        self.body.position      = (x, y)
        self.body.angle         = angle
        self.body.velocity_func = _limit_velocity
        space.add(self.body)

        self.shape: Optional[pymunk.Shape] = None
        self._build_shape()

    def _build_shape(self):
        if self.shape is not None and self.shape in self.space.shapes:
            self.space.remove(self.shape)
        self.shape = None

        if self.orientation == ORIENT_VERTICAL:
            shape = pymunk.Circle(self.body, CUP_PHYS_RADIUS_V)
            self.body.moment = pymunk.moment_for_circle(
                CUP_MASS, 0, CUP_PHYS_RADIUS_V)
            shape.friction   = 0.45
            shape.elasticity = 0.06
        else:
            hl = CUP_PHYS_HALF_LEN - CUP_CORNER_R
            hw = CUP_PHYS_HALF_WID - CUP_CORNER_R
            verts = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
            shape = pymunk.Poly(self.body, verts, radius=CUP_CORNER_R)
            self.body.moment = pymunk.moment_for_poly(
                CUP_MASS, verts, (0, 0), radius=CUP_CORNER_R)
            shape.friction   = 0.40
            shape.elasticity = 0.06

        shape.collision_type = COLL_TYPE_CUP
        shape.game_object    = self
        shape.filter = CARRIED_FILTER if self.carried_by else NORMAL_CUP_FILTER
        self.space.add(shape)
        self.shape = shape

    def knock_over(self, hit_angle: float, impulse_mag: float = 0.0):
        if self.orientation == ORIENT_HORIZONTAL:
            return
        # Always flip on any registered collision (cup behavior preserved)
        self.orientation = ORIENT_HORIZONTAL
        self.body.angle = hit_angle
        self.body.angular_velocity = min(5.0, max(1.5, impulse_mag * 0.035))
        self._build_shape()
        self._knock_cooldown = 0.12

    def update(self, dt: float):
        if self.carried_by is not None:
            if self.shape and self.shape.filter is not CARRIED_FILTER:
                self.shape.filter = CARRIED_FILTER
            return
        if self.shape and self.shape.filter is CARRIED_FILTER:
            self.shape.filter = NORMAL_CUP_FILTER
        if self._knock_cooldown > 0:
            self._knock_cooldown = max(0.0, self._knock_cooldown - dt)
        self.angle = self.body.angle

    def set_carried(self):
        if self.shape:
            self.shape.filter = CARRIED_FILTER

    def set_released(self):
        if self.shape:
            self.shape.filter = NORMAL_CUP_FILTER

    def insert_pin(self, pin):
        if self.contains_pin is None:
            self.contains_pin = pin
            pin.carried_by = None
            return True
        return False

    def get_points(self):
        base = POINTS_CUP_IN_GOAL
        if self.contains_pin:
            base += POINTS_STACK_BONUS
        return base

    def draw(self, surface):
        sc = RENDER_SCALE
        px = self.body.position.x * sc
        py = self.body.position.y * sc
        if self.orientation == ORIENT_VERTICAL:
            self._draw_vertical(surface, px, py)
        else:
            self._draw_horizontal(surface, px, py)

    def _draw_vertical(self, surface, px, py):
        r = CUP_DRAW_RADIUS_V
        cx, cy = int(px), int(py)

        # Bottom half = WHITE
        pygame.draw.circle(surface, C_WHITE, (cx, cy), r)

        # Top half = DARK GREY (only draw the upper semicircle)
        points = []
        for angle in range(180, 361):           # 180° to 360° = top half
            rad = math.radians(angle)
            x = cx + r * math.cos(rad)
            y = cy + r * math.sin(rad)
            points.append((x, y))
        points.append((cx, cy))                 # center point to close the shape
        pygame.draw.polygon(surface, C_GRAY_DARK, points)

        # Clean black outline
        pygame.draw.circle(surface, C_OUTLINE, (cx, cy), r, 2)

        # Center split line (optional but nice)
        pygame.draw.line(surface, C_OUTLINE, (cx - r, cy), (cx + r, cy), 1)

        # Nested pin (small yellow dot in center)
        if self.contains_pin:
            inner_r = max(2, int(r * 0.38))
            pygame.draw.circle(surface, C_YELLOW, (cx, cy), inner_r)
            pygame.draw.circle(surface, C_OUTLINE, (cx, cy), inner_r, 1)

    def _draw_horizontal(self, surface, px, py):
        hl = PIN_DRAW_HALF_LEN * 0.88
        hw = CUP_DRAW_WIDTH_H * 1.3
        a = self.angle
        cos_a, sin_a = math.cos(a), math.sin(a)
        cx, cy = px, py
        px_perp = -sin_a; py_perp = cos_a
        lx = cx - hl * cos_a; ly = cy - hl * sin_a
        lx_top = lx + px_perp * (hw / 2); ly_top = ly + py_perp * (hw / 2)
        lx_bot = lx - px_perp * (hw / 2); ly_bot = ly - py_perp * (hw / 2)
        rx = cx + hl * cos_a; ry = cy + hl * sin_a
        rx_top = rx + px_perp * (hw / 2); ry_top = ry + py_perp * (hw / 2)
        rx_bot = rx - px_perp * (hw / 2); ry_bot = ry - py_perp * (hw / 2)
        center_x, center_y = cx, cy
        if self.side == 'white':
            color_left, color_right = C_WHITE, C_GRAY_DARK
        else:
            color_left, color_right = C_GRAY_DARK, C_WHITE
        pygame.draw.circle(surface, (40, 40, 45), (int(center_x), int(center_y)), 6)
        pygame.draw.circle(surface, C_OUTLINE, (int(center_x), int(center_y)), 6, 2)
        pygame.draw.polygon(surface, color_left,
                            [(lx_top, ly_top), (lx_bot, ly_bot), (center_x, center_y)])
        pygame.draw.polygon(surface, color_right,
                            [(rx_top, ry_top), (rx_bot, ry_bot), (center_x, center_y)])
        outline_pts = [(lx_top, ly_top), (lx_bot, ly_bot),
                       (center_x, center_y),
                       (rx_top, ry_top), (rx_bot, ry_bot)]
        pygame.draw.polygon(surface, C_OUTLINE, outline_pts, 3)
        pygame.draw.circle(surface, C_OUTLINE, (int(center_x), int(center_y)), 3)

    def _draw_nested_pin(self, surface, px, py, r):
        inner_r = max(3, r // 2)
        pygame.draw.circle(surface, C_YELLOW, (px, py), inner_r)
        pygame.draw.circle(surface, C_OUTLINE, (px, py), inner_r, 1)


# ════════════════════════════════════════════════════════════════════════════
# FIELD GOAL
# ════════════════════════════════════════════════════════════════════════════
class FieldGoal:
    def __init__(self, goal_dict: dict, space: pymunk.Space = None):
        self.goal_id  = goal_dict["id"]
        self.alliance = goal_dict["alliance"]
        self.x        = goal_dict["x"]
        self.y        = goal_dict["y"]
        self.radius   = goal_dict["radius"]
        self.label    = goal_dict["label"]
        self.scored_pins = []
        self.scored_cups = []
        self.stack_count = 0

        if space:
            self.phys_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            self.phys_body.position = (self.x, self.y)
            self.phys_shape = pymunk.Circle(self.phys_body, self.radius * 0.25)
            self.phys_shape.friction   = 0.6
            self.phys_shape.elasticity = 0.08
            self.phys_shape.collision_type = COLL_TYPE_GOAL
            self.phys_shape.filter = pymunk.ShapeFilter(
                categories=CAT_GOAL,
                mask=CAT_WALL | CAT_ROBOT | CAT_PIN | CAT_CUP,
            )
            space.add(self.phys_body, self.phys_shape)

    def contains(self, px, py):
        dx, dy = px - self.x, py - self.y
        return math.sqrt(dx * dx + dy * dy) <= self.radius * 1.5

    def get_score(self):
        total = sum(c.get_points() for c in self.scored_cups)
        for pin in self.scored_pins:
            if pin.scored and not getattr(pin, '_counted_in_cup', False):
                total += pin.get_points()
        return total

    def draw(self, surface, font_small):
        sx = int(self.x * RENDER_SCALE)
        sy = int(self.y * RENDER_SCALE)
        r  = int(self.radius * RENDER_SCALE)
        if self.alliance == "red":
            base_color, accent_color = (180, 35, 40), (230, 80, 80)
        elif self.alliance == "blue":
            base_color, accent_color = (30, 70, 180), (80, 130, 230)
        else:
            base_color, accent_color = (80, 80, 90), (140, 140, 155)
        post_w = max(3, r // 3)
        pygame.draw.circle(surface, accent_color, (sx, sy), post_w // 2 + 5)
        pygame.draw.circle(surface, (220, 220, 230), (sx, sy), post_w // 2 + 2)
        if self.stack_count > 0:
            for i in range(min(self.stack_count, 5)):
                oy = sy - r + 60 - i * 7
                scol = (255, 200, 40) if i % 2 == 0 else (200, 60, 60)
                pygame.draw.circle(surface, scol, (sx, oy), 5)
                pygame.draw.circle(surface, C_OUTLINE, (sx, oy), 5, 1)
        score = self.get_score()
        if font_small and score > 0:
            txt = font_small.render(f"+{score}", True, (255, 230, 80))
            surface.blit(txt, (sx - txt.get_width() // 2, sy + r + 3))


# ════════════════════════════════════════════════════════════════════════════
# FIELD TOGGLE (Roller)
# ════════════════════════════════════════════════════════════════════════════
class FieldToggle:
    def __init__(self, toggle_dict):
        self.toggle_id   = toggle_dict["id"]
        self.x           = toggle_dict["x"]
        self.y           = toggle_dict["y"]
        self.owner       = toggle_dict["owner"]
        self.quadrant    = toggle_dict["quadrant"]
        self.orientation = toggle_dict.get("orientation", ORIENT_HORIZONTAL)
        self.length = 20.0
        self.width  = 2.0

    def try_interact(self, robot):
        dist = math.sqrt((robot.body.position.x - self.x) ** 2 +
                         (robot.body.position.y - self.y) ** 2)
        if dist <= TOGGLE_INTERACTION_RANGE:
            if self.owner == "red":
                self.owner = "blue"
            elif self.owner == "blue":
                self.owner = "yellow"
            else:
                self.owner = "red" if robot.alliance == "red" else "blue"
            return True
        return False

    def draw(self, surface):
        sc = RENDER_SCALE
        sx = int(self.x * sc); sy = int(self.y * sc)
        length_px = int(self.length * sc); width_px = int(self.width * sc)
        if self.owner == "red":   color = (210, 50, 50)
        elif self.owner == "blue": color = (50, 100, 210)
        else:                      color = (240, 195, 20)
        if self.orientation == ORIENT_HORIZONTAL:
            rect = pygame.Rect(sx - length_px // 2, sy - width_px // 2,
                               length_px, width_px)
        else:
            rect = pygame.Rect(sx - width_px // 2, sy - length_px // 2,
                               width_px, length_px)
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, (255, 255, 255), rect, 3)


# ════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def create_field_objects(space: pymunk.Space):
    pins = []
    for i, p in enumerate(PIN_STARTS):
        pins.append(GamePin(i, p["color"], p["x"], p["y"], space,
                            p.get("orientation", ORIENT_VERTICAL),
                            p.get("angle", 0.0)))
    cups = []
    for i, c in enumerate(CUP_STARTS):
        cups.append(GameCup(i, c["side"], c["x"], c["y"], space,
                            c.get("orientation", ORIENT_VERTICAL),
                            c.get("angle", 0.0)))
    return pins, cups

def create_goals(space: pymunk.Space = None):
    return [FieldGoal(g, space) for g in GOALS]

def create_toggles():
    return [FieldToggle(t) for t in TOGGLES]