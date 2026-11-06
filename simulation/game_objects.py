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
    CENTER_GOAL_ID,
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
# Physics sizes (inches)
PIN_PHYS_HALF_LEN = 3.25         # official Pin: 6.5" (165 mm) long
PIN_PHYS_HALF_WID = 0.77 #1.05
PIN_PHYS_RADIUS_V = 0.77 #1.05
PIN_CORNER_R      = 0.35
PIN_MASS          = 0.55


CUP_REAL_DIAMETER = 3.15         # official Cup: 3.15" (80 mm) diameter
CUP_REAL_HEIGHT   = 6.5          # official Cup: 6.5" (164.5 mm) tall
CUP_PHYS_RADIUS_V = CUP_REAL_DIAMETER / 2          # 2.0 inches
CUP_PHYS_HALF_WID = CUP_REAL_DIAMETER / 2
CUP_PHYS_HALF_LEN = CUP_REAL_HEIGHT / 2            # 2.25 inches
CUP_CORNER_R      = 0.30
CUP_MASS          = 0.80

# Draw sizes (pixels)
PIN_DRAW_HALF_LEN = 19# 12
PIN_DRAW_WIDTH_H  = 7 #26
PIN_DRAW_RADIUS_V = 9 # 10
CUP_DRAW_RADIUS_V = int(CUP_PHYS_RADIUS_V * RENDER_SCALE)   # 10 pixels
CUP_DRAW_HALF_LEN = int(CUP_PHYS_HALF_LEN * RENDER_SCALE)   # 11.25 pixels
CUP_DRAW_WIDTH_H  = int(CUP_PHYS_HALF_WID * RENDER_SCALE)   # 10 pixels

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
CAT_ROBOT_RED  = 1 << 6     # extra category bits so auton walls can block
CAT_ROBOT_BLUE = 1 << 7     # only one alliance's robots

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
NEST_EJECT_IMPULSE    = 40.0   # impulse needed to knock a nested pin out of a cup
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
        self.owned_by = None
        self.flipped = False          # ← NEW: which half is "up"
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
    
    def flip(self):
        """Flip which colored end is 'up' (the scored end)."""
        self.flipped = not self.flipped
        if self.orientation == ORIENT_HORIZONTAL:
            self.body.angle += math.pi
            self.angle = self.body.angle

    def get_colors(self):
        return {
            "red":          (C_RED, C_RED),
            "blue":         (C_BLUE, C_BLUE),
            "yellow":       (C_YELLOW, C_YELLOW),
            "yellow_yellow": (C_YELLOW, C_YELLOW),   # ← ADD THIS LINE
            "red_yellow":   (C_RED, C_YELLOW),
            "blue_yellow":  (C_BLUE, C_YELLOW),
            "red_blue":     (C_RED, C_BLUE),
            "blue_red":     (C_BLUE, C_RED),
        }.get(self.color, (C_GRAY_MID, C_GRAY_MID))

    def get_up_color(self):
        """Returns the color of the 'up' (visible/scored) half as an RGB tuple."""
        c1, c2 = self.get_colors()
        return c2 if self.flipped else c1

    def get_down_color(self):
        """Returns the color of the 'down' half as an RGB tuple."""
        c1, c2 = self.get_colors()
        return c1 if self.flipped else c2

    # Colour-name helpers — return plain strings so callers can compare
    # without importing the C_* RGB constants from game_objects.
    _COLOR_NAME_MAP = {
        "red":           ("red",    "red"),
        "blue":          ("blue",   "blue"),
        "yellow":        ("yellow", "yellow"),
        "yellow_yellow": ("yellow", "yellow"),
        "red_yellow":    ("red",    "yellow"),
        "blue_yellow":   ("blue",   "yellow"),
        "red_blue":      ("red",    "blue"),
        "blue_red":      ("blue",   "red"),
    }

    @property
    def up_half_name(self) -> str:
        """Name string of the up-facing half: 'red', 'blue', or 'yellow'."""
        c1, c2 = self._COLOR_NAME_MAP.get(self.color, ("gray", "gray"))
        return c2 if self.flipped else c1

    @property
    def down_half_name(self) -> str:
        """Name string of the down-facing half: 'red', 'blue', or 'yellow'."""
        c1, c2 = self._COLOR_NAME_MAP.get(self.color, ("gray", "gray"))
        return c1 if self.flipped else c2


    def draw(self, surface):
        sc = RENDER_SCALE
        px = self.body.position.x * sc
        py = self.body.position.y * sc
        c1, c2 = self.get_colors()

        # === UNDERGLOW ===
        if self.is_yellow_owned and hasattr(self, 'owned_by') and self.owned_by:
            extension = 6   # Consistent extension past the pin

            if self.orientation == ORIENT_VERTICAL:
                glow_r = PIN_DRAW_RADIUS_V + 6
                size = int(glow_r * 2.8)
                glow_surf = pygame.Surface((size, size), pygame.SRCALPHA)

                if self.owned_by == "red":
                    glow_color = (255, 0, 0, 90)
                else:
                    glow_color = (0, 70, 255, 90)

                center = size // 2
                pygame.draw.circle(glow_surf, glow_color, (center, center), glow_r * 1.2)

                # === MANUAL CENTERING OFFSET ===
                offset_x = 0.48   # ← Adjust this (positive = right, negative = left)
                offset_y = 0.525   # ← Adjust this (positive = down, negative = up)

                surface.blit(glow_surf, (round(px - center + offset_x), 
                                         round(py - center + offset_y)))
            else:
                glow_w = int(PIN_DRAW_HALF_LEN * 2) + extension
                glow_h = int(PIN_DRAW_WIDTH_H * 2) + extension

                glow_surf = pygame.Surface((glow_w + 30, glow_h + 30), pygame.SRCALPHA)

                if self.owned_by == "red":
                    glow_color = (255, 0, 0, 90)        # More vibrant red
                else:
                    glow_color = (0, 70, 255, 90)      # More vibrant blue

                pygame.draw.ellipse(glow_surf, glow_color, (15, 15, glow_w, glow_h))
                rotated_glow = pygame.transform.rotate(glow_surf, -math.degrees(self.angle))
                rot_rect = rotated_glow.get_rect(center=(px, py))
                surface.blit(rotated_glow, rot_rect)

        # Draw the pin on top
        if self.orientation == ORIENT_VERTICAL:
            self._draw_vertical(surface, px, py, c1, c2)
        else:
            self._draw_horizontal(surface, px, py, c1, c2)

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
        # pygame.draw.circle(surface, (255, 255, 255, 100), (int(px), int(py)), 2)

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
        # pygame.draw.circle(surface, C_OUTLINE, (int(cx), int(cy)), 3)

    def get_points(self, toggle_owner=None):
        if self.is_yellow and toggle_owner:
            return POINTS_YELLOW_OWNED
        return POINTS_PIN_IN_GOAL

    def get_half_points(self, is_top: bool, toggle_owner=None):
        """Returns points for one half of the pin based on actual orientation."""
        c1, c2 = self.get_colors()
        
        # Determine which color is on top
        if self.flipped:
            top_color = c2  # flipped = c2 is up
            bottom_color = c1
        else:
            top_color = c1  # normal = c1 is up
            bottom_color = c2

        # Get the color for this half
        half_color = top_color if is_top else bottom_color

        # Scoring rules
        if half_color == C_YELLOW:
            if toggle_owner:
                return POINTS_YELLOW_OWNED
            else:
                return 0
        elif half_color in (C_RED, C_BLUE):
            return POINTS_PIN_IN_GOAL
        else:
            return 0


# ════════════════════════════════════════════════════════════════════════════
# GAME CUP
# ════════════════════════════════════════════════════════════════════════════
class GameCup:
    def __init__(self, cup_id: int, side: str, x: float, y: float,
                 space: pymunk.Space,
                 orientation: str = ORIENT_VERTICAL, angle: float = 0.0,
                 contains_pin: str = None, clear_up: bool = True):
        self.cup_id      = cup_id
        self.side        = side
        self.space       = space
        self.orientation = orientation
        self.angle       = angle

        self.contains_pin    = None
        self.scored          = False
        self.goal_id         = None
        self.carried_by      = None
        self.clear_on_top    = bool(clear_up)   # manual: 24 gray-up, 12 clear-up
        self.flipped         = False
        self._knock_cooldown = 0.0

        moment = pymunk.moment_for_circle(CUP_MASS, 0, CUP_PHYS_RADIUS_V)
        self.body = pymunk.Body(CUP_MASS, moment)
        self.body.position      = (x, y)
        self.body.angle         = angle
        self.body.velocity_func = _limit_velocity
        space.add(self.body)

        self.shape: Optional[pymunk.Shape] = None
        self._build_shape()

        # NEW: Create nested pin if specified
        if contains_pin:
            from simulation.game_objects import GamePin
            pin = GamePin(
                pin_id=999 + cup_id,
                color=contains_pin,
                x=x,
                y=y,
                space=space,
                orientation=ORIENT_VERTICAL,
                angle=0.0
            )
            pin.is_nested = True   # ← Mark it so it doesn't get drawn at full size
            pin.carried_by = None
            pin.scored = False
            self.contains_pin = pin
            # Make the nested pin not collide until separated
            if pin.shape:
                pin.shape.filter = CARRIED_FILTER

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

        # A nested pin only flies out on a hard hit; on a gentle tip it stays
        # inside the (now lying) cup and can still be intaken with it.
        if self.contains_pin and impulse_mag < NEST_EJECT_IMPULSE:
            return
        if self.contains_pin:
            pin = self.contains_pin
            pin.carried_by = None
            pin.scored = False
            pin.is_nested = False   # ← IMPORTANT: allow it to be drawn again
            pin.orientation = ORIENT_HORIZONTAL
            pin.body.angle = hit_angle + 0.4

            import math
            # Spawn ~1 cup length away (more natural)
            spawn_distance = CUP_PHYS_HALF_LEN * 2.2
            offset_x = math.cos(hit_angle) * spawn_distance
            offset_y = math.sin(hit_angle) * spawn_distance
            pin.body.position = (
                self.body.position.x + offset_x,
                self.body.position.y + offset_y
            )
            pin.body.velocity = (0, 0)
            pin.body.angular_velocity = 3.0
            if pin.shape:
                pin.shape.filter = NORMAL_PIN_FILTER

            self.contains_pin = None
            #print("[Simulator] Nested pin separated from cup (via knock_over)")

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

    def flip(self):
        """Flip which side of the cup faces down (front of robot = down)."""
        self.flipped = not self.flipped

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

        # For cups: flipped reverses the meaning (front of robot = down)
        effective_clear_up = self.clear_on_top if not self.flipped else not self.clear_on_top

        if effective_clear_up:
            top_color = C_WHITE          # Clear on top
            bottom_color = C_GRAY_DARK   # Dark on bottom
        else:
            top_color = C_GRAY_DARK      # Dark on top
            bottom_color = C_WHITE       # Clear on bottom

        # Draw bottom half first
        pygame.draw.circle(surface, bottom_color, (cx, cy), r)

        # Draw top half (semicircle)
        points = []
        if effective_clear_up:
            for angle in range(180, 361):
                rad = math.radians(angle)
                x = cx + r * math.cos(rad)
                y = cy + r * math.sin(rad)
                points.append((x, y))
        else:
            for angle in range(0, 181):
                rad = math.radians(angle)
                x = cx + r * math.cos(rad)
                y = cy + r * math.sin(rad)
                points.append((x, y))

        points.append((cx, cy))
        pygame.draw.polygon(surface, top_color, points)

        # Outline and center line
        pygame.draw.circle(surface, C_OUTLINE, (cx, cy), r, 2)
        pygame.draw.line(surface, C_OUTLINE, (cx - r, cy), (cx + r, cy), 1)

        # Nested pin indicator (smaller so cup is visible underneath)
        if self.contains_pin:
            pin = self.contains_pin
            c1, c2 = pin.get_colors()
            inner_r = max(3, int(r * 0.7))   # Clearly smaller than the cup

            # Draw smaller bicolor pin
            if c1 == c2:
                pygame.draw.circle(surface, c1, (cx, cy), inner_r)
            else:
                # Bicolor split
                pygame.draw.circle(surface, c1, (cx, cy), inner_r)
                points = []
                for angle in range(180, 361):
                    rad = math.radians(angle)
                    x = cx + inner_r * math.cos(rad)
                    y = cy + inner_r * math.sin(rad)
                    points.append((x, y))
                points.append((cx, cy))
                pygame.draw.polygon(surface, c2, points)

            pygame.draw.circle(surface, C_OUTLINE, (cx, cy), inner_r, 1)

    def _draw_horizontal(self, surface, px, py):
        """Horizontal cup — clean minimalist look copied from your reference."""
        hl = PIN_DRAW_HALF_LEN * 0.88
        hw = CUP_DRAW_WIDTH_H * 1.3
        a = self.angle
        cos_a, sin_a = math.cos(a), math.sin(a)
        cx, cy = px, py
        px_perp = -sin_a
        py_perp = cos_a

        lx = cx - hl * cos_a
        ly = cy - hl * sin_a
        lx_top = lx + px_perp * (hw / 2)
        ly_top = ly + py_perp * (hw / 2)
        lx_bot = lx - px_perp * (hw / 2)
        ly_bot = ly - py_perp * (hw / 2)

        rx = cx + hl * cos_a
        ry = cy + hl * sin_a
        rx_top = rx + px_perp * (hw / 2)
        ry_top = ry + py_perp * (hw / 2)
        rx_bot = rx - px_perp * (hw / 2)
        ry_bot = ry - py_perp * (hw / 2)

        center_x, center_y = cx, cy

        # Determine colors based on flip state (white = clear side)
        effective_clear_up = self.clear_on_top if not self.flipped else not self.clear_on_top
        if effective_clear_up:
            color_left, color_right = C_WHITE, C_GRAY_DARK
        else:
            color_left, color_right = C_GRAY_DARK, C_WHITE

        # Small dark center circle
        pygame.draw.circle(surface, (40, 40, 45), (int(center_x), int(center_y)), 6)
        pygame.draw.circle(surface, C_OUTLINE, (int(center_x), int(center_y)), 6, 2)

        # Two triangles meeting at center
        pygame.draw.polygon(surface, color_left,
                            [(lx_top, ly_top), (lx_bot, ly_bot), (center_x, center_y)])
        pygame.draw.polygon(surface, color_right,
                            [(rx_top, ry_top), (rx_bot, ry_bot), (center_x, center_y)])

        # Outline
        outline_pts = [(lx_top, ly_top), (lx_bot, ly_bot),
                    (center_x, center_y),
                    (rx_top, ry_top), (rx_bot, ry_bot)]
        pygame.draw.polygon(surface, C_OUTLINE, outline_pts, 3)

        # Center dot
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
        from simulation.scoring import goal_region
        self.kind = ("center" if self.goal_id == CENTER_GOAL_ID else
                     "neutral" if self.alliance == "neutral" else "alliance")
        self.region = goal_region(self.goal_id, self.x, self.y, CENTER_GOAL_ID)

        # NEW: Proper ordered stack (bottom → top)
        self.stack = []          # list of (obj, is_pin) tuples
        self.stack_count = 0

        # Live per-alliance score cache (updated by get_score every frame)
        self.red_score  = 0
        self.blue_score = 0

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

    def add_to_stack(self, obj):
        """Add a scored object to the stack (pin or cup)."""
        is_pin = isinstance(obj, GamePin)
        self.stack.append((obj, is_pin))
        self.stack_count = len(self.stack)
    
    def stack_desc(self):
        """Stack as plain tuples for simulation.scoring."""
        out = []
        for obj, is_pin in self.stack:
            if is_pin:
                out.append(("pin", obj.down_half_name, obj.up_half_name))
            else:
                flipped = getattr(obj, "flipped", False)
                clear_up = (not obj.clear_on_top) if flipped else obj.clear_on_top
                out.append(("cup", clear_up))
        return out

    def get_score(self, toggles=None, midfield_majority=None, skills=False):
        """Per-alliance points for this goal (manual v2.0 SC2-SC5, RSC3).

        midfield_majority: owner of Midfield yellow pins — only known at match
          end (None/'tie' during play → Midfield yellows score 0 live).
        skills: Robot Skills rules (RSC3); everything is credited to red.
        Updates self.red_score / self.blue_score; returns the total.
        """
        from simulation.scoring import score_stack, TOGGLE_OF_QUADRANT
        toggle_owner = None
        if self.region != "midfield" and toggles:
            tid = TOGGLE_OF_QUADRANT[self.region]
            for t in toggles:
                if t.toggle_id == tid:
                    if t.owner in ("red", "blue"):
                        toggle_owner = t.owner
                    break
        mid_owner = midfield_majority if midfield_majority in ("red", "blue") else None
        pts = score_stack(self.stack_desc(), self.region, toggle_owner,
                          mid_owner, skills=skills)
        self.red_score = pts["red"]
        self.blue_score = pts["blue"]
        return self.red_score + self.blue_score

    def draw(self, surface, font_small):
        sx = int(self.x * RENDER_SCALE)
        sy = int(self.y * RENDER_SCALE)
        r  = int(self.radius * RENDER_SCALE)

        # Goal post colors
        if self.alliance == "red":
            inner_color  = (230, 80, 80)
            border_color = (160, 30, 35)
        elif self.alliance == "blue":
            inner_color  = (80, 130, 230)
            border_color = (25, 55, 160)
        else:
            inner_color  = (220, 220, 230)
            border_color = (140, 140, 155)

        post_r = max(3, r // 3) // 2 + 6
        pygame.draw.circle(surface, border_color, (sx, sy), post_r + 2)
        pygame.draw.circle(surface, inner_color,  (sx, sy), post_r)

        if not self.stack:
            score = self.red_score + self.blue_score
            if font_small and score > 0:
                txt = font_small.render(f"+{score}", True, (255, 230, 80))
                surface.blit(txt, (sx - txt.get_width() // 2, sy + 12))
            return

        # ── Helpers ──────────────────────────────────────────────────────
        # ── Helpers ──────────────────────────────────────────────────────
        def eff_clear_up(cup):
            """True = clear/white side is UP on the goal (dark went down)."""
            flipped = getattr(cup, 'flipped', False)
            # flipped = True means player flipped it → opposite side now faces forward (goes down)
            return not cup.clear_on_top if flipped else cup.clear_on_top

        HALF_R   = 6
        CUP_R    = 7
        LINE_HW  = 7
        LINE_T   = 3

        n = len(self.stack)
        vis = []
        for i, (obj, is_pin) in enumerate(self.stack):
            if not is_pin:
                vis.append(None)
                continue
            if i == 0:
                down_vis = True      # sits in the goal, still visible (SC3)
            else:
                prev_obj, prev_is_pin = self.stack[i - 1]
                down_vis = eff_clear_up(prev_obj) if not prev_is_pin else True
            if i + 1 >= n:
                up_vis = True
            else:
                next_obj, next_is_pin = self.stack[i + 1]
                up_vis = (not eff_clear_up(next_obj)) if not next_is_pin else True
            vis.append((down_vis, up_vis))

        # Start close to goal post
        y = sy - 7

        for i, (obj, is_pin) in enumerate(self.stack):
            if is_pin:
                down_vis, up_vis = vis[i]
                down_c = obj.get_down_color()
                up_c   = obj.get_up_color()

                # DOWN half
                down_y = y
                if down_vis:
                    pygame.draw.circle(surface, down_c,   (sx, down_y), HALF_R)
                    pygame.draw.circle(surface, C_OUTLINE, (sx, down_y), HALF_R, 1)
                    y = down_y - HALF_R          # Flush with bottom of circle
                else:
                    pygame.draw.line(surface, C_OUTLINE,
                                    (sx - LINE_HW - 1, down_y),
                                    (sx + LINE_HW + 1, down_y), LINE_T + 1)
                    pygame.draw.line(surface, down_c,
                                    (sx - LINE_HW, down_y),
                                    (sx + LINE_HW, down_y), LINE_T)
                    y = down_y - LINE_T          # Flush with bottom of line

                # UP half
                up_y = y
                if up_vis:
                    pygame.draw.circle(surface, up_c,     (sx, up_y), HALF_R)
                    pygame.draw.circle(surface, C_OUTLINE, (sx, up_y), HALF_R, 1)
                    y = up_y - HALF_R
                else:
                    pygame.draw.line(surface, C_OUTLINE,
                                    (sx - LINE_HW - 1, up_y),
                                    (sx + LINE_HW + 1, up_y), LINE_T + 1)
                    pygame.draw.line(surface, up_c,
                                    (sx - LINE_HW, up_y),
                                    (sx + LINE_HW, up_y), LINE_T)
                    y = up_y - LINE_T

            else:  # GameCup
                cup_y = y - CUP_R
                cup_col = C_WHITE if eff_clear_up(obj) else C_GRAY_DARK
                pygame.draw.circle(surface, cup_col,   (sx, cup_y), CUP_R)
                pygame.draw.circle(surface, C_OUTLINE, (sx, cup_y), CUP_R, 1)
                y = cup_y - CUP_R

        # Score label
        score = self.red_score + self.blue_score   # cached by rules engine
        if font_small and score > 0:
            txt = font_small.render(f"+{score}", True, (255, 230, 80))
            surface.blit(txt, (sx - txt.get_width() // 2, sy + 12))

#works

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
        """Single-flip alliance takeover (matches Robot.try_toggle).
        Pressing toggle near a non-own toggle sets it to the robot's
        alliance.  Own toggles are no-op (closes the cycle exploit
        where flipping your own toggle would give it to the opponent).
        """
        dist = math.sqrt((robot.body.position.x - self.x) ** 2 +
                         (robot.body.position.y - self.y) ** 2)
        if dist <= TOGGLE_INTERACTION_RANGE:
            if self.owner == robot.alliance:
                return False
            self.owner = robot.alliance
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
    nested_pins = []   # ← NEW: collect nested pins
    for i, c in enumerate(CUP_STARTS):
        cup = GameCup(
            i, 
            c["side"], 
            c["x"], 
            c["y"], 
            space,
            c.get("orientation", ORIENT_VERTICAL),
            c.get("angle", 0.0),
            c.get("contains_pin", None),
            c.get("clear_up", True),
        )
        cups.append(cup)
        
        # If this cup has a nested pin, add it to the pins list so it gets drawn
        if cup.contains_pin:
            nested_pins.append(cup.contains_pin)

    # SG7b: the 28 Scoring Objects that begin ON the Autonomous Line (the four
    # pin+cup clusters on the y = x tape and the four red/blue pin cups on
    # the Midfield tape) may be used by either alliance during auton.
    for obj in pins + nested_pins + cups:
        obj.on_auton_line = starts_on_auton_line(obj.body.position.x,
                                                 obj.body.position.y)

    # Return both normal pins + nested pins
    return pins + nested_pins, cups


def starts_on_auton_line(x: float, y: float) -> bool:
    if abs(y - x) / math.sqrt(2) <= 6.0:            # cluster on the y = x tape
        return True
    l1 = abs(x - 72) + abs(y - 72)
    return abs(l1 - 24) <= 2.0                      # on the Midfield tape

def create_goals(space: pymunk.Space = None):
    return [FieldGoal(g, space) for g in GOALS]

def create_toggles():
    return [FieldToggle(t) for t in TOGGLES]