import math

# =============================================================================
# FIELD DIMENSIONS
# =============================================================================
FIELD_WIDTH = 144.0
FIELD_HEIGHT = 144.0
TILE_SIZE = 24.0

# =============================================================================
# RENDERING
# =============================================================================
RENDER_SCALE = 5.0
SCREEN_W = int(FIELD_WIDTH * RENDER_SCALE)
SCREEN_H = int(FIELD_HEIGHT * RENDER_SCALE)

def s(inches):
    return inches * RENDER_SCALE

# =============================================================================
# MATCH TIMING
# =============================================================================
AUTONOMOUS_SECONDS = 15
DRIVER_SECONDS = 105
SETTLE_SECONDS = 5
TOTAL_SECONDS = AUTONOMOUS_SECONDS + DRIVER_SECONDS

# =============================================================================
# ROBOT CONSTANTS
# =============================================================================
MAX_ROBOT_SIZE_START = 18.0
ROBOT_MASS = 5.0
ROBOT_DRIVE_FORCE = 3000.0
ROBOT_TURN_TORQUE = 15000.0
ROBOT_MAX_SPEED = 400.0
INTAKE_RADIUS = 10.0    # v6: reduced from 14 — robots must approach precisely
SCORING_RADIUS = 12.0   # v6: reduced from 16 — tighter scoring contact required
MAX_PINS_HELD = 1
MAX_CUPS_HELD = 1

# =============================================================================
# SCORING (per official VEX Override manual)
# =============================================================================
POINTS_PIN_IN_GOAL = 5          # Alliance pin placed in goal
POINTS_CUP_IN_GOAL = 0          # Cups alone are worth NOTHING
POINTS_STACK_BONUS = 3          # Bonus per additional properly nested pin (requires cup between pins)
POINTS_YELLOW_OWNED = 10        # Yellow pin owned via toggle control
POINTS_MIDFIELD_PARK = 8
POINTS_AUTONOMOUS_BONUS = 12

# =============================================================================
# FOULS
# =============================================================================
FOUL_PINNING_SECONDS = 3.0
FOUL_STANDARD_PTS = 5
FOUL_TECHNICAL_PTS = 10

FOULS = {
    "excessive_pinning": {"points": FOUL_STANDARD_PTS, "description": "Pinning opponent > 3 seconds"},
    "illegal_possession": {"points": FOUL_STANDARD_PTS, "description": "Holding more than 1 Pin + 1 Cup"},
    "illegal_expansion": {"points": FOUL_TECHNICAL_PTS, "description": "Expanding beyond legal limits"},
}

# =============================================================================
# ROBOT STARTING POSITIONS
# =============================================================================
ROBOT_STARTS = {
    "red1": {"pos": (10, 72), "angle": 0.0},
    "red2": {"pos": (72, 10), "angle": math.pi / 2},
    "blue1": {"pos": (72, 136), "angle": math.pi / 2 * 3},
    "blue2": {"pos": (136, 72), "angle": math.pi},
}

# =============================================================================
# GOALS
# =============================================================================
GOALS = [
    {"id": 0, "alliance": "red", "x": 48, "y": 120, "radius": 10, "label": "R-Low"},
    {"id": 1, "alliance": "red", "x": 24, "y": 96, "radius": 10, "label": "R-High"},
    {"id": 2, "alliance": "blue", "x": 96, "y": 24, "radius": 10, "label": "B-Low"},
    {"id": 3, "alliance": "blue", "x": 120, "y": 48, "radius": 10, "label": "B-High"},
    {"id": 4, "alliance": "neutral", "x": 72, "y": 72, "radius": 12, "label": "Center"},
    {"id": 5, "alliance": "neutral", "x": 48, "y": 24, "radius": 10, "label": "NW"},
    {"id": 6, "alliance": "neutral", "x": 24, "y": 48, "radius": 10, "label": "SW"},
    {"id": 7, "alliance": "neutral", "x": 96, "y": 120, "radius": 10, "label": "SE"},
    {"id": 8, "alliance": "neutral", "x": 120, "y": 96, "radius": 10, "label": "NE"},
]


# =============================================================================
# TOGGLES (Rollers) - Zero wall offset (flush against each wall)
# =============================================================================
TOGGLES = [
    # Left wall - vertical toggle (flush)
    {"id": 1, "x": 1,                    "y": FIELD_HEIGHT/2, "owner": "yellow", "quadrant": "red",  "orientation": "vertical"},
    # Right wall - vertical toggle (flush)
    {"id": 2, "x": FIELD_WIDTH - 1,      "y": FIELD_HEIGHT/2, "owner": "yellow", "quadrant": "blue", "orientation": "vertical"},
    # Top wall - horizontal toggle (flush)
    {"id": 3, "x": FIELD_WIDTH/2,         "y": 1,              "owner": "yellow", "quadrant": "red",  "orientation": "horizontal"},
    # Bottom wall - horizontal toggle (flush)
    {"id": 4, "x": FIELD_WIDTH/2,         "y": FIELD_HEIGHT - 1, "owner": "yellow", "quadrant": "blue", "orientation": "horizontal"},
]

TOGGLE_INTERACTION_RANGE = 18.0

# =============================================================================
# MIDFIELD
# =============================================================================
MIDFIELD_CENTER = (72, 72)
MIDFIELD_HALF = 24

# =============================================================================
# ENDGAME (last 20 seconds of driver period)
# =============================================================================
ENDGAME_SECONDS          = 20  # duration of endgame period
CENTER_GOAL_ID           = 4   # goal id of the central/midfield goal
# Central goal stack height limit during endgame: Pin + Cup + Pin = 3 items.
# Once the central goal stack reaches this, no more scoring on it during endgame.
ENDGAME_CENTER_MAX_STACK = 3

# =============================================================================
# ORIENTATION CONSTANTS
# =============================================================================
ORIENT_VERTICAL = "vertical"
ORIENT_HORIZONTAL = "horizontal"

# =============================================================================
# PIN STARTING POSITIONS
# =============================================================================
PIN_STARTS = [
    # Top wall
    {"color": "red_yellow", "x": 18, "y": 24, "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    {"color": "red_yellow", "x": 24, "y": 30, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    {"color": "blue_yellow", "x": 30, "y": 24, "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    {"color": "blue_yellow", "x": 24, "y": 18, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
    
    {"color": "red_yellow",  "x": 42, "y": 48,  "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    {"color": "red_yellow",  "x": 48, "y": 54,  "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    {"color": "blue_yellow", "x": 54, "y": 48,  "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    {"color": "blue_yellow", "x": 48, "y": 42,  "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
        
    {"color": "red_yellow",  "x": 90, "y": 96,  "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    {"color": "red_yellow",  "x": 96, "y": 102, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    {"color": "blue_yellow", "x": 102, "y": 96,  "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    {"color": "blue_yellow", "x": 96, "y": 90,  "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
        
    {"color": "red_yellow",  "x": 114, "y": 120, "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    {"color": "red_yellow",  "x": 120, "y": 126, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    {"color": "blue_yellow", "x": 126, "y": 120, "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    {"color": "blue_yellow", "x": 120, "y": 114, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
]

# =============================================================================
# CUP STARTING POSITIONS
# =============================================================================
CUP_STARTS = [
    # Top Left/Bottem right diagonal cups
    # Normal loose cups (no nested pin)
    {"side": "gray", "x": 24, "y": 24, "orientation": ORIENT_VERTICAL, "angle": -math.pi / 2},
    {"side": "gray", "x": 48, "y": 48, "orientation": ORIENT_VERTICAL, "angle": math.pi / 2},
    {"side": "gray", "x": 96, "y": 96, "orientation": ORIENT_VERTICAL, "angle": math.pi / 2},
    {"side": "gray", "x": 120, "y": 120, "orientation": ORIENT_VERTICAL, "angle": math.pi / 2},

    # NEW: Cup with nested pin at start (vertical)
    {"side": "gray", "x": 0, "y": 46, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "x": 0, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 0, "y": 50, "orientation": ORIENT_VERTICAL, "angle": 0.0},

    {"side": "gray", "x": 0, "y": 93, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "x": 0, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 0, "y": 98, "orientation": ORIENT_VERTICAL, "angle": 0.0},


    {"side": "gray", "x": 144, "y": 45, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "x": 144, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 144, "y": 50, "orientation": ORIENT_VERTICAL, "angle": 0.0},

    {"side": "gray", "x": 144, "y": 93, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "x": 144, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 144, "y": 98, "orientation": ORIENT_VERTICAL, "angle": 0.0},


    {"side": "gray", "x": 46, "y": 5, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "x": 48, "y": 5, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 50, "y": 5, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    
    {"side": "gray", "x": 94, "y": 5, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "x": 96, "y": 5, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 98, "y": 5, "orientation": ORIENT_VERTICAL, "angle": 0.0},


    {"side": "gray", "x": 46, "y": 144, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "x": 48, "y": 144, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 50, "y": 144, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    
    {"side": "gray", "x": 94, "y": 144, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "x": 96, "y": 144, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 98, "y": 144, "orientation": ORIENT_VERTICAL, "angle": 0.0},


    {"side": "gray", "x": 24, "y": 120, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 48, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 96, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "x": 120, "y": 24, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
]

# =============================================================================
# MATCH LOADING AREAS (corners)
# =============================================================================
LOADING_ZONE_WIDTH  = 12.0   # Half tile (X dimension)
LOADING_ZONE_HEIGHT = 24.0   # 1 full tile (Y dimension)

# Red alliance (left side)
RED_LOADING_ZONES = [
    {"x": 0, "y": 0},      # Top-left
    {"x": 0, "y": 120},    # Bottom-left (144 - 24)
]

# Blue alliance (right side)
BLUE_LOADING_ZONES = [
    {"x": 132, "y": 0},    # Top-right (144 - 12)
    {"x": 132, "y": 120},  # Bottom-right
]

# =============================================================================
# MATCH LOADING INVENTORY
# =============================================================================
MATCH_LOADS_CUPS = 10
MATCH_LOADS_ALLIANCE_PINS = 12
MATCH_LOADS_YELLOW_PINS = 1