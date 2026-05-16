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
ROBOT_DRIVE_FORCE = 900.0
ROBOT_TURN_TORQUE = 8000.0
ROBOT_MAX_SPEED = 120.0
INTAKE_RADIUS = 14.0
SCORING_RADIUS = 16.0
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
# ORIENTATION CONSTANTS
# =============================================================================
ORIENT_VERTICAL = "vertical"
ORIENT_HORIZONTAL = "horizontal"

# =============================================================================
# PIN STARTING POSITIONS
# =============================================================================
PIN_STARTS = [
    # Top wall
    {"color": "red_yellow", "x": 36, "y": 80, "orientation": ORIENT_VERTICAL, "angle": math.pi / 2},
    # {"color": "red_yellow", "x": 60, "y": 6, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
    # {"color": "yellow", "x": 72, "y": 6, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
    # {"color": "blue_yellow", "x": 84, "y": 6, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
    # {"color": "blue_yellow", "x": 108, "y": 6, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
    # # Bottom wall
    # {"color": "red_yellow", "x": 36, "y": 138, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    # {"color": "red_yellow", "x": 60, "y": 138, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    # {"color": "yellow", "x": 72, "y": 138, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    # {"color": "blue_yellow", "x": 84, "y": 138, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    # {"color": "blue_yellow", "x": 108, "y": 138, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    # # Left wall
    # {"color": "red_yellow", "x": 6, "y": 60, "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    # {"color": "yellow", "x": 6, "y": 72, "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    # {"color": "red_yellow", "x": 6, "y": 84, "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    # # Right wall
    # {"color": "blue_yellow", "x": 138, "y": 60, "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    # {"color": "yellow", "x": 138, "y": 72, "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    # {"color": "blue_yellow", "x": 138, "y": 84, "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    # # Interior - Vertical
    # {"color": "red", "x": 36, "y": 36, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 48, "y": 24, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 24, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "blue", "x": 108, "y": 36, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 96, "y": 24, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 120, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "red", "x": 36, "y": 108, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 48, "y": 120, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 24, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "blue", "x": 108, "y": 108, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 96, "y": 120, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 120, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # # Center cluster
    # {"color": "red_yellow", "x": 60, "y": 60, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "blue_yellow", "x": 84, "y": 60, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "red_yellow", "x": 60, "y": 84, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "blue_yellow", "x": 84, "y": 84, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 72, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 72, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 48, "y": 72, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    # {"color": "yellow", "x": 96, "y": 72, "orientation": ORIENT_VERTICAL, "angle": 0.0},
]

# =============================================================================
# CUP STARTING POSITIONS
# =============================================================================
CUP_STARTS = [
    # # Top
    {"side": "white", "x": 80, "y": 80, "orientation": ORIENT_VERTICAL, "angle": math.pi / 2},
    # {"side": "white", "x": 24, "y": 12, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
    # {"side": "gray", "x": 48, "y": 12, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
    # {"side": "white", "x": 96, "y": 12, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
    # {"side": "gray", "x": 120, "y": 12, "orientation": ORIENT_HORIZONTAL, "angle": math.pi / 2},
    # # Bottom
    # {"side": "white", "x": 24, "y": 132, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    # {"side": "gray", "x": 48, "y": 132, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    # {"side": "white", "x": 96, "y": 132, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    # {"side": "gray", "x": 120, "y": 132, "orientation": ORIENT_HORIZONTAL, "angle": -math.pi / 2},
    # # Left
    # {"side": "white", "x": 12, "y": 24, "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    # {"side": "gray", "x": 12, "y": 48, "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    # {"side": "white", "x": 12, "y": 96, "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    # {"side": "gray", "x": 12, "y": 120, "orientation": ORIENT_HORIZONTAL, "angle": 0.0},
    # # Right
    # {"side": "white", "x": 132, "y": 24, "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    # {"side": "gray", "x": 132, "y": 48, "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    # {"side": "white", "x": 132, "y": 96, "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
    # {"side": "gray", "x": 132, "y": 120, "orientation": ORIENT_HORIZONTAL, "angle": math.pi},
]