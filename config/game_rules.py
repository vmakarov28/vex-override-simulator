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

# Robot Skills (single-robot solo challenge per VEX rules).
# Standard VEX Skills run = 60 seconds, one robot on the field, no alliance
# opposition.  All elements/goals available; the player scores every visible
# pin half regardless of color (red/blue/yellow).
SKILLS_SECONDS = 60

# =============================================================================
# ROBOT CONSTANTS
# =============================================================================
MAX_ROBOT_SIZE_START = 18.0
ROBOT_MASS = 5.0
ROBOT_DRIVE_FORCE = 5000.0
ROBOT_TURN_TORQUE = 25000.0
ROBOT_MAX_SPEED = 700.0
INTAKE_RADIUS = 10.0    # v6: reduced from 14 — robots must approach precisely
SCORING_RADIUS = 12.0   # v6: reduced from 16 — tighter scoring contact required
MAX_PINS_HELD = 1
MAX_CUPS_HELD = 1

# =============================================================================
# SCORING (official V5RC Override game manual v2.0, Sept 3 2026)
# =============================================================================
# Scoring table:   each Scored red/blue Pin HALF ..... 5
#                  each Scored (Owned) yellow HALF .... 10
#                  each Robot ending in the Midfield .. 8
#                  Autonomous Bonus ................... 12  (tie: 6 each)
# A Pin half is Scored if the Pin is Placed and that half is not nested in
# the OPAQUE half of a Cup (SC3).  A Pin sitting directly on a Goal shows
# BOTH halves (manual Figure SC5: the bottom yellow half earns +10).
POINTS_PIN_IN_GOAL = 5          # per visible red/blue half
POINTS_CUP_IN_GOAL = 0          # Cups alone are worth NOTHING
POINTS_STACK_BONUS = 0          # There is NO stack bonus in the manual (was 3)
POINTS_YELLOW_OWNED = 10        # per visible Owned yellow half
POINTS_MIDFIELD_PARK = 8        # per Robot at least partially in Midfield at match END (SC6)
POINTS_AUTONOMOUS_BONUS = 12
POINTS_AUTONOMOUS_TIE = 6       # SC7b: a tie (incl. 0-0) gives each alliance 6

# Autonomous Win Point (SC8, v2.0 standard events).  "Pins" = Scored halves,
# counting only Quadrants on your own side of the Autonomous Line.
AWP_MIN_PINS = 6
AWP_MIN_GOALS_WITH_2 = 2
AWP_MIN_PINS_WORLDS = 7
AWP_MIN_GOALS_WITH_2_WORLDS = 3

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
# SG1: each Robot starts in its own Quadrant, touching the Field Perimeter on
# its Alliance's side of the Autonomous Line.  The red Alliance Station is the
# LEFT wall; red Quadrants are LEFT (x small) and BOTTOM (y large); blue
# Quadrants are TOP and RIGHT (manual Figures SG-1 / Q-1).
# (Before this fix red2 started at the TOP wall and blue1 at the BOTTOM wall,
#  i.e. inside the opponent's Quadrants, which SG1d/f does not allow.)
# angle = heading of the robot FRONT (front faces into the field).
ROBOT_STARTS = {
    "red1":  {"pos": (9.5, 72),   "angle": 0.0},            # left wall, faces +x
    "red2":  {"pos": (72, 134.5), "angle": -math.pi / 2},   # bottom wall, faces -y
    "blue1": {"pos": (72, 9.5),   "angle": math.pi / 2},    # top wall, faces +y
    "blue2": {"pos": (134.5, 72), "angle": math.pi},        # right wall, faces -x
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
    # Top wall - horizontal toggle (flush) — BLUE Quadrant (contains B-Low)
    {"id": 3, "x": FIELD_WIDTH/2,         "y": 1,              "owner": "yellow", "quadrant": "blue", "orientation": "horizontal"},
    # Bottom wall - horizontal toggle (flush) — RED Quadrant (contains R-Low)
    {"id": 4, "x": FIELD_WIDTH/2,         "y": FIELD_HEIGHT - 1, "owner": "yellow", "quadrant": "red",  "orientation": "horizontal"},
]
# Seconds a robot must push on a toggle to rotate it to a new set state.
# (The old sim flipped toggles instantly.)
TOGGLE_SET_TIME = 0.6

TOGGLE_INTERACTION_RANGE = 18.0

# =============================================================================
# MIDFIELD
# =============================================================================
MIDFIELD_CENTER = (72, 72)
MIDFIELD_HALF = 24

# =============================================================================
# ENDGAME (last 10 seconds of a Head-to-Head Match — glossary "Endgame")
# =============================================================================
ENDGAME_SECONDS          = 10  # was 20; the manual says the last 10 s
CENTER_GOAL_ID           = 4   # goal id of the central/midfield goal
# SG12 (v2.0): Robots may not Place ANY Scoring Object on the Midfield Goal
# during the Endgame.  (The old "lock after Pin+Cup+Pin" rule is not in the
# manual.)  0 = no placements allowed during Endgame.
ENDGAME_CENTER_MAX_STACK = 0

# Goal heights (inches) from the glossary.  Each Placed Pin adds ~6.5" to a
# stack (cups sheath the pins, so they add no extra height).
GOAL_HEIGHT = {"alliance": 3.25, "neutral": 5.8, "center": 8.7}
PIN_STACK_HEIGHT = 6.5

# =============================================================================
# AUTONOMOUS LINE (SG7 / SG8)
# =============================================================================
# The Autonomous Line is the pair of white tapes from the top-left corner to
# the bottom-right corner (the line y = x) plus the tape square around the
# Midfield.  Red side = y > x (left + bottom Quadrants), blue side = y < x.
# During auton a robot may touch tiles/objects on its own side, ON the line,
# or in the Midfield, but never the opponent's side.
AUTON_LINE_HALF_WIDTH = 1.5     # half the width of the double-tape band (in)

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
    {"side": "gray", "clear_up": True, "x": 24, "y": 24, "orientation": ORIENT_VERTICAL, "angle": -math.pi / 2},
    {"side": "gray", "clear_up": True, "x": 48, "y": 48, "orientation": ORIENT_VERTICAL, "angle": math.pi / 2},
    {"side": "gray", "clear_up": True, "x": 96, "y": 96, "orientation": ORIENT_VERTICAL, "angle": math.pi / 2},
    {"side": "gray", "clear_up": True, "x": 120, "y": 120, "orientation": ORIENT_VERTICAL, "angle": math.pi / 2},

    # NEW: Cup with nested pin at start (vertical)
    {"side": "gray", "clear_up": False, "x": 2, "y": 44.8, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "clear_up": False, "x": 2, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": False, "x": 2, "y": 51.2, "orientation": ORIENT_VERTICAL, "angle": 0.0},

    {"side": "gray", "clear_up": False, "x": 2, "y": 92.8, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "clear_up": False, "x": 2, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": False, "x": 2, "y": 99.2, "orientation": ORIENT_VERTICAL, "angle": 0.0},


    {"side": "gray", "clear_up": False, "x": 142, "y": 44.8, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "clear_up": False, "x": 142, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": False, "x": 142, "y": 51.2, "orientation": ORIENT_VERTICAL, "angle": 0.0},

    {"side": "gray", "clear_up": False, "x": 142, "y": 92.8, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "clear_up": False, "x": 142, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": False, "x": 142, "y": 99.2, "orientation": ORIENT_VERTICAL, "angle": 0.0},


    {"side": "gray", "clear_up": False, "x": 44.8, "y": 2, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "clear_up": False, "x": 48, "y": 2, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": False, "x": 51.2, "y": 2, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    
    {"side": "gray", "clear_up": False, "x": 92.8, "y": 2, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "clear_up": False, "x": 96, "y": 2, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": False, "x": 99.2, "y": 2, "orientation": ORIENT_VERTICAL, "angle": 0.0},


    {"side": "gray", "clear_up": False, "x": 44.8, "y": 142, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "clear_up": False, "x": 48, "y": 142, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": False, "x": 51.2, "y": 142, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    
    {"side": "gray", "clear_up": False, "x": 92.8, "y": 142, "orientation": ORIENT_VERTICAL, "angle": 0.0},
    {"side": "gray", "clear_up": False, "x": 96, "y": 142, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": False, "x": 99.2, "y": 142, "orientation": ORIENT_VERTICAL, "angle": 0.0},


    {"side": "gray", "clear_up": True, "x": 24, "y": 120, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": True, "x": 48, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": True, "x": 96, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},
    {"side": "gray", "clear_up": True, "x": 120, "y": 24, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "yellow_yellow"},


    {"side": "gray", "clear_up": True, "x": 48, "y": 72, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "blue_red"},
    {"side": "gray", "clear_up": True, "x": 96, "y": 72, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "red_blue"},
    {"side": "gray", "clear_up": True, "x": 72, "y": 48, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "red_blue"},
    {"side": "gray", "clear_up": True, "x": 72, "y": 96, "orientation": ORIENT_VERTICAL, "angle": 0.0, "contains_pin": "blue_red"},
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
# MATCH LOADING INVENTORY (per alliance; Preloads are SEPARATE)
# =============================================================================
MATCH_LOADS_CUPS = 10
MATCH_LOADS_ALLIANCE_PINS = 10   # was 12, and preloads were wrongly taken from it
MATCH_LOADS_YELLOW_PINS = 1
# Robot Skills (RSC4b): 3 red/yellow, 4 blue/yellow, 7 cups, red loaders only
SKILLS_LOADS_CUPS = 7
SKILLS_LOADS_RED_PINS = 3
SKILLS_LOADS_BLUE_PINS = 4
SKILLS_LOADS_YELLOW_PINS = 0
# Seconds for a drive-team member to drop an element into a Loader and for it
# to reach the loader's bottom opening.
MATCH_LOAD_TIME = 1.0

# =============================================================================
# PRE-PLACED PINS (Head-to-Head only — manual Figure FO-2)
# =============================================================================
# Each of the 5 neutral Goals starts the Match with one yellow/yellow Pin
# already Placed on it (17 yellow pins on field = 5 on goals + 8 in wall cups
# + 4 in diagonal cups).  In Robot Skills all Goals start empty (RSC4c).
PREPLACED_GOAL_PINS = {4: "yellow_yellow", 5: "yellow_yellow", 6: "yellow_yellow",
                       7: "yellow_yellow", 8: "yellow_yellow"}