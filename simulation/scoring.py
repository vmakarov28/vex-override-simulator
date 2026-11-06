"""
simulation/scoring.py
─────────────────────
Pure scoring functions for VEX V5RC Override (game manual v2.0).

No pygame / pymunk imports, so the physics simulator AND the route planner
(strategy/) both use exactly the same rules.

Stack model
-----------
A goal stack is a bottom→top list of items:
    ("pin", down_half, up_half)      e.g. ("pin", "yellow", "red")
    ("cup", clear_up)                clear_up=True → transparent half on top
Legal stacks alternate pin, cup, pin, cup ... starting with a pin.

Visibility (SC3): a pin half is Scored unless it sits inside the OPAQUE half
of a cup.
  * bottom pin's DOWN half sits in the Goal → visible (Figure SC5 shows the
    bottom yellow half of the stack earning +10).
  * a cup's lower half sheaths the pin below's UP half; its upper half
    sheaths the pin above's DOWN half.  Exactly one of those two halves is
    hidden by every cup.
  * the top pin's UP half is in open air → visible.

So N pins + (N-1) cups on one goal show N+1 halves.  A new (cup, pin) pair
on an existing stack is worth exactly ONE extra half, while a pin on an
empty goal is worth TWO halves.
"""

from typing import List, Optional, Sequence, Tuple, Dict

PTS_ALLIANCE_HALF = 5
PTS_YELLOW_HALF = 10
PTS_MIDFIELD_ROBOT = 8
PTS_AUTON_BONUS = 12
PTS_AUTON_TIE = 6

CENTER = (72.0, 72.0)

# Quadrant → toggle id, and the colour of each Quadrant (Figure Q-1)
TOGGLE_OF_QUADRANT = {"left": 1, "right": 2, "top": 3, "bottom": 4}
QUADRANT_COLOR = {"left": "red", "bottom": "red", "top": "blue", "right": "blue"}


def quadrant_of(x: float, y: float) -> str:
    """Which triangular Quadrant (left/right/top/bottom) a point is in."""
    dx, dy = x - CENTER[0], y - CENTER[1]
    if abs(dx) >= abs(dy):
        return "left" if dx <= 0 else "right"
    return "top" if dy <= 0 else "bottom"


def goal_region(goal_id: int, x: float, y: float, center_goal_id: int = 4) -> str:
    """'midfield' for the center goal, else the Quadrant name."""
    if goal_id == center_goal_id:
        return "midfield"
    return quadrant_of(x, y)


def half_visibility(stack: Sequence[tuple]) -> List[Optional[Tuple[bool, bool]]]:
    """Per stack index: (down_visible, up_visible) for pins, None for cups."""
    n = len(stack)
    out: List[Optional[Tuple[bool, bool]]] = []
    for i, item in enumerate(stack):
        if item[0] != "pin":
            out.append(None)
            continue
        # DOWN half
        if i == 0:
            down_vis = True            # nested in the Goal, still visible
        else:
            below = stack[i - 1]
            # cup below: its UPPER half holds this pin's down half.
            down_vis = bool(below[1]) if below[0] == "cup" else True
        # UP half
        if i + 1 >= n:
            up_vis = True
        else:
            above = stack[i + 1]
            # cup above: its LOWER half holds this pin's up half; the lower
            # half is clear exactly when clear_up is False.
            up_vis = (not above[1]) if above[0] == "cup" else True
        out.append((down_vis, up_vis))
    return out


def score_stack(stack: Sequence[tuple], region: str,
                toggle_owner: Optional[str],
                midfield_owner: Optional[str],
                skills: bool = False) -> Dict[str, int]:
    """Points for one goal.

    region          'midfield' or a Quadrant name
    toggle_owner    owner of that Quadrant's toggle ('red'/'blue'/None)
    midfield_owner  head-to-head: alliance with MORE robots in the Midfield at
                    match end (None during play / tie).  Skills: 'red' if the
                    robot ends in the Midfield, else None.
    skills          apply RSC3 restrictions (red halves only in red Quadrants
                    or Midfield, blue only in blue Quadrants or Midfield,
                    toggles only confer ownership when the toggle colour
                    matches the Quadrant colour).  All points go to 'red'.
    """
    red = blue = 0
    if region == "midfield":
        yellow_owner = midfield_owner
    else:
        yellow_owner = toggle_owner
        if skills and toggle_owner != QUADRANT_COLOR[region]:
            yellow_owner = None
    vis = half_visibility(stack)
    for item, v in zip(stack, vis):
        if v is None:
            continue
        down_vis, up_vis = v
        for visible, colour in ((down_vis, item[1]), (up_vis, item[2])):
            if not visible:
                continue
            if colour == "red":
                if skills:
                    if region == "midfield" or QUADRANT_COLOR[region] == "red":
                        red += PTS_ALLIANCE_HALF
                else:
                    red += PTS_ALLIANCE_HALF
            elif colour == "blue":
                if skills:
                    if region == "midfield" or QUADRANT_COLOR[region] == "blue":
                        red += PTS_ALLIANCE_HALF
                else:
                    blue += PTS_ALLIANCE_HALF
            elif colour == "yellow":
                if yellow_owner == "red" or (skills and yellow_owner):
                    red += PTS_YELLOW_HALF
                elif yellow_owner == "blue":
                    blue += PTS_YELLOW_HALF
    return {"red": red, "blue": blue}


def scored_halves(stack: Sequence[tuple], region: str,
                  toggle_owner: Optional[str], alliance: str) -> int:
    """Number of Scored pin halves for `alliance` in one goal during the
    Autonomous Period (used for the Autonomous Win Point).  Midfield yellow
    ownership is excluded from auton scoring (SC7a)."""
    n = 0
    vis = half_visibility(stack)
    for item, v in zip(stack, vis):
        if v is None:
            continue
        for visible, colour in ((v[0], item[1]), (v[1], item[2])):
            if not visible:
                continue
            if colour == alliance:
                n += 1
            elif colour == "yellow" and region != "midfield" and toggle_owner == alliance:
                n += 1
    return n


def can_place(stack: Sequence[tuple], kind: str) -> bool:
    """Stack legality: pin on empty goal or on a cup; cup only on a pin."""
    if kind == "pin":
        return (not stack) or stack[-1][0] == "cup"
    return bool(stack) and stack[-1][0] == "pin"


def stack_height(goal_kind: str, n_pins: int, goal_heights: Dict[str, float],
                 pin_h: float = 6.5) -> float:
    return goal_heights[goal_kind] + pin_h * n_pins
