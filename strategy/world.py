"""
strategy/world.py
─────────────────
Fast, abstract model of the Override field for route search.

It is built from the SAME config the physics simulator uses
(config/game_rules.py) and scores with the SAME functions
(simulation/scoring.py), so a plan's predicted score is exactly what the
physics sim awards if the robot executes the plan.

Coordinates: inches, x → right, y → down (red Alliance Station = left wall).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import game_rules as GR
from config.robot_profile import RobotProfile
from simulation.scoring import (score_stack, goal_region, QUADRANT_COLOR,
                                TOGGLE_OF_QUADRANT, quadrant_of, scored_halves,
                                can_place)


def _pin_halves(color: str, flipped: bool = False) -> Tuple[str, str]:
    """(down, up) half colours for a pin colour string, like GamePin."""
    names = {"red": ("red", "red"), "blue": ("blue", "blue"),
             "yellow": ("yellow", "yellow"), "yellow_yellow": ("yellow", "yellow"),
             "red_yellow": ("red", "yellow"), "blue_yellow": ("blue", "yellow"),
             "red_blue": ("red", "blue"), "blue_red": ("blue", "red")}
    c1, c2 = names[color]
    up = c2 if flipped else c1
    down = c1 if flipped else c2
    return (down, up)


def auton_side_ok(alliance: str, x: float, y: float) -> bool:
    w = GR.AUTON_LINE_HALF_WIDTH
    if abs(x - 72) + abs(y - 72) <= GR.MIDFIELD_HALF:
        return True
    return (y >= x - w) if alliance == "red" else (y <= x + w)


def starts_on_auton_line(x: float, y: float) -> bool:
    if abs(y - x) / math.sqrt(2) <= 6.0:
        return True
    return abs(abs(x - 72) + abs(y - 72) - 24) <= 2.0


@dataclass
class FieldObj:
    oid: int
    kind: str                      # "pin" or "cup"
    x: float
    y: float
    vertical: bool
    pin: Optional[Tuple[str, str]] = None      # (down, up) for a pin
    nested: Optional[Tuple[str, str]] = None   # nested pin (down, up) for a cup
    nested_color: Optional[str] = None
    color: Optional[str] = None                # pin colour string
    clear_up: bool = True
    on_line: bool = False
    label: str = ""


@dataclass
class GoalInfo:
    gid: int
    x: float
    y: float
    alliance: str
    kind: str          # alliance / neutral / center
    region: str        # quadrant name or "midfield"
    label: str


@dataclass
class ToggleInfo:
    tid: int
    x: float
    y: float
    quadrant_color: str
    nx: float          # inward unit normal
    ny: float


class World:
    """Static description of one game mode ("match" or "skills")."""

    def __init__(self, mode: str = "match", alliance: str = "red"):
        assert mode in ("match", "skills")
        self.mode = mode
        self.skills = mode == "skills"
        self.alliance = alliance
        self.objects: List[FieldObj] = []
        oid = 0
        for p in GR.PIN_STARTS:
            self.objects.append(FieldObj(
                oid, "pin", p["x"], p["y"],
                vertical=p.get("orientation") == GR.ORIENT_VERTICAL,
                pin=_pin_halves(p["color"]), color=p["color"],
                on_line=starts_on_auton_line(p["x"], p["y"]),
                label=f"{p['color']} pin @({p['x']:.0f},{p['y']:.0f})"))
            oid += 1
        for c in GR.CUP_STARTS:
            nested = c.get("contains_pin")
            self.objects.append(FieldObj(
                oid, "cup", c["x"], c["y"],
                vertical=c.get("orientation") == GR.ORIENT_VERTICAL,
                nested=_pin_halves(nested) if nested else None,
                nested_color=nested,
                clear_up=c.get("clear_up", True),
                on_line=starts_on_auton_line(c["x"], c["y"]),
                label=(f"cup+{nested} @({c['x']:.0f},{c['y']:.0f})" if nested
                       else f"cup @({c['x']:.0f},{c['y']:.0f})")))
            oid += 1
        self.obj_by_id = {o.oid: o for o in self.objects}

        self.goals: List[GoalInfo] = []
        for g in GR.GOALS:
            kind = ("center" if g["id"] == GR.CENTER_GOAL_ID else
                    "neutral" if g["alliance"] == "neutral" else "alliance")
            self.goals.append(GoalInfo(g["id"], g["x"], g["y"], g["alliance"], kind,
                                       goal_region(g["id"], g["x"], g["y"], GR.CENTER_GOAL_ID),
                                       g["label"]))
        self.goal_index = {g.gid: i for i, g in enumerate(self.goals)}

        self.toggles: List[ToggleInfo] = []
        for t in GR.TOGGLES:
            x, y = t["x"], t["y"]
            if x < 5:   nx, ny = 1, 0
            elif x > 139: nx, ny = -1, 0
            elif y < 5: nx, ny = 0, 1
            else:       nx, ny = 0, -1
            self.toggles.append(ToggleInfo(t["id"], x, y, t["quadrant"], nx, ny))

        # Loader / Load Zone centres for the robot to park in
        if alliance == "red":
            self.loaders = [(8.0, 13.0), (8.0, 131.0)]
        else:
            self.loaders = [(136.0, 13.0), (136.0, 131.0)]

    # ── initial dynamic state pieces ──────────────────────────────────────
    def initial_stacks(self) -> Tuple[Tuple, ...]:
        stacks = []
        for g in self.goals:
            colour = GR.PREPLACED_GOAL_PINS.get(g.gid) if not self.skills else None
            stacks.append((("pin",) + _pin_halves(colour),) if colour else ())
        return tuple(stacks)

    def initial_loads(self) -> Tuple[int, int, int, int]:
        """(cups, alliance pins, yellow pins, blue pins[skills])"""
        if self.skills:
            return (GR.SKILLS_LOADS_CUPS, GR.SKILLS_LOADS_RED_PINS,
                    GR.SKILLS_LOADS_YELLOW_PINS, GR.SKILLS_LOADS_BLUE_PINS)
        return (GR.MATCH_LOADS_CUPS, GR.MATCH_LOADS_ALLIANCE_PINS,
                GR.MATCH_LOADS_YELLOW_PINS, 0)

    # ── scoring ───────────────────────────────────────────────────────────
    def score(self, stacks, toggles, midfield_owner=None) -> Dict[str, int]:
        """Total goal points for both alliances (Skills: all under 'red')."""
        red = blue = 0
        for g, st in zip(self.goals, stacks):
            if not st:
                continue
            owner = None
            if g.region != "midfield":
                o = toggles[TOGGLE_OF_QUADRANT[g.region] - 1]
                owner = o if o in ("red", "blue") else None
            pts = score_stack(st, g.region, owner,
                              midfield_owner if g.region == "midfield" else None,
                              skills=self.skills)
            red += pts["red"]
            blue += pts["blue"]
        return {"red": red, "blue": blue}

    def awp_status(self, stacks, toggles, alliance: str):
        total = goals2 = 0
        for g, st in zip(self.goals, stacks):
            if not st:
                continue
            if g.region != "midfield" and QUADRANT_COLOR[g.region] != alliance:
                continue
            owner = None
            if g.region != "midfield":
                o = toggles[TOGGLE_OF_QUADRANT[g.region] - 1]
                owner = o if o in ("red", "blue") else None
            n = scored_halves(st, g.region, owner, alliance)
            total += n
            goals2 += n >= 2
        return total, goals2


# ═══════════════════════════════════════════════════════════════════════════
# Motion model (differential drive, can drive forward or backward)
# ═══════════════════════════════════════════════════════════════════════════
def ang_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def drive_time(dist: float, prof: RobotProfile) -> float:
    if dist <= 0.05:
        return 0.0
    v, a = prof.max_speed, prof.accel
    d_acc = v * v / a           # accelerate + brake distance
    if dist >= d_acc:
        return dist / v + v / a
    return 2.0 * math.sqrt(dist / a)


def turn_time(angle: float, prof: RobotProfile) -> float:
    if angle < math.radians(3):
        return 0.0
    return angle / math.radians(prof.turn_rate) + 0.08


PATH_FACTOR = 1.08      # straight-line distance → driven distance
SEGMENT_OVERHEAD = 0.12  # settle / alignment slop per move (s)


def move_to_reach(x, y, h, tx, ty, reach: float, use: str, prof: RobotProfile):
    """Drive so that a point `reach` inches ahead of (use='front') or behind
    (use='rear') the robot centre lands on (tx, ty), approaching in a straight
    line.  use='center' puts the centre at distance `reach` from the target.
    Returns (time, new_x, new_y, new_heading)."""
    dx, dy = tx - x, ty - y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return 0.0, x, y, h
    ux, uy = dx / d, dy / d
    travel = max(0.0, d - reach)
    move_dir = math.atan2(uy, ux)
    if use == "rear":
        need_h = move_dir + math.pi          # drive backwards into it
    elif use == "front":
        need_h = move_dir
    else:   # centre: whichever way round is quicker
        need_h = move_dir if ang_diff(h, move_dir) <= math.pi / 2 else move_dir + math.pi
    t = turn_time(ang_diff(h, need_h), prof)
    t += drive_time(travel * PATH_FACTOR, prof) + (SEGMENT_OVERHEAD if travel > 1 else 0.0)
    nx, ny = x + ux * travel, y + uy * travel
    return t, nx, ny, need_h
