"""
config/robot_profile.py
───────────────────────
Physical / mechanism description of a robot.  Both the physics simulator
(simulation/robot.py) and the route planner (strategy/) read this, so a
route the planner finds is one the simulated robot can actually drive.

USER_ROBOT models the team's real robot as described:
  * FRONT intake: picks up any loose Pin (lying or standing) or Cup,
    including a Cup with a Pin nested in it (one motion grabs both).
  * BACK intake: can only pick up VERTICAL (standing) Pins.  A Pin standing
    up in a Cup counts as vertical — the back intake pulls the Pin out and
    leaves the Cup on the floor (set rear_pulls_nested_pin=False if your
    back mechanism cannot do that).
  * SCORING: only out of the BACK — the robot reverses onto a Goal.

Numbers you should measure on the real robot and edit here:
  max_speed, accel, turn_rate, and the mechanism times.  Everything
  downstream (auton routes, skills routes) re-optimises automatically.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class RobotProfile:
    name: str = "user_robot"

    # Footprint (inches).  Length is front→back.
    width: float = 15.0
    length: float = 15.0

    # Drivetrain (measured from the physics sim at real-time step:
    # ~60 in/s top speed, ~190 deg/s turning).  Real V5 450-rpm 3.25" drives
    # are typically 55–70 in/s.
    max_speed: float = 60.0         # in/s
    accel: float = 90.0             # in/s^2 (planner's trapezoid profile)
    turn_rate: float = 210.0        # deg/s

    # Intake capabilities
    front_intake_pins: bool = True
    front_intake_cups: bool = True
    rear_intake_vertical_pins: bool = True
    rear_pulls_nested_pin: bool = True

    # Scoring side: "rear" (user robot) or "front"
    score_side: str = "rear"

    # Orientation control.  With can_flip_* the robot can choose which pin
    # half / cup half goes down (costs flip_time).  Set False if your robot
    # cannot reorient elements — the planner then keeps pickup orientation.
    can_flip_pin: bool = True
    can_flip_cup: bool = True
    flip_time: float = 0.4          # s

    # Mechanism times (s) — time the robot is stopped doing the action
    intake_time: float = 0.35       # grab one element
    score_pin_time: float = 0.45    # place a pin out the back
    score_cup_time: float = 0.45    # place a cup out the back
    score_pair_time: float = 0.6    # place a held cup + pin in one sequence
    toggle_time: float = 0.6        # rotate a toggle to a new set state
    match_load_time: float = 1.0    # human drops element, robot intakes it

    # Tallest point the robot can Place onto (inches above the floor).
    # Manual max robot height is 50".  Goals: alliance 3.25", neutral 5.8",
    # center 8.7", and each Placed Pin adds ~6.5".
    max_place_height: float = 50.0

    # Max pins/cups held (SG6)
    max_pins: int = 1
    max_cups: int = 1

    def to_dict(self):
        return asdict(self)


USER_ROBOT = RobotProfile()

# The robot the original sim modelled (front intake + center-point scoring)
LEGACY_ROBOT = RobotProfile(name="legacy", width=17.0, length=15.0,
                            score_side="center", rear_intake_vertical_pins=False)

DEFAULT_PROFILE = USER_ROBOT
