"""
Rule-fidelity tests against the official V5RC Override manual (v2.0).

Run:  venv\\Scripts\\python.exe -m pytest tests -q
  or: venv\\Scripts\\python.exe tests\\test_rules.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.scoring import score_stack, half_visibility, scored_halves


# Figure SC5 stack, bottom → top (6 pins, 5 cups).  Reading the photo:
# pin1 yellow/red both visible, pin2 fully hidden, pin3 yellow/yellow both
# visible, pin4 top blue visible, pin5 top red visible, pin6 top red visible.
FIG_SC5 = [
    ("pin", "yellow", "red"),
    ("cup", False),          # lower half clear (shows pin1 top), upper opaque
    ("pin", "red", "blue"),  # hidden both
    ("cup", True),           # lower opaque, upper clear
    ("pin", "yellow", "yellow"),
    ("cup", False),
    ("pin", "red", "blue"),  # bottom hidden, top blue visible
    ("cup", False),
    ("pin", "blue", "red"),  # bottom hidden, top red visible
    ("cup", False),
    ("pin", "blue", "red"),  # bottom hidden, top red visible (open air)
]


def test_figure_sc5_toggle_yellow():
    assert score_stack(FIG_SC5, "left", None, None) == {"red": 15, "blue": 5}


def test_figure_sc5_toggle_blue():
    assert score_stack(FIG_SC5, "left", "blue", None) == {"red": 15, "blue": 35}


def test_figure_sc5_toggle_red():
    assert score_stack(FIG_SC5, "left", "red", None) == {"red": 45, "blue": 5}


def test_bottom_pin_shows_both_halves():
    vis = half_visibility([("pin", "yellow", "red")])
    assert vis[0] == (True, True)
    assert score_stack([("pin", "yellow", "red")], "left", "red", None) == {"red": 15, "blue": 0}


def test_each_cup_hides_exactly_one_half():
    for clear_up in (True, False):
        st = [("pin", "red", "red"), ("cup", clear_up), ("pin", "red", "red")]
        assert score_stack(st, "left", None, None)["red"] == 15   # 3 of 4 halves


def test_midfield_yellow_needs_majority():
    st = [("pin", "yellow", "yellow")]
    assert score_stack(st, "midfield", "red", None) == {"red": 0, "blue": 0}
    assert score_stack(st, "midfield", None, "blue") == {"red": 0, "blue": 20}


def test_skills_quadrant_colour_rules():
    # Red half in a BLUE quadrant does not score in Skills (RSC3a.i)
    st = [("pin", "yellow", "red")]
    assert score_stack(st, "top", "blue", None, skills=True)["red"] == 10   # yellow only
    # Toggle must match the quadrant colour (RSC3a.iii.1)
    assert score_stack(st, "top", "red", None, skills=True)["red"] == 0
    assert score_stack(st, "left", "red", None, skills=True)["red"] == 15
    # Blue half scores in a blue quadrant, credited to the single skills score
    assert score_stack([("pin", "blue", "blue")], "right", None, None, skills=True)["red"] == 10
    # Midfield: red and blue both score
    assert score_stack([("pin", "red", "blue")], "midfield", None, None, skills=True)["red"] == 10


def test_awp_halves_exclude_midfield_yellow():
    st = [("pin", "yellow", "red")]
    assert scored_halves(st, "midfield", None, "red") == 1
    assert scored_halves(st, "left", "red", "red") == 2


# ── simulator-level checks ─────────────────────────────────────────────────
def _sim(**kw):
    from simulation.rules_engine import RulesEngine
    RulesEngine.VERBOSE = False
    from simulation.simulator import OverrideSimulator
    return OverrideSimulator(headless=True, **kw)


def test_start_positions_legal():
    sim = _sim()
    side = {"red": lambda x, y: y > x, "blue": lambda x, y: y < x}
    quads = set()
    from simulation.scoring import quadrant_of, QUADRANT_COLOR
    for r in sim.robots:
        x, y = r.body.position
        assert side[r.alliance](x, y), r.robot_id
        q = quadrant_of(x, y)
        assert QUADRANT_COLOR[q] == r.alliance, (r.robot_id, q)
        quads.add(q)
    assert len(quads) == 4          # SG1d: no two robots share a quadrant


def test_preload_is_one_pin_no_cup():
    sim = _sim()
    for r in sim.robots:
        assert r.carrying_pin is not None and r.carrying_cup is None
    assert sim.red_alliance_pins_left == 10 and sim.red_cups_left == 10


def test_preplaced_neutral_pins_and_toggle_value():
    sim = _sim()
    sim.step(1 / 60, [None] * 4)
    assert sim.red_score == 0 and sim.blue_score == 0
    sim.toggles[0].owner = "red"     # left toggle → left quadrant SW goal
    sim.step(1 / 60, [None] * 4)
    assert sim.red_score == 20


def test_physics_keeps_up_with_clock():
    sim = _sim(skills_mode=True)
    r = sim.robots[0]
    x0 = r.body.position.x
    for _ in range(20):                       # 1.0 s at CONTROL_DT = 0.05
        sim.step(0.05, [{"left": 1, "right": 1}])
    assert r.body.position.x - x0 > 35        # ~60 in/s robot, not 1/3 speed


def test_auton_line_blocks_crossing_but_allows_midfield():
    sim = _sim()
    sim.timer_started = True
    red1 = sim.robots[0]
    # drive red1 at the blue side: aim up-right across y = x
    red1.body.position = (40, 60)
    red1.body.angle = -math.pi / 2           # facing -y (toward y < x)
    for _ in range(40):
        sim.step(0.05, [{"left": 1, "right": 1}, None, None, None])
    x, y = red1.body.position
    assert y >= x - 12, (x, y)               # body centre stays on red side
    # Midfield is reachable during auton
    assert sim.auton_side_ok("red", 72, 60) and not sim.auton_side_ok("red", 100, 40)


def test_auton_tie_bonus():
    sim = _sim()
    sim.timer_started = True
    while sim.match_phase == "autonomous":
        sim.step(0.05, [None] * 4)
    assert sim.rules_engine.auton_bonus_red == 6 and sim.rules_engine.auton_bonus_blue == 6


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:      # noqa
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
