# Rules & simulator audit (Nov 2026)

Checked the simulator against the **official V5RC Override game manual v2.0
(Sept 3, 2026)** — `content.vexrobotics.com/docs/2026-2027/override/files/override-2.0.pdf`.
Everything below was wrong before this audit and is now fixed.  Regression
tests: `venv\Scripts\python.exe tests\test_rules.py` (includes the manual's own
worked example, Figure SC5: 15/5, 15/35, 45/5).

## Scoring (biggest impact on which strategies look good)

| # | Was | Manual | Fix |
|---|-----|--------|-----|
| 1 | Bottom pin's lower half hidden by the goal post | Both halves of a pin sitting on a goal are visible (Fig. SC5: bottom yellow half = +10) | `simulation/scoring.py` |
| 2 | +3 "stack bonus" per nested pin | No stack bonus exists | `POINTS_STACK_BONUS = 0` |
| 3 | +8/robot in Midfield added live for the last 20 s | +8/robot **ending** the match in the Midfield (SC6) | counted once at match end |
| 4 | Endgame = last 20 s; center goal "locks" at Pin+Cup+Pin | Endgame = last 10 s; **no** placing on the Midfield goal during Endgame (SG12 v2.0) | `ENDGAME_SECONDS = 10` |
| 5 | Auton tie → nobody gets the bonus | Tie (incl. 0-0) → 6 pts each (SC7b) | `POINTS_AUTONOMOUS_TIE` |
| 6 | No Autonomous Win Point | SC8: ≥6 scored pins on own side, ≥2 goals with ≥2, no perimeter contact | `RulesEngine.autonomous_win_point` |
| 7 | Skills: every half on the field counted, any toggle colour worked | RSC3: red halves only in red quadrants/Midfield, blue only in blue quadrants/Midfield, toggle must match its quadrant's colour; Midfield yellows owned if robot ends there | `score_stack(..., skills=True)` |

## Field setup

| # | Was | Manual | Fix |
|---|-----|--------|-----|
| 8 | red2 started at the TOP wall, blue1 at the BOTTOM wall — inside the opponent's quadrants | SG1: each robot in its own quadrant on its alliance's side (red = left + bottom, blue = top + right) | `ROBOT_STARTS` |
| 9 | Neutral goals started empty | Each of the 5 neutral goals starts with a yellow/yellow pin Placed on it (Fig. FO-2). Setting a toggle is instantly worth 20 | `PREPLACED_GOAL_PINS` |
| 10 | Preload = pin + cup, taken out of the match loads | Preload = **one pin** (SG5), separate from the match loads | `_preload_robots` |
| 11 | Match loads 12 alliance pins (−preloads) | 10 cups, 10 alliance pins, 1 yellow pin per alliance; Skills: 3 red/yellow, 4 blue/yellow, 7 cups (RSC4b) | config + `_reset_inventory` |
| 12 | Only red1/blue1 could match load | Either robot can use its alliance's loaders | `_try_bot_match_load` |
| 13 | Top/bottom toggles labelled with the wrong quadrant colour | Top = blue quadrant, bottom = red quadrant | `TOGGLES` |
| 14 | All cups clear-side-up | 24 wall cups gray-side-up, 12 interior cups clear-side-up | `clear_up` per cup |
| 15 | Cups 4" wide / pins 6" long | Cup 3.15" × 6.5", pin 1.6" × 6.5" | `game_objects.py` |

## Autonomous period

| # | Was | Manual | Fix |
|---|-----|--------|-----|
| 16 | Each robot sealed in its own wedge; Midfield walled off | SG7/SG8: one Autonomous Line (the y = x double tape + Midfield square). Each alliance may use its whole half, the line, the Midfield, and the 28 objects that start on the line | per-alliance walls + `on_auton_line` |
| 17 | Intake could reach objects across the line | Can't touch objects on the opponent's side | `Robot.touch_allowed` |

## Physics / timing

| # | Was | Effect | Fix |
|---|-----|--------|-----|
| 18 | `MAX_SUBSTEPS = 4` (1/60 s) per step, but bots/RL step at 1/20 s | Robots only got **1/3 of the match time** to move — every bot and RL run was in slow motion | `MAX_SUBSTEPS = 32` |
| 19 | Drive force applied on the first substep only | Robot speed depended on step size | force applied every substep |
| 20 | Intake cooldown counted down 1/60 s per call | Mechanism timers 3× too long for bots | uses real `dt` |
| 21 | Any touch (impulse ≥ 0.01) tipped standing cups and threw their pin out | Brushing a wall cup ruined the loaded cup beside it | tip at impulse ≥ 3; pin only ejected on a hard hit (≥ 40) |
| 22 | Toggles flipped instantly | Real toggles need a push | `TOGGLE_SET_TIME = 0.6 s` |

## Robot model

`config/robot_profile.py` now describes the robot, and both the simulator and
the planner use it.  `USER_ROBOT` = front intake for any element, back intake
for standing pins only, **scores only out of the back**.  The old behaviour
(score from the robot centre) is `LEGACY_ROBOT`; the HeuristicBot runners and
RL env use it because they were written for it.

## What was stopping strategy discovery

* Wrong point values (1–7) made the bot/RL chase the wrong things (stack
  bonus, live parking for 20 s, under-valued first pins and toggles).
* The 3× slow-motion physics (18) made every timing-based trade-off wrong.
* Field-setup errors (8–11, 16) changed what auton could even reach.
* The HeuristicBot is a greedy priority tree: it can't plan ahead, so it can't
  find routes.  The new planner (`strategy/`) searches whole action sequences
  and verifies each route in physics.
* The RL reward (`training/env_wrapper.py`) is ~60 hand-tuned shaping terms
  built on the old rules; it has had exploits before (PROBLEM 69/70).  The
  bottom-half rule copy there is fixed, but the reward should be rebuilt
  around the score delta from the corrected rules engine before retraining.
