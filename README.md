# VEX Override Simulator

A high-fidelity 2D physics simulator for the **2026-2027 VEX V5 Robotics Competition game: Override**.

Built with **Pygame + Pymunk** for realistic rigid-body physics, accurate scoring, and **PyTorch** for strategy training.

<img width="420" height="420" alt="simVid(compressed)" src="https://github.com/user-attachments/assets/714f8448-763d-4070-8269-308d5dd230e2" />

---

## Features

- **Full Physics Simulation** — Rigid-body dynamics, velocity limiting, realistic friction/elasticity, and collision handling.
- **Accurate Scoring** — Per-half pin scoring, clear/dark cup orientation, toggle-controlled yellow ownership, and full stack visibility rules.
- **Autonomous Period** — 15-second auton with white-tape wall enforcement (each robot confined to its starting wedge), correct +12 autonomous bonus awarded to the winner, and all four robots actively competing in their own zone.
- **Endgame System** — Complete implementation of the last 20 seconds:
  - +8 points per robot parked in the Midfield (live bonus)
  - Center goal locks after Pin + Cup + Pin stack (3 items)
  - SC5b yellow ownership applied at match end based on Midfield robot majority
- **Match Loading** — Realistic inventory system with 5 load combinations. Full cup+pin pair is correctly picked up in a single trip.
- **Heuristic Bot** — Deterministic rule-based AI agent (`agents/heuristic_bot.py`) that plays near-optimally:
  - Partner coordination (shared target registry)
  - Obstacle avoidance and stuck-escape
  - Toggle takeover, post-score toggle claims
  - Chain stacking, cup orientation pre-flip, carry-patience force-score
  - Match-load cycling as primary element source during driver control
- **Interactive Controls** — Full keyboard support for driving, intaking, scoring, flipping, toggling, and match loading.
- **Live HUD** — Real-time scores, timer, Midfield parking bonus, and matchload inventory tracking.
- **Video Recording** — Matches can be recorded to MP4 (720×720 @ 30 fps).

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/vmakarov28/override_sim.git
cd override_sim

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py --mode interactive        # keyboard-controlled match
python run_with_bots.py                  # watch 4 heuristic bots play
python run_with_bots.py --matches 5      # run 5 matches and print stats
python run_with_bots.py --video out.mp4  # record to video
```

---

## Run Modes

| Command | Description |
|---|---|
| `python main.py --mode interactive` | Drive robots with keyboard |
| `python main.py --mode render` | Watch a trained RL policy |
| `python main.py --mode train-vis` | Train RL policy with visualisation |
| `python run_with_bots.py` | 4 heuristic bots, single match |
| `python run_with_bots.py --matches N` | Run N matches, print score summary |
| `python run_with_bots.py --video path.mp4` | Record match to video |
| `python run_with_bots.py --headless` | Headless (no window) |

---

## Controls (Interactive Mode)

| Action | Red Alliance | Blue Alliance |
|---|---|---|
| Drive | W / A / S / D | ↑ / ← / ↓ / → |
| Intake | **E** | **Right Shift** |
| Score Pin | **Q** | **Right Ctrl** |
| Score Cup | **Shift + Q** | **Shift + Right Ctrl** |
| Flip Pin | **F** | **[** |
| Flip Cup | **Shift + F** | **Shift + [** |
| Toggle (Roller) | **T** | — |
| Match Load | **M + 1–5** | **M + 1–5** |
| Reset Match | **R** | **R** |

---

## Match Loading

Drive inside your alliance's **Match Loading Zone** (red/blue taped rectangles in the corners) then press the key combination.

| Key | Spawns |
|---|---|
| **M + 1** | Loaded Cup + Alliance/Yellow Pin (cup with nested pin — one intake grabs both) |
| **M + 2** | Loaded Cup + Yellow/Yellow Pin |
| **M + 3** | Individual Cup only |
| **M + 4** | Individual Alliance Pin only |
| **M + 5** | Individual Yellow/Yellow Pin only |

Only red1 / blue1 can match-load. Match loading is disabled during the Autonomous period.

---

## Autonomous Period

- Lasts **15 seconds** at the start of each match.
- The field is divided into **four wedge-shaped zones** by white tape diagonal lines. Each robot starts in its own wedge.
- **Robots may not cross any white tape line** — the simulator enforces this with physical walls along all 8 boundary segments (4 outer diagonals + 4 midfield diamond edges).
- All four robots actively compete during auton: scoring goals, picking up elements, and claiming toggles within their wedge.
- At the end of auton, the alliance with the higher score earns a **+12 autonomous bonus** (tie = no bonus). The bonus persists through the rest of the match.

---

## Endgame (Last 20 Seconds)

- **+8 points per robot** parked in the Midfield diamond (live, updates every frame).
- **Center Goal** locks after reaching Pin + Cup + Pin (3 items).
- Yellow pin ownership in the center goal is decided by **Midfield robot majority** at match end (SC5b rule).

---

## Scoring Summary

| Item | Points |
|---|---|
| Alliance pin in goal (per visible half) | 5 |
| Yellow pin half (toggle-controlled) | 10 |
| Cup alone | 0 |
| Stack bonus (per extra nested pin) | 3 |
| Robot parked in Midfield (endgame) | +8 each |
| Autonomous bonus (winner only) | +12 |

---

## Project Structure

```
override_sim/
├── agents/
│   ├── heuristic_bot.py      # Rule-based deterministic AI agent
│   └── bot_event_log.py      # Structured match event logger
├── config/
│   ├── game_rules.py         # Field layout, scoring constants, robot starts
│   └── hyperparameters.py    # RL training configuration
├── environment/
│   └── override_env.py       # Gym-compatible RL environment wrapper
├── evaluation/
│   ├── evaluate.py           # Match evaluation harness
│   └── video_recorder.py     # MP4 recording via OpenCV
├── simulation/
│   ├── simulator.py          # Core match loop, physics, auton walls
│   ├── rules_engine.py       # Live scoring, foul detection, SC5b
│   ├── robot.py              # Robot physics, intake, scoring
│   └── game_objects.py       # Pins, cups, goals, toggles
├── training/
│   ├── env_wrapper.py        # Multi-agent RL wrapper with reward shaping
│   ├── network.py            # MAPPO actor-critic network
│   └── ...
├── run_with_bots.py          # Heuristic bot match runner
└── main.py                   # Entry point (interactive / render / train)
```

---

## Requirements

- Python 3.10+
- pygame
- pymunk
- numpy
- torch (for RL training only)
- opencv-python (for video recording only)

See `requirements.txt` for pinned versions.
