# VEX Override Simulator

A high-fidelity 2D physics simulator for the **2026-2027 VEX V5 Robotics Competition game: Override**.

Built with **Pygame + Pymunk** for realistic rigid-body physics, accurate scoring, and **PyTorch** for strategy training.

<img width="420" height="420" alt="simVid(compressed)" src="https://github.com/user-attachments/assets/714f8448-763d-4070-8269-308d5dd230e2" />

---

## Features

- **Full Physics Simulation** — Rigid-body dynamics, velocity limiting, realistic friction/elasticity, and collision handling.
- **Accurate Scoring** — Per-half pin scoring, clear/dark cup orientation, toggle-controlled yellow ownership, stack bonuses, and full visibility rules.
- **Autonomous Period** — 15-second auton with white-tape wall enforcement (each robot confined to its starting wedge), correct +12 autonomous bonus awarded to the winner, and all four robots actively competing in their own zone.
- **Endgame System** — Complete implementation of the last 20 seconds:
  - +8 points per robot parked in the Midfield (live bonus)
  - Center goal locks after Pin + Cup + Pin stack (3 items)
  - SC5b yellow ownership applied at match end based on Midfield robot majority
- **Match Loading** — Realistic inventory system with 5 load combinations. Full cup+pin pair is correctly picked up in a single trip.
- **VEX Robot Skills Mode** *(new in v1.1)* — Single-robot solo challenge per the official VEX Skills format:
  - One robot, 60-second timer, no opposing alliance
  - Every visible pin half (red, blue, or owned-yellow) counts toward a single score
  - Score screen at match end instead of a winner banner
  - HeuristicBot scores **82 points** on the deterministic skills field
- **Improved Heuristic Bot** *(v1.1)* — Deterministic rule-based AI agent (`agents/heuristic_bot.py`):
  - **Strong goal commit-lock** — once a target is chosen, the bot stays locked on instead of oscillating between near-equal options
  - **1-step lookahead** — prefers goals that set up a fast follow-up (near a loading zone or a partial-stack goal)
  - **Skills-mode awareness** — recognizes that red/blue alliance distinctions don't apply when there's no opponent, so it scores on every available half
  - **Stricter pre-flip threshold** — only flips en route when the orientation gain justifies the 2-second cooldown
  - **Longer pickup timeout** — persists on a chosen pin/cup for 5 seconds before blacklisting (was 2.5s), eliminating target ping-pong
  - Partner coordination, obstacle avoidance, stuck-escape, toggle takeover, post-score toggle claims, chain stacking, match-load cycling
- **Interactive Controls** — Full keyboard support for driving, intaking, scoring, flipping, toggling, and match loading.
- **Live HUD** — Real-time scores, timer, Midfield parking bonus, matchload inventory tracking, and a dedicated Skills score panel.
- **Video Recording** — Matches can be recorded to MP4 (720×720 @ 30 fps).
- **Headless Benchmark Harness** — `bench.py` for fast bot-vs-bot evaluation; `run_bot_skills.py` for skills-mode runs.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/vmakarov28/vex-override-simulator.git
cd vex-override-simulator

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py --mode interactive        # keyboard-controlled 2v2 match
python main.py --mode skills             # VEX Robot Skills (solo 60s)
python run_with_bots.py                  # watch 4 heuristic bots play
python run_bot_skills.py                 # watch the bot run skills
```

---

## Run Modes

### Interactive & Skills (`main.py`)

| Command | Description |
|---|---|
| `python main.py --mode interactive` | Keyboard-controlled 2v2 match (default) |
| `python main.py --mode skills` | VEX Robot Skills — solo 60-second run, single score counter |
| `python main.py --mode train` | Start/resume MAPPO self-play training |
| `python main.py --mode train-vis` | Training with periodic live visualization |
| `python main.py --mode eval --checkpoint PATH` | Evaluate a trained checkpoint |

### Heuristic Bot Match (`run_with_bots.py`)

| Command | Description |
|---|---|
| `python run_with_bots.py` | 4 heuristic bots, single 2v2 match |
| `python run_with_bots.py --matches 5` | Run 5 matches, print score summary |
| `python run_with_bots.py --bots red1,red2` | Only red bots use the heuristic |
| `python run_with_bots.py --video path.mp4` | Record match to MP4 |
| `python run_with_bots.py --no-reasons-log` | Skip per-step reasoning log |

### Heuristic Bot Skills (`run_bot_skills.py`)

| Command | Description |
|---|---|
| `python run_bot_skills.py` | Watch the bot run skills (with video) |
| `python run_bot_skills.py --matches 5` | Run 5 back-to-back skills attempts |
| `python run_bot_skills.py --seed 1234` | Specific seed (bot is deterministic — only field jitter changes) |
| `python run_bot_skills.py --headless` | No window — fast benchmark with mean/std/min/max |
| `python run_bot_skills.py --headless --matches 20` | Twenty-run skills benchmark |
| `python run_bot_skills.py --auto-close` | Close immediately after the run (useful for video-only saves) |
| `python run_bot_skills.py --video ''` | Disable video recording |

### Benchmarks (`bench.py`)

| Command | Description |
|---|---|
| `python bench.py` | 20 matches, 4-bot self-play, print mean/std/min/max |
| `python bench.py --matches 50 --quiet` | 50 matches, summary only |
| `python bench.py --out results.json` | Save per-seed results for paired comparison |
| `python bench.py --compare baseline.json` | Paired t-test vs a prior baseline (great for iterating on the bot) |

---

## Controls (Interactive 2v2)

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
| Quit | **ESC** | **ESC** |

In **Skills mode**, only the Red controls are active (one robot), and only the red side's match-load inventory is shown.

---

## Match Loading

Drive inside your alliance's **Match Loading Zone** (red/blue taped rectangles in the corners) then press the key combination.

| Key | Spawns |
|---|---|
| **M + 1** | Loaded Cup + Alliance Pin (cup with nested pin — one intake grabs both) |
| **M + 2** | Loaded Cup + Yellow Pin |
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
- **Skills mode has no autonomous period** — the 60-second run is all driver-control.

---

## Endgame (Last 20 Seconds of Driver / Skills)

- **+8 points per robot** parked in the Midfield diamond (live, updates every frame).
- **Center Goal** locks after reaching Pin + Cup + Pin (3 items).
- Yellow pin ownership in the center goal is decided by **Midfield robot majority** at match end (SC5b rule).
- In Skills mode, the player's lone robot trivially holds the midfield majority, so any yellow halves visible in the center goal at match end score +10 each.

---

## Scoring Summary

| Item | Points |
|---|---|
| Alliance pin in goal (per visible half) | 5 |
| Yellow pin half (toggle-controlled, or skills-mode in center) | 10 |
| Cup alone | 0 |
| Stack bonus (per extra nested pin, requires cup between pins) | 3 |
| Robot parked in Midfield (endgame, per robot per second) | +8 |
| Autonomous bonus (winning alliance only) | +12 |

In **Skills mode**, the bot's `RulesEngine` folds blue-side scoring into the red total (no opposing alliance), so every visible half on the entire field contributes to the player's single score.

---

## Heuristic Bot Performance (v1.1)

Deterministic skills-mode benchmark (seed 2000, 60-second run):

| Version | Skills Score |
|---|---|
| v1.0 (initial release) | 59 |
| **v1.1** (current) | **82** |

Match-mode (4-bot self-play, 20-seed bench):

| Version | Mean Total |
|---|---|
| v1.0 | ~135 |
| **v1.1** | **162** |

Reproduce:

```bash
python run_bot_skills.py --headless --matches 10        # ~82 skills
python bench.py --matches 20 --quiet                    # ~162 total match
```

---

## Project Structure

```
override_sim/
├── agents/
│   ├── heuristic_bot.py      # Rule-based deterministic AI agent
│   └── bot_event_log.py      # Structured match event logger
├── config/
│   ├── game_rules.py         # Field layout, scoring constants, SKILLS_SECONDS
│   └── hyperparameters.py    # RL training configuration
├── evaluation/
│   ├── evaluate.py           # Match evaluation harness
│   └── video_recorder.py     # MP4 recording via OpenCV
├── simulation/
│   ├── simulator.py          # Core match loop, physics, skills-mode toggle
│   ├── rules_engine.py       # Live scoring, foul detection, SC5b, skills folding
│   ├── robot.py              # Robot physics, intake, scoring
│   └── game_objects.py       # Pins, cups, goals, toggles
├── training/
│   ├── env_wrapper.py        # Multi-agent RL wrapper with reward shaping
│   ├── network.py            # MAPPO actor-critic network
│   └── ...
├── bench.py                  # Fast headless 4-bot benchmark
├── run_bot_skills.py         # Single-bot skills runner
├── run_with_bots.py          # Heuristic bot 2v2 match runner
└── main.py                   # Entry point (interactive / skills / train / eval)
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

---

## What's New in v1.1

- **VEX Robot Skills mode** — official solo-challenge format with single-score HUD and score-screen end overlay
- **HeuristicBot v15** — +23 points on skills (59 → 82) and ~+27 on match totals via:
  - Strong goal commit-lock (eliminates "swinging" indecision)
  - Pickup-timeout 50→100 steps (no more pin-target ping-pong)
  - Skills-mode-aware value functions (the bot now correctly values blue pins/goals as friendly when there's no opponent)
  - Pre-flip threshold tightened (no wasted cooldowns on tiny gains)
  - 1-step lookahead chain bonus
- **`run_bot_skills.py`** — dedicated skills runner with `--video`, `--auto-close`, headless mode, and a mean/std/min/max summary
- **Skills-aware fixes** — `_valid_goal_for_me`, `_half_pts_for_me`, `_pin_value`, `_live_yellow_multiplier` all branch on `sim.skills_mode`
- **Video recorder unicode fix** — Windows console no longer crashes on the post-save print
