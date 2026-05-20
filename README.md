# VEX Override Simulator

A high-fidelity 2D physics simulator for the **2026-2027 VEX V5 Robotics Competition game: Override**.

Built with **Pygame + Pymunk** for realistic rigid-body physics, accurate scoring, and **Pytorch** for strategy training.

---

## Features

- **Full Physics Simulation** — Rounded convex polygons, velocity limiting, realistic friction/elasticity, and proper collision handling.
- **Accurate Scoring** — Per-half pin scoring, clear/dark cup orientation, toggle-controlled yellow ownership, and proper stack visibility rules.
- **Endgame System** — Complete implementation of the last 20 seconds:
  - 18-inch height limit in the Midfield
  - Live +8 parking bonus per robot in the Midfield
  - Center goal locks after Pin + Cup + Pin stack (3 items)
  - SC5b yellow ownership applied at match end based on Midfield robot majority
- **Match Loading** — Realistic inventory system with 5 different load combinations.
- **Interactive Controls** — Full keyboard support for driving, intaking, scoring, flipping, toggling, and match loading.
- **Live HUD** — Real-time scores, timer, Midfield parking bonus, and matchload inventory tracking.

---

## Controls

| Action              | Red Alliance (WASD)      | Blue Alliance (Arrows)      |
|---------------------|--------------------------|-----------------------------|
| Drive               | W / A / S / D            | ↑ / ← / ↓ / →               |
| Intake              | **E**                    | **Right Shift**             |
| Score Pin           | **Q**                    | **Right Ctrl**              |
| Score Cup           | **Shift + Q**            | **Shift + Right Ctrl**      |
| Flip Pin            | **F**                    | **[**                         |
| Flip Cup            | **Shift + F**            | **Shift + [**                 |
| Toggle (Roller)     | **T**                    | —                           |
| Match Load          | **M + 1/2/3/4/5**        | **M + 1/2/3/4/5**           |
| Reset Match         | **R**                    | **R**                       |

---

## Match Loading System

Drive inside your alliance’s **Match Loading Zone** and touch the wall (red or blue taped rectangles in the corners).

### Keybinds

| Keybind   | What Spawns                                     |
|-----------|-------------------------------------------------|
| **M + 1** | Nested Cup + **Alliance/Yellow Pin**            | 
| **M + 2** | Nested Cup + **Yellow/Yellow Pin**              |
| **M + 3** | **Individual Cup**                              |
| **M + 4** | **Individual Alliance/Yellow Pin**              | 
| **M + 5** | **Individual Yellow/Yellow Pin**                |

---

## Endgame (Last 20 Seconds)

- **18-inch height limit** while in the Midfield so scoring cannot exceed goal/pin/cup/pin height.
- **+8 points per robot** parked in the Midfield (live bonus)
- **Center Goal** locks after reaching Pin + Cup + Pin (3 items)
- Yellow pin ownership in center goal is decided by **Midfield robot majority** at match end (SC5b)

---

## How to Run

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt

# Run the simulator
python main.py --mode interactive


cd C:\Users\aipla\Desktop\override_sim
venv\Scripts\activate
python main.py --mode interactive


del artifacts\models\*.pt
python main.py --mode train-vis
