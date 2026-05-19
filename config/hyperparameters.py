"""
config/hyperparameters.py  (v5)
=================================
REWARD REDESIGN v5
------------------
Cumulative fixes (v3 problems preserved for history; v5 adds new fixes):

  PROBLEM 1 (v3): Approach reward saturated at destination.
    FIX: Continuous proximity reward fires every step based on dist to
    nearest valid goal, keeping the robot at the goal.

  PROBLEM 2 (v3): No incentive to actually press the score button.
    FIX: score_attempt_in_zone reward; now gated on stack-legal attempts only.

  PROBLEM 3 (v3): Robots park while holding objects.
    FIX: holding_timeout_penalty ramps up after HOLDING_TIMEOUT_STEPS.

  PROBLEM 4 (v3): Idle penalty too weak.
    FIX: -0.05/step idle + start_zone_penalty.

  PROBLEM 5 (v3→v5): Flip spam and no cost.
    FIX: flip_penalty = -0.15 per flip.  Small enough not to suppress
    intentional flips; large enough that random flip spam is unprofitable.

  PROBLEM 6 (v3): Pinning exploit.
    FIX: PINNING_STEPS_LIMIT contact tracking + per-step penalty.

  PROBLEM 7 (v3): Carrying robot approaches opponent goals.
    FIX: _carrying_target_dist() uses valid goals only.

  PROBLEM 8 (v5): Cup denial rewards were inverted.
    FIX: eff_clear=True means dark bottom faces pin → pin UP is BLOCKED
    (successful denial).  The if/else branches in _compute_rewards were
    swapped, rewarding failure and punishing success.  Now corrected.

  PROBLEM 9 (v5): Pin causal reward evaluated invisible halves.
    FIX: DOWN half of the bottom-most pin (index 0) is always hidden by
    the goal post and never scores; it is now excluded from the reward
    signal.  Deeper pins check actual cup-below visibility before awarding.

  PROBLEM 10 (v5): Yellow pin reward was asymmetric / non-causal.
    FIX: Yellow reward is now assigned to whichever alliance owns the
    toggle for that goal, regardless of who placed the pin.  The placing
    alliance receives score_opp_half if the toggle is owned by the enemy.

  PROBLEM 11 (v5): Proximity reward fired regardless of scoring legality.
    FIX: carrying_proximity_scale only fires when the robot's held element
    can make a legal score at a reachable goal (correct stack state).

  PROBLEM 12 (v5): Robots crowded at goals with the wrong element.
    FIX: fetch_needed_scale redirects robots toward the missing element
    type (pin or cup) when they cannot score anywhere.  wrong_element_loiter
    penalty actively discourages camping within scoring radius of an
    unreachable goal.

  PROBLEM 13 (v5): score_attempt_in_zone fired on illegal attempts.
    FIX: Reward now checks can_score_pin / can_score_cup at the attempted
    goal before awarding, so button-mashing on an empty goal earns nothing.

  PROBLEM 14 (v5): Observation lacked top-pin UP color per goal.
    FIX: Each goal slot in the 524-dim observation gains 3 bits (one-hot
    for red/blue/yellow) encoding the UP-half color of the goal's top pin.
    OBS_DIM updated from 524 → 551.
"""

# -------------------------------------------------------------------------
# OBSERVATION / ACTION SPACE
# -------------------------------------------------------------------------
OBS_DIM     = 551   # 524 base + 3 bits × 9 goals for top-pin UP color
ACTION_CONT = 2
ACTION_DISC = 7

# -------------------------------------------------------------------------
# ENVIRONMENT
# -------------------------------------------------------------------------
CONTROL_HZ        = 20
CONTROL_DT        = 1.0 / CONTROL_HZ
MAX_EPISODE_STEPS = int((15 + 105) * CONTROL_HZ)   # 2400 steps = full match

# -------------------------------------------------------------------------
# ACTION COOLDOWNS  (in control steps; x CONTROL_DT = seconds)
# -------------------------------------------------------------------------
COOLDOWN_TOGGLE     = 100   # 5 s
COOLDOWN_FLIP       =  40   # 2 s  (flip is still allowed, just costly)
COOLDOWN_SCORE      =  10   # 0.5 s
COOLDOWN_MATCH_LOAD =  30   # 1.5 s
COOLDOWN_INTAKE     =   5   # 0.25 s debounce

# -------------------------------------------------------------------------
# MAPPO CORE (Anti-Collapse Settings)
# -------------------------------------------------------------------------
LEARNING_RATE        = 2.5e-4        # Slower learning = more stable
CRITIC_LR            = 8e-4
GAMMA                = 0.99
GAE_LAMBDA           = 0.95
CLIP_EPS             = 0.25          # More conservative updates
VALUE_LOSS_COEF      = 0.5
ENTROPY_COEF             = 0.025       # Higher initial entropy
ENTROPY_COEF_MIN         = 0.005
ENTROPY_ANNEAL_STEPS     = 12_000_000  # Keep exploration longer
MAX_GRAD_NORM            = 0.5

# -------------------------------------------------------------------------
# ROLLOUT / BATCH
# -------------------------------------------------------------------------
NUM_PARALLEL_ENVS = 32
ROLLOUT_STEPS     = 512
MINIBATCH_SIZE    = 256
PPO_EPOCHS        = 10
TOTAL_ENV_STEPS   = 20_000_000

# -------------------------------------------------------------------------
# NETWORK ARCHITECTURE
# -------------------------------------------------------------------------
ACTOR_HIDDEN  = [512, 256, 128]
CRITIC_HIDDEN = [512, 256, 128]
LOG_STD_MIN   = -4.0
LOG_STD_MAX   = 0.0

# -------------------------------------------------------------------------
# SELF-PLAY / POLICY POOL (Anti-Collapse Settings)
# -------------------------------------------------------------------------
POOL_SIZE              = 16          # Increased for better diversity
POOL_SAMPLE_PROB       = 0.90        # Sample historical opponents 90% of time
CHECKPOINT_EVERY       = 500
SELF_PLAY_OPPONENT_MIX = 0.65        # 65% pool + 35% latest policy

# -------------------------------------------------------------------------
# REWARD WEIGHTS  (v3)
#
# Hierarchy:
#   1. Terminal win/loss          -> game outcome matters most
#   2. Score delta                -> dominant step-level signal
#   3. Causal scoring events      -> large one-time bonuses
#   4. Carrying proximity (dense) -> constant pull toward goal while carrying
#   5. Score attempt reward       -> explicit reinforcement for pressing score
#   6. Empty-hand approach delta  -> keeps robots moving toward objects
#   7. Holding timeout penalty    -> forces robots to score, not park
#   8. Flip penalty               -> cuts flip spam
#   9. Idle + start zone penalty  -> breaks parked blue robots
#  10. Pinning violation          -> penalises illegal prolonged contact
# -------------------------------------------------------------------------
REWARD_WEIGHTS = {
    # --- Terminal --------------------------------------------------------
    "win_terminal":           10.0,   # x (score_diff / 80)

    # --- Step-level score signal -----------------------------------------
    "score_delta":             5.0,   # (my_delta - opp_delta) per step

    # --- Causal scoring events -------------------------------------------
    "score_own_pin":           6.0,
    "score_yellow_owned":      8.0,
    "score_yellow_neutral":    3.0,
    "score_opp_half":         -4.0,

    # --- Cup denial ------------------------------------------------------
    "denial_success":          5.0,
    "denial_own":             -4.0,
    "denial_preserved_opp":   -1.0,

    # --- Goal stacking ---------------------------------------------------
    "stack_bonus":             0.8,

    # --- Toggle (event-only) ---------------------------------------------
    "toggle_gain":             3.0,
    "toggle_loss":            -2.0,

    # --- Carrying proximity (continuous, every step) ---------------------
    "carrying_proximity_scale": 0.15,   # max per step when at goal with correct element
    "fetch_needed_scale":        0.10,   # approach reward toward the element type needed to continue stack

    # --- Score attempt (explicit reinforcement for pressing the button) --
    "score_attempt_in_zone":   0.8,

    # --- Empty-hand approach shaping -------------------------------------
    "approach_scale":          8.0,

    # --- Object interaction ----------------------------------------------
    "intake_success":          0.8,
    "drop_penalty":           -0.5,

    # --- Penalties -------------------------------------------------------
    "holding_penalty_rate":   -0.20,   # grows to this after HOLDING_RAMP_STEPS
    "flip_penalty":           -0.15,   # small cost per flip; only flip when pin/cup color justifies it
    "idle_penalty":           -0.05,
    "start_zone_penalty":     -0.05,
    "pinning_violation":      -0.8,
    "wrong_element_loiter":   -0.08,   # per-step penalty for parking at a goal with the wrong element

    # --- Endgame ---------------------------------------------------------
    "midfield_endgame":        0.08,
}

# -------------------------------------------------------------------------
# NEW v3 CONSTANTS (holding timeout, pinning, start zone, etc.)
# -------------------------------------------------------------------------
HOLDING_TIMEOUT_STEPS = 40     # after this many steps carrying, penalty starts
HOLDING_RAMP_STEPS    = 60     # penalty reaches full strength after this many more steps

PINNING_STEPS_LIMIT   = 60     # 3 seconds at 20 Hz
PINNING_CONTACT_DIST  = 22.0   # inches (robot width + margin)

START_ZONE_RADIUS     = 20.0   # inches from spawn point
IDLE_SPEED_THRESHOLD  = 4.0    # below this = effectively stopped

GOAL_PROXIMITY_NORM   = 36.0   # normalization constant for proximity reward

# Diagonal of the 144x144 field (used for approach reward normalization)
FIELD_DIAGONAL = 203.65   # sqrt(144^2 + 144^2)

# -------------------------------------------------------------------------
# CURRICULUM STAGES
# -------------------------------------------------------------------------
CURRICULUM_STAGES = [
    {
        "id": 1,
        "name": "Solo Scoring",
        "description": "1 robot, 1 pin, 1 goal. Learn basic intake and scoring.",
        "n_robots": 1,
        "use_cups": False,
        "use_toggles": False,
        "use_endgame": False,
        "success_metric": "score_per_episode",
        "success_threshold": 5.0,
        "min_steps": 100_000,
    },
    {
        "id": 2,
        "name": "Cups and Stacking",
        "description": "Add cups; learn orientation and stack bonuses.",
        "n_robots": 2,
        "use_cups": True,
        "use_toggles": False,
        "use_endgame": False,
        "success_metric": "avg_stack_height",
        "success_threshold": 1.5,
        "min_steps": 200_000,
    },
    {
        "id": 3,
        "name": "Toggle Control",
        "description": "Toggles active; learn yellow pin prioritization.",
        "n_robots": 2,
        "use_cups": True,
        "use_toggles": True,
        "use_endgame": False,
        "success_metric": "toggle_ownership_pct",
        "success_threshold": 0.4,
        "min_steps": 300_000,
    },
    {
        "id": 4,
        "name": "Full 2v2",
        "description": "All four robots; full scoring with reduced endgame.",
        "n_robots": 4,
        "use_cups": True,
        "use_toggles": True,
        "use_endgame": False,
        "success_metric": "win_rate_vs_random",
        "success_threshold": 0.65,
        "min_steps": 500_000,
    },
    {
        "id": 5,
        "name": "Endgame",
        "description": "Full match including endgame parking and center goal lock.",
        "n_robots": 4,
        "use_cups": True,
        "use_toggles": True,
        "use_endgame": True,
        "success_metric": "win_rate_vs_prev_stage",
        "success_threshold": 0.55,
        "min_steps": 1_000_000,
    },
    {
        "id": 6,
        "name": "Endgame Specialist",
        "description": "50% episodes start at t=100s; max weight on denial+midfield.",
        "n_robots": 4,
        "use_cups": True,
        "use_toggles": True,
        "use_endgame": True,
        "late_start_prob": 0.5,
        "success_metric": "endgame_denial_rate",
        "success_threshold": 0.30,
        "min_steps": 1_500_000,
    },
]

# -------------------------------------------------------------------------
# LOGGING / CHECKPOINTING
# -------------------------------------------------------------------------
LOG_EVERY_UPDATES  = 50
SAVE_EVERY_UPDATES = 500
EVAL_EVERY_UPDATES = 500
EVAL_NUM_MATCHES   = 5
RECORD_VIDEO       = True

ARTIFACTS_DIR = "artifacts"
MODELS_DIR    = "artifacts/models"
VIDEOS_DIR    = "artifacts/videos"
LOGS_DIR      = "artifacts/logs"
HEATMAPS_DIR  = "artifacts/heatmaps"

# -------------------------------------------------------------------------
# RANDOM NETWORK DISTILLATION (RND) - Anti-Collapse / Curiosity
# -------------------------------------------------------------------------
RND_ENABLED           = True
RND_REWARD_SCALE      = 0.02          # Intrinsic reward strength
RND_UPDATE_EVERY      = 8             # Update RND network every N steps
RND_HIDDEN            = [512, 256]    # RND predictor network size
RND_LR                = 1e-4          # Learning rate for RND predictor
