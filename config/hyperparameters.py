"""
config/hyperparameters.py  (v7)
=================================
REWARD REDESIGN v7
------------------
Cumulative fixes (v3/v5 problems preserved for history; v6 adds new fixes):

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

  PROBLEM 15 (v6): Robots spin in place rather than drive.
    FIX: spin_penalty fires every step when angular velocity is high AND
    translational speed is low, making in-place spinning unprofitable.

  PROBLEM 16 (v6): Robots camp near an already-owned toggle all match.
    FIX: toggle_camping penalty fires every step a robot lingers within
    TOGGLE_CAMP_RADIUS of a toggle its own alliance already controls.

  PROBLEM 17 (v6): No direct incentive to drive quickly toward targets.
    FIX: forward_speed_scale rewards the component of velocity that points
    toward the robot's current target (goal when carrying, nearest element
    when empty), giving a dense signal for purposeful fast movement.

  PROBLEM 18 (v6): Wrong-element loiter penalty too weak (was -0.08).
    FIX: Raised to -0.15/step so the penalty beats the proximity reward
    at close range, actively expelling the robot from unserviceable goals.

  PROBLEM 19 (v6): fetch_needed_scale too weak (was 0.10).
    FIX: Raised to 0.15 to more aggressively redirect robots toward the
    element type they need to resume scoring.

  PROBLEM 20 (v6): Intake/scoring radii too generous (16/14 in).
    FIX: SCORING_RADIUS 16→12, INTAKE_RADIUS 14→10 in game_rules.py so
    robots must approach precisely before interacting.

  PROBLEM 21 (v7): Both alliance robots crowd the same goal.
    FIX: teammate_overlap_penalty fires when both alliance robots sit
    within scoring radius of the same goal simultaneously, encouraging
    division of labour.

  PROBLEM 22 (v7): No positive feedback for fast cycles.
    FIX: time_to_score_bonus pays out at the scoring moment, scaled
    inversely by how long the robot had been carrying — short carries
    earn more than long ones.

  PROBLEM 23 (v7): Yellow-pin priority only kicks in at score time.
    FIX: yellow_approach_scale gives extra approach reward to empty-handed
    robots heading toward yellow-sided pins when their alliance currently
    owns the relevant toggle, biasing pickup choices earlier.

  PROBLEM 24 (v7): Robots park in midfield the entire endgame.
    FIX: midfield_endgame ramps linearly from 1× → ENDGAME_RAMP_MAX_MULT
    over the last ENDGAME_RAMP_SECONDS so late-second parking is
    strongly preferred over second-19 parking.

  PROBLEM 25 (v7): Robots couldn't reason about score deficit or
    spinning behaviour at the policy level.
    FIX: Observation gains 3 new global features —
      (a) alliance-relative score delta  (my − opp) / 80
      (b) heading-vs-velocity cosine alignment for self
      (c) endgame urgency ramp (0 outside endgame, 0→1 inside).
    OBS_DIM updated from 551 → 554.

  PROBLEM 26 (v7): Robots couldn't selectively attend to relevant goals.
    FIX: Policy/Critic now use a per-goal embedding + dot-product
    attention pooling over the 9 goal slots, with the non-goal context
    serving as the query.  Replaces blind MLP flattening.

  PROBLEM 27 (v7): No visibility into which reward signals were firing.
    FIX: env_wrapper now tracks running per-component reward sums.
    drain_reward_components() exposes them; training loops log them.

  PROBLEM 28 (v7): Robots crowded the same element / goal.
    FIX: ally_separation_bonus pays a small per-step reward when the two
    alliance robots are at least ALLY_SEPARATION_TARGET inches apart.

  PROBLEM 29 (v7): Visualisation cadence too sparse during early learning.
    FIX: First 2 M env steps record a vis video every 100 updates rather
    than every 500.  Also: a final result video is recorded at end of run.
"""

# -------------------------------------------------------------------------
# OBSERVATION / ACTION SPACE
# -------------------------------------------------------------------------
OBS_DIM     = 554   # v6: 551 base + 3 v7 features (score_delta_my, heading-vel align, endgame urgency)
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
TOTAL_ENV_STEPS   = 40_000_000

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
    "fetch_needed_scale":        0.15,   # approach reward toward the element type needed to continue stack

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
    "wrong_element_loiter":   -0.15,   # per-step: carrying wrong element within scoring radius of goal
    "spin_penalty":           -0.10,   # per-step: high angular velocity + low translational speed
    "toggle_camping":         -0.08,   # per-step: loitering near a toggle your alliance already owns
    "forward_speed_scale":     0.03,   # per-step: velocity component pointing toward current target

    # --- v7: division of labour & cycle efficiency -----------------------
    "teammate_overlap_penalty": -0.12,  # per-step: both alliance robots inside SCORING_RADIUS of same goal
    "time_to_score_bonus":       1.5,   # one-time at score moment: bonus × max(0, 1 - carry_steps/TARGET)
    "yellow_approach_scale":     0.06,  # per-step: bonus when empty robot approaches yellow pin & alliance owns toggle
    "ally_separation_bonus":     0.02,  # per-step: bonus when teammates >= ALLY_SEPARATION_TARGET apart

    # --- Endgame ---------------------------------------------------------
    "midfield_endgame":        0.08,   # base per-step reward; multiplied by 1..ENDGAME_RAMP_MAX_MULT in final seconds
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
# v6: Spin / camping / forward-speed constants
# -------------------------------------------------------------------------
# Spin penalty fires when |angular_velocity| > threshold AND trans_speed < threshold.
# Angular velocity is in rad/s (Pymunk default).  A robot spinning freely in place
# typically exceeds 2 rad/s; normal turning while driving is usually below 1.5 rad/s.
SPIN_ANG_VEL_THRESHOLD  = 2.0    # rad/s — above this counts as "spinning"
SPIN_TRANS_THRESHOLD    = 20.0   # inches/s — below this = not meaningfully moving forward

# Toggle camping: penalise lingering near a toggle the robot's own alliance already owns.
# Set slightly wider than TOGGLE_INTERACTION_RANGE (18 in) so a robot just sitting
# adjacent to an owned toggle also gets penalised.
TOGGLE_CAMP_RADIUS      = 24.0   # inches

# -------------------------------------------------------------------------
# v7: Division-of-labour / endgame ramp constants
# -------------------------------------------------------------------------
# Two alliance robots are penalised if both end up inside SCORING_RADIUS of
# the same goal (camping/crowding behaviour).  Detection uses SCORING_RADIUS
# from game_rules directly — no extra constant needed here.

# Ally separation target — minimum distance (inches) between teammates that
# earns the per-step `ally_separation_bonus`.  ~3 robot lengths apart.
ALLY_SEPARATION_TARGET  = 45.0

# Time-to-score: a robot that scores within TIME_TO_SCORE_TARGET steps of
# picking up earns close-to-full bonus; longer carries fade to zero linearly.
# 80 steps at 20 Hz = 4 seconds of carrying.
TIME_TO_SCORE_TARGET    = 80

# Endgame midfield ramp: in the final ENDGAME_RAMP_SECONDS of the match the
# midfield_endgame reward multiplier ramps linearly from 1× → ENDGAME_RAMP_MAX_MULT.
# This makes second-1 parking >> second-19 parking, teaching last-second commit.
ENDGAME_RAMP_SECONDS    = 10.0
ENDGAME_RAMP_MAX_MULT   = 4.0

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
