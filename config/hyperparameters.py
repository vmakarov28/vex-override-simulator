"""
config/hyperparameters.py  (v9)
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

  PROBLEM 30 (v8): Proximity reward dominated causal scoring 10:1.
    FIX: carrying_proximity_scale 0.15→0.05; score_own_pin 6→15;
    score_yellow_owned 8→20; score_yellow_neutral 3→8; denial_success
    5→12; score_delta weight 5→7.  Causal events now dominate shaping.

  PROBLEM 31 (v8): Score attempt rate near zero across all training.
    FIX: score_attempt_in_zone 0.8→4.0.  Pressing the score button
    at a legal goal is now worth ~80 steps of proximity reward.

  PROBLEM 32 (v8): Linear holding-timeout ramp too weak vs proximity.
    FIX: Ramp formula changed to quadratic in env_wrapper so penalty
    becomes catastrophic after 2×HOLDING_TIMEOUT_STEPS.

  PROBLEM 33 (v8): Proximity reward fires for indefinitely parked robots.
    FIX: Proximity cuts to zero after PROX_CARRY_DECAY_STEPS (35 steps
    = 1.75 s) carry time, creating a hard deadline to score or lose pull.

  PROBLEM 34 (v8): score_delta always logs 0.000 (red+blue cancel).
    FIX: Added score_delta_red / score_delta_blue per-alliance keys to
    the reward-component tracker so the signal is visible in [Rwd] logs.

  PROBLEM 35 (v8): Toggle camping persists immediately after capture.
    FIX: TOGGLE_LEAVE_GRACE_STEPS (20) grace window after any toggle
    flip; toggle_camping -0.08→-0.15 to push robots away faster.

  PROBLEM 36 (v8): Fast cycle not rewarded enough.
    FIX: time_to_score_bonus 1.5→4.0; TIME_TO_SCORE_TARGET 80→35 steps
    (1.75 s).  A robot scoring in under 1.75 s earns the full bonus.

  PROBLEM 37 (v8): Entropy collapsing too fast onto suboptimal policy.
    FIX: ENTROPY_ANNEAL_STEPS 12M→30M; ENTROPY_COEF_MIN 0.005→0.008.

  PROBLEM 38 (v8): RND intrinsic reward near-zero (0.002 / update).
    FIX: RND_REWARD_SCALE 0.02→0.10.

  PROBLEM 39 (v8): Self-play pool too homogeneous (90% old checkpoints).
    FIX: POOL_SAMPLE_PROB 0.90→0.75.

  PROBLEM 40 (v8.1): time_to_score_bonus paid full bonus every score.
    BUG: _carry_steps was reset to 0 in step() before _compute_rewards
    ran (robot's carrying_pin became None during sim.step()).  Section 2
    read the reset value, ratio was always 1.0, fast-cycle signal
    completely absent.  FIX: step() captures pre_step_carry BEFORE the
    counter update and passes it to _compute_rewards.

  PROBLEM 41 (v8.1): No reward differentiation for endgame scoring.
    FIX: endgame_score_multiplier (1.5×) on all causal scoring events
    (pin scoring, denial, stack_bonus) while rules_engine.endgame_active.

  PROBLEM 42 (v8.1): No positive reward for defensive play.
    FIX: defensive_position_bonus (+0.05/step) when an empty-handed
    robot is positioned on the line between a carrying enemy and that
    enemy's nearest scorable goal.  Encourages blocking without contact.

  PROBLEM 43 (v8.1): No reward for grabbing elements before opponents.
    FIX: resource_denial_bonus (+0.5 one-time) when a robot intakes an
    element that an opponent was the nearest robot to at the start of
    the step.  Rewards proactive resource control.

  PROBLEM 44 (v8.1): Yellow approach only fired when alliance owned a
    toggle, missing the strategic "go grab yellow then flip toggle" path.
    FIX: Yellow approach now also fires at reduced scale
    (yellow_approach_unowned = 0.03) when alliance owns no toggles.

  PROBLEM 45 (v8.1): Observation lacked self-carry-duration awareness,
    holding-penalty anticipation, opponent carrying colour, and global
    cycle/resource state, blocking strategic reasoning.
    FIX: 10 new observation features appended after the v7 globals
    (OBS_DIM 554 → 564):
      - own carry_step normalised (1)
      - own holding-overshoot ratio (1)
      - opp1, opp2 carrying pin UP colour one-hot (3 + 3)
      - yellow pins remaining normalised (1)
      - can-score-anywhere bit (1)

  PROBLEM 46 (v8.2): Midfield parking reward fired for the entire 20-second
    endgame at a negligible rate (0.08/step × ramp), so robots never learned
    to treat parking as a discrete last-second commitment — it was just a
    weak always-on trickle.
    FIX: Reward now ONLY fires in the final PARK_WINDOW_SECONDS (3 s) of
    the match at a strong flat rate (midfield_endgame 0.08 → 1.0, no ramp).
    90 total reward for perfect 3-second parking ≈ 6 pin scores — meaningful
    but not so dominant that scoring robots abandon their last elements.
    ENDGAME_RAMP_SECONDS aligned to 3 s so obs[ptr+19] urgency ramp rises
    0→1 over exactly the same window, giving the policy a clean park-now cue.

  PROBLEM 47 (v8.2): Quadratic holding-timeout had no ceiling — at 400
    carry-steps overshoot=360, ratio=36, penalty=-7.2/step, causing
    catastrophic gradient spikes for any robot stuck in a carry loop.
    FIX: Ratio capped at HOLDING_RAMP_SQ_CAP (9.0), so the maximum
    per-step penalty is -0.20 × 9 = -1.8/step.  The cap engages at
    overshoot = 3 × HOLDING_RAMP_STEPS (~11 s of carrying) — by that
    point the penalty is already catastrophic; further escalation only
    destabilises training without changing the robot's optimal action.

  PROBLEM 48 (v8.2): CRITIC_LR (8e-4) was imported in mappo.py but
    both policy and critic used LEARNING_RATE (2.5e-4).  Slower critic
    convergence delayed value estimates, slowing policy improvement.
    FIX: Separate policy_optimizer (LEARNING_RATE) and critic_optimizer
    (CRITIC_LR) per alliance, with independent grad-norm clips.  Checkpoint
    format updated to store four optimizer state-dicts.  obs_dim also stored
    in checkpoint so mismatched architecture is detected at load time.

  PROBLEM 49 (v8.2): environment/override_env.py (PettingZoo wrapper) still
    used the old linear uncapped holding-timeout and had no PROX_CARRY_DECAY
    cut-off — evaluation rewards diverged from training rewards, making
    evaluation scores meaningless as a training signal proxy.
    FIX: PettingZoo wrapper section 3 and 5 updated to match v8 training
    wrapper (quadratic + capped timeout; proximity cut at PROX_CARRY_DECAY).

  PROBLEM 50 (v8.2): Drop+recarry exploit introduced by PROX_CARRY_DECAY.
    With v8's 35-step proximity cut-off, a robot near a goal could:
      drop (-0.5)  +  re-intake (+0.8)  =  +0.3 net, AND reset carry_steps
      to 0, granting another 35 steps × 0.05 = 1.75 of proximity reward.
    Total: ~+2.05 per drop-recarry cycle.  Not exploitable in v7 (no decay)
    but a real attack surface in v8.  COOLDOWN_INTAKE=5 only delays the cycle.
    FIX: drop_penalty -0.5 -> -1.0.  Drop+intake is now -0.2 net, removing
    the positive-feedback loop while still letting accidental drops recover.

  PROBLEM 51 (v8.2): utils/opponent_pool.py had no architecture fingerprint.
    A stale pool.pt from v7 (OBS_DIM=554) would silently load() but crash
    later inside sample() when load_state_dict() hit shape mismatches —
    cryptic failure deep inside CUDA at first opponent sample.
    FIX: Store OBS_DIM in pool file; raise ValueError on load() if it
    differs from current OBS_DIM, with instructions to delete the file.

  PROBLEM 52 (v8.2): get_action_mask() took an unused rules_engine
    parameter, suggesting legality depends on rules state when it does not.
    FIX: Removed the parameter; updated env_wrapper.py call site.

  PROBLEM 53 (v8.3): Observation was missing explicit speed-magnitude features.
    Policy had rvx/rvy for self and opponents but had to rediscover
    sqrt(rvx²+rvy²) — a nonlinear operation — slowing spin-avoidance and
    speed-reward learning.  Teammate carry_steps was also absent, preventing
    per-robot carry-duration awareness needed for goal-assignment coordination.
    FIX: 4-dim v8.2 obs block appended after v8.1:
      - own speed magnitude / MAX_SPEED (1)
      - teammate carry_steps / TIME_TO_SCORE_TARGET (1)
      - opp1 speed magnitude / MAX_SPEED (1)
      - opp2 speed magnitude / MAX_SPEED (1)
    Also: per-pin nearest-goal distance added to each pin slot (matches what
    cups already had), giving robots directional guidance on which pin to
    fetch based on proximity to the intended goal.  Pins 20×10→20×11 dims.
    OBS_DIM 564 → 588.

  PROBLEM 54 (v8.3): forward_speed_scale (0.03/step) was only 30% the
    strength of spin_penalty (-0.10/step).  Policy learned "don't spin" but
    had weak incentive to go fast — the positive signal wasn't worth changing
    locomotion patterns.  Also: no reward for raw speed magnitude while
    carrying (only heading-aligned component).
    FIX 1: forward_speed_scale 0.03 → 0.06 (doubles direction-aligned
    speed reward, now balanced against spin_penalty).
    FIX 2: carrying_speed_scale = 0.015/step fires on raw |v|/MAX_SPEED
    while carrying, creating a direction-agnostic "go fast with the goods"
    layer.  Max total from both signals: 0.075/step; holding_timeout cap
    at -1.8/step dominates after step ~45, preventing circling exploit.

  PROBLEM 55 (v8.3): time_to_score_bonus rewards the carry phase (pickup →
    score) but the fetch phase (score → next pickup) was shaped only by the
    delta-based approach_scale, which saturates at zero once near the target.
    No positive one-time signal motivated urgency in the return-fetch phase.
    FIX: intake_cycle_bonus = 1.5 fires at pickup: bonus × max(0, 1 −
    steps_since_last_score / INTAKE_CYCLE_TARGET).  A robot returning
    quickly from its last score earns up to 1.5; a slow robot earns
    progressively less.  Drop-exploit guard: bonus only fires when the
    most recent relevant event (score or drop) was a score, preventing the
    drop→re-intake cycle from collecting the bonus twice.

  PROBLEM 56 (v8.3): environment/override_env.py (PettingZoo wrapper)
    midfield_endgame section fired for the entire 20-second endgame at
    1.0/step → 400 reward per robot per game.  Training wrapper correctly
    gates it to the final PARK_WINDOW_SECONDS (3 s) → 60 reward per robot.
    The 27× inflation made evaluation scores incomparable to training, and
    could cause the evaluation policy to over-value parking vs. scoring.
    FIX: Added `tr <= PARK_WINDOW_SECONDS` gate to PettingZoo wrapper
    section 12, matching training/env_wrapper.py v8.2 behaviour exactly.

  PROBLEM 57 (v9): Drop-cycle exploit at 16 M steps.
    With drop_penalty=-1.0 and intake_success=+0.8, robots learned to
    pick up an element and drop it every ~35 steps rather than carry it to
    a goal.  Net cycle cost: +0.8 - 1.0 = -0.2/cycle, far cheaper than the
    holding_timeout penalty ramp for stationary robots.  Additionally, the
    holding_timeout penalty fired on ANY carrier regardless of speed, so
    actively moving carriers were also penalized, discouraging carries.
    FIX 1: drop_penalty -1.0 → -2.0 (closes drop-cycle loop: +0.8 - 2.0 = -1.2).
    FIX 2: holding_timeout now speed-gated — only fires when robot speed
    < IDLE_SPEED_THRESHOLD.  Actively moving carriers are exempt; the penalty
    exclusively targets stationary/parked holds.

  PROBLEM 58 (v9): Pinning exploit never self-corrected (25× escalation).
    pinning_violation=-0.8 was ~11× weaker than score_delta=7.0.  A pinning
    robot could deny 1 score per 8.75 steps of pinning and still come out
    ahead.  Log shows pinning_violation went from -8.6 at 81k steps to
    -205 at 13 M steps, never reversing.
    FIX: pinning_violation -0.8 → -4.0 (5× increase).  A pinning robot
    now has to deny >1 score per 1.75 steps to break even — not feasible.
    Also: PINNING_STEPS_LIMIT 60 → 40 (2 s threshold instead of 3 s).

  PROBLEM 59 (v9): No absolute own-score signal — score_delta (my − opp)
    cancels to near-zero in symmetric self-play, giving no unambiguous
    "put points on the board" gradient.
    FIX: own_score_abs = 5.0 fires on my alliance's score increase alone
    (unconditional, same direction for both robots on the scoring team).

  PROBLEM 60 (v9): No reward for scoring while under defensive pressure.
    Robots near goals would retreat when an opponent approached rather than
    completing the score — defensive harassment trivially stopped scoring.
    FIX: score_under_pressure = 5.0 one-time bonus at score moment when any
    opponent is within PINNING_CONTACT_DIST.  Also: win_threshold_bonus = 10.0
    fires once when lead first crosses +15 pts, giving a clear "win state"
    signal not present in step-level score_delta.

  PROBLEM 61 (v9): SC5b violation — center goal yellow halves scored live
    via toggle ownership.  Per the VEX Override manual, yellow halves placed
    in the center (midfield) goal are NOT owned by toggles; ownership is
    decided at match end by midfield robot majority (strict majority wins
    them all, ties leave them unclaimed at 0 pts).  The previous code
    awarded toggle-based yellow points for the center goal LIVE, leading
    to videos where both alliances banked yellow points in the center goal
    during play despite neither having midfield majority.  Regular red/blue
    halves in the center goal are unaffected — they continue to score live.
    FIX (game scoring):  FieldGoal.get_score now zeroes center-goal yellow
    halves when midfield_majority is None (live) or "tie".  RulesEngine
    .calculate_final_score recounts robots in the Midfield at the exact
    match-end instant and applies the SC5b majority (or "tie" → 0).
    FIX (reward shaping):  env_wrapper section 10 skips center-goal yellow
    halves at placement; the count is tallied and paid out in the terminal
    block based on the same midfield-majority rule.

  PROBLEM 62 (v9.1): Robots entered midfield during the last 20 s but
    drifted out before the old 3-s park window opened — earning zero SC5b
    reward for the commitment.  The narrow window also had no exit cost, so
    cycling in-and-out was free.
    FIX: PARK_WINDOW_SECONDS 3 → 8 s.  Both midfield_endgame and
    sc5b_park_bonus are ramped linearly (rate = weight × progress, where
    progress = (8-tr)/8 → 0 at t=8 s, 1 at t=0 s), so being in midfield
    later is always worth more per step.  A new midfield_exit_penalty fires
    once when a robot leaves the midfield during the window, scaled
    quadratically: penalty = midfield_exit_penalty × progress².  The
    quadratic exponent means exit later = more punishment, and because
    penalty > step-reward at every progress value, leave-and-reenter
    cycling is always net-negative.  ENDGAME_RAMP_SECONDS updated to 8 s
    so obs[ptr+19] urgency ramp rises from the same 8-s horizon.

  PROBLEM 63 (v9.1): Allied robots crash into and pin each other.
    teammate_overlap_penalty (−0.12) and the small (0.02) binary
    ally_separation_bonus were too weak to deter allied collisions.  No
    direct cost existed for physical contact between allies.
    FIX: ally_contact_penalty = −3.0/step when allies are within
    ALLY_CONTACT_DIST (14 in).  teammate_overlap_penalty −0.12 → −2.0
    (×17) so same-goal crowding is expensive, not trivial.

  PROBLEM 64 (v9.1): Ally separation had no gradient — robots at 44 in
    apart earned the same +0 as robots touching (binary threshold at 45 in).
    FIX: separation bonus now scales linearly with distance:
    reward = ally_separation_bonus × min(1, d / ALLY_SEPARATION_TARGET).
    Gradient always pulls teammates apart, not just past a cliff edge.
"""

# -------------------------------------------------------------------------
# OBSERVATION / ACTION SPACE
# -------------------------------------------------------------------------
OBS_DIM     = 592   # v9: 588 base + 4 (v9 pressure features: in_scoring_range, being_pinned_frac, score_lead_tight, dist_to_nearest_scorable)
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
ENTROPY_COEF_MIN         = 0.008
ENTROPY_ANNEAL_STEPS     = 30_000_000  # Keep exploration longer
MAX_GRAD_NORM            = 0.5

# -------------------------------------------------------------------------
# ROLLOUT / BATCH
# -------------------------------------------------------------------------
NUM_PARALLEL_ENVS = 32
ROLLOUT_STEPS     = 512
MINIBATCH_SIZE    = 256
PPO_EPOCHS        = 10
TOTAL_ENV_STEPS   = 6_500_000    # ~5 hour run (~410 steps/sec × 17,000s budget)

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
POOL_SAMPLE_PROB       = 0.75        # Sample historical opponents 75% of time
CHECKPOINT_EVERY       = 500
SELF_PLAY_OPPONENT_MIX = 0.65        # 65% pool + 35% latest policy

# -------------------------------------------------------------------------
# REWARD WEIGHTS  (v8)
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
    "score_delta":             3.5,   # (my_delta - opp_delta) per step — v9: 7.0→3.5 to reduce symmetric cancellation
    "own_score_abs":           5.0,   # v9: unconditional own-alliance score increase (always positive gradient)

    # --- Causal scoring events -------------------------------------------
    "score_own_pin":          15.0,
    "score_yellow_owned":     20.0,
    "score_yellow_neutral":    8.0,
    "score_opp_half":         -4.0,

    # --- Cup denial ------------------------------------------------------
    "denial_success":         12.0,
    "denial_own":             -4.0,
    "denial_preserved_opp":   -1.0,

    # --- Goal stacking ---------------------------------------------------
    "stack_bonus":             0.8,

    # --- Toggle (event-only) ---------------------------------------------
    "toggle_gain":             3.0,
    "toggle_loss":            -2.0,

    # --- Carrying proximity (continuous, every step) ---------------------
    "carrying_proximity_scale": 0.05,   # max per step when at goal with correct element
    "fetch_needed_scale":        0.08,   # approach reward toward the element type needed to continue stack

    # --- Score attempt (explicit reinforcement for pressing the button) --
    "score_attempt_in_zone":   4.0,

    # --- Empty-hand approach shaping -------------------------------------
    "approach_scale":          8.0,

    # --- Object interaction ----------------------------------------------
    "intake_success":          0.8,
    "drop_penalty":           -2.0,   # v9: -1.0 → -2.0 to close drop-cycle exploit (PROBLEM 57); +0.8-2.0=-1.2 net

    # --- Penalties -------------------------------------------------------
    "holding_penalty_rate":   -0.20,   # grows to this after HOLDING_RAMP_STEPS
    "flip_penalty":           -0.15,   # small cost per flip; only flip when pin/cup color justifies it
    "idle_penalty":           -0.05,
    "start_zone_penalty":     -0.05,
    "pinning_violation":      -4.0,   # v9: -0.8 → -4.0 (5×) to make pinning unprofitable vs score_delta (PROBLEM 58)
    "wrong_element_loiter":   -0.15,   # per-step: carrying wrong element within scoring radius of goal
    "spin_penalty":           -0.10,   # per-step: high angular velocity + low translational speed
    "toggle_camping":         -0.15,   # per-step: loitering near a toggle your alliance already owns
    "forward_speed_scale":     0.06,   # per-step: velocity component pointing toward current target (v8.3: 0.03→0.06)
    "carrying_speed_scale":    0.015,  # per-step: raw speed bonus when carrying — direction-agnostic "go fast" layer
    "intake_cycle_bonus":      1.5,    # one-time at intake: bonus × max(0, 1 - steps_since_last_score/INTAKE_CYCLE_TARGET)

    # --- v7: division of labour & cycle efficiency -----------------------
    "teammate_overlap_penalty": -2.0,   # v9.1: -0.12→-2.0 (×17) — same-goal crowding now clearly expensive (PROBLEM 63)
    "time_to_score_bonus":       4.0,   # one-time at score moment: bonus × max(0, 1 - carry_steps/TARGET)
    "yellow_approach_scale":     0.06,  # per-step: bonus when empty robot approaches yellow pin & alliance owns toggle
    "ally_separation_bonus":     0.02,  # per-step: gradient bonus = weight × min(1, d/TARGET); 0 when touching, full at TARGET (PROBLEM 64)
    "ally_contact_penalty":     -3.0,   # v9.1: per-step when allies within ALLY_CONTACT_DIST (PROBLEM 63)

    # --- v8.1: Strategy & defensive shaping ------------------------------
    "endgame_score_multiplier":  1.5,    # multiplier on causal scoring events during endgame
    "resource_denial_bonus":     0.5,    # one-time bonus when intaking an element an opponent was closer to
    "defensive_position_bonus":  0.05,   # per-step: empty-handed robot between a carrying enemy and their nearest scorable goal
    "yellow_approach_unowned":   0.03,   # yellow_approach scale when alliance doesn't yet own a toggle (encourages flip-first strategy)

    # --- v9: Anti-hijack / pressure-scoring rewards ----------------------
    "score_under_pressure":      5.0,    # one-time at score moment when any opponent is within PINNING_CONTACT_DIST
    "win_threshold_bonus":      10.0,    # one-time when lead first crosses +15 pts
    "sc5b_park_bonus":           1.5,    # per-step × progress ramp; 0 at t=8 s, 1.5/step at t=0 s (SC5b strategic shaping)

    # --- Endgame ---------------------------------------------------------
    "midfield_endgame":          1.0,   # per-step × progress ramp (0 at window open → 1.0/step at t=0); 8-s window (PROBLEM 62)
    "midfield_exit_penalty":   -40.0,   # v9.1: one-time on midfield exit during window; scaled ×progress² so late exits cost more (PROBLEM 62)
}

# -------------------------------------------------------------------------
# NEW v3 CONSTANTS (holding timeout, pinning, start zone, etc.)
# -------------------------------------------------------------------------
HOLDING_TIMEOUT_STEPS = 40     # after this many steps carrying, penalty starts
HOLDING_RAMP_STEPS    = 60     # one "ramp unit" for the quadratic formula
# Quadratic cap: ratio = min((overshoot/HOLDING_RAMP_STEPS)^2, CAP)
# Cap = 9.0  →  max penalty = |holding_penalty_rate| × 9 = 1.8/step.
# Engages at overshoot = 3 × HOLDING_RAMP_STEPS = 180 steps (~9 s of carry).
HOLDING_RAMP_SQ_CAP   = 9.0

PINNING_STEPS_LIMIT   = 40     # v9: 3 s → 2 s at 20 Hz (tighter limit alongside 5× penalty increase)
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
ALLY_CONTACT_DIST       = 14.0   # v9.1: inches — below this triggers ally_contact_penalty (≈robot width)

# Time-to-score: a robot that scores within TIME_TO_SCORE_TARGET steps of
# picking up earns close-to-full bonus; longer carries fade to zero linearly.
# 35 steps at 20 Hz = 1.75 seconds of carrying.
TIME_TO_SCORE_TARGET    = 30     # v9: 35→30 steps (1.5 s); tighter fast-cycle window

# Endgame midfield ramp: in the final ENDGAME_RAMP_SECONDS of the match the
# midfield_endgame reward multiplier ramps linearly from 1× → ENDGAME_RAMP_MAX_MULT.
# This makes second-1 parking >> second-19 parking, teaching last-second commit.
ENDGAME_RAMP_SECONDS    = 8.0    # v9.1: 3→8 s to align obs urgency ramp with the wider park window (PROBLEM 62)
ENDGAME_RAMP_MAX_MULT   = 4.0   # unused by section 12 since v8.2 (no ramp); kept for reference

# v9.1: park window extended 3→8 s.  Reward ramps with progress so later
# parking is always worth more per step.  Exit penalty deters leaving.
PARK_WINDOW_SECONDS     = 8.0

# -------------------------------------------------------------------------
# v8.3: Cycle-speed / carrying-drive constants
# -------------------------------------------------------------------------
# Intake cycle bonus target: a robot that returns from its last score and
# picks up the next element within this many steps earns the full bonus;
# bonus scales to 0 at >= INTAKE_CYCLE_TARGET steps.  50 steps = 2.5 s.
INTAKE_CYCLE_TARGET     = 60     # v9: 50→60 steps (3 s window); easier for robots to earn full cycle bonus

# -------------------------------------------------------------------------
# v8: Cycle-efficiency / toggle-leave constants
# -------------------------------------------------------------------------
# Proximity reward hard cut-off: after carrying for this many steps without
# scoring, carrying_proximity_scale drops to zero.  Creates a 1.75-second
# deadline before the carrot disappears (then holding timeout stick starts).
PROX_CARRY_DECAY_STEPS  = 35

# Grace window after a toggle flip: robots are NOT penalised for toggle_camping
# during this window, giving them time to physically leave the toggle zone.
TOGGLE_LEAVE_GRACE_STEPS = 20

# Defensive blocking: max perpendicular distance from the (opp → opp_goal) line
# at which a defender counts as "in the way".  ~1 robot width.
DEFENSIVE_LINE_PERP_DIST = 18.0

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
LOG_EVERY_UPDATES  = 10
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
RND_REWARD_SCALE      = 0.10          # Intrinsic reward strength
RND_UPDATE_EVERY      = 8             # Update RND network every N steps
RND_HIDDEN            = [512, 256]    # RND predictor network size
RND_LR                = 1e-4          # Learning rate for RND predictor
