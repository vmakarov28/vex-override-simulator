"""
training/env_wrapper.py  (v9.2)
==========================================================================
Reward shaping history:
  v4 — Holding ramp, anti-reward-hacking baseline
  v5 — Legality-gated proximity, fetch-needed redirect, wrong-element
        loiter, causal cup denial, yellow toggle-owner attribution,
        score-attempt legality gate, top-pin UP color observation (551-dim)
  v6 — Spin penalty (in-place spinning), toggle-camping penalty,
        forward-speed reward (velocity toward target), strengthened
        wrong_element_loiter (-0.15) and fetch_needed_scale (0.15),
        tighter intake/scoring radii (game_rules.py: 10/12 in)
  v7 — Teammate-goal overlap penalty (division of labour),
        time-to-score bonus (positive cycle-time signal),
        yellow-pin approach (early toggle-aware prioritisation),
        ally separation bonus (anti-crowding), ramping midfield_endgame
        (1×→4× across final 10 s).  +3 obs features (554-dim) and
        per-component reward tracking exposed via drain_reward_components().
  v8 — Cycle-efficiency overhaul: proximity hard cut-off at 35 carry-steps,
        quadratic holding-timeout ramp, proximity_scale 0.15→0.05, causal
        events 2-3× larger, score_attempt_in_zone 0.8→4.0, time_to_score
        bonus 1.5→4.0 (35-step target), toggle grace window, per-alliance
        score_delta logging, RND 5× stronger, entropy anneal 2.5× longer.
  v8.3 — Drive-speed overhaul: forward_speed_scale 0.03→0.06,
        carrying_speed_scale (0.015/step raw speed while carrying),
        intake_cycle_bonus (1.5 one-time at pickup based on return speed
        from last score; drop-exploit guarded).  +20 per-pin obs dims,
        +4 movement-intelligence obs dims (588-dim total).
  v9   — Drop-cycle exploit fix: drop_penalty -1.0→-2.0; holding_timeout
        speed-gated (only fires when robot speed < IDLE_SPEED_THRESHOLD,
        exempting actively-moving carriers).  Pinning: penalty -0.8→-4.0,
        PINNING_STEPS_LIMIT 60→40.  New rewards: own_score_abs (absolute
        own-alliance scoring), score_under_pressure (scoring while being
        contested), win_threshold_bonus (one-time +15-pt lead crossing).
        +4 obs dims (592-dim total): in_scoring_range, being_pinned_frac,
        score_lead_tight, dist_to_nearest_scorable.
  v9.1 — Park window 3→8 s with progress ramp (later = more reward/step).
        midfield_exit_penalty fires on midfield exit during window, scaled
        ×progress² so late exits cost more; anti-exploit proven.
        ally_contact_penalty −3.0/step when allies within ALLY_CONTACT_DIST.
        ally_separation_bonus upgraded to linear gradient (0→full over
        0→ALLY_SEPARATION_TARGET).  teammate_overlap_penalty −0.12→−2.0.
  v9.2 — Yellow self-cancel gap fixed (PROBLEM 65): section 10 cup placement
        now handles yellow-topped pins via toggle ownership; three new causal
        events: yellow_self_cancel (−8), yellow_deny_bonus (+12),
        yellow_preserve_bonus (+3).  Cup orientation pre-placement shaping
        added as section 3g (PROBLEM 66): proximity-weighted reward/penalty
        for carrying cup in correct/wrong orientation while approaching a
        goal with a pin on top.  OBS_DIM 592→610: two new per-goal obs
        features (yellow_toggle_mine, cup_place_quality) giving the network
        direct context for cup-flip decisions (PROBLEM 67).
        Requires fresh training run.
"""

import math
import time
import itertools
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List

from simulation.simulator import OverrideSimulator
from utils.observation_builder import build_all_observations, get_action_mask, OBS_DIM
from config.hyperparameters import (
    CONTROL_DT, MAX_EPISODE_STEPS, REWARD_WEIGHTS,
    COOLDOWN_TOGGLE, COOLDOWN_FLIP, COOLDOWN_SCORE,
    COOLDOWN_MATCH_LOAD, COOLDOWN_INTAKE,
    IDLE_SPEED_THRESHOLD, FIELD_DIAGONAL,
    START_ZONE_RADIUS, HOLDING_TIMEOUT_STEPS, HOLDING_RAMP_STEPS,
    PINNING_STEPS_LIMIT, PINNING_CONTACT_DIST,
    GOAL_PROXIMITY_NORM,
    SPIN_ANG_VEL_THRESHOLD, SPIN_TRANS_THRESHOLD, TOGGLE_CAMP_RADIUS,
    ALLY_SEPARATION_TARGET, ALLY_CONTACT_DIST, TIME_TO_SCORE_TARGET,
    PROX_CARRY_DECAY_STEPS, TOGGLE_LEAVE_GRACE_STEPS,
    DEFENSIVE_LINE_PERP_DIST, PARK_WINDOW_SECONDS,
    HOLDING_RAMP_SQ_CAP, INTAKE_CYCLE_TARGET,
)
from config.game_rules import (
    SCORING_RADIUS, ENDGAME_SECONDS, TOTAL_SECONDS,
    AUTONOMOUS_SECONDS, MIDFIELD_HALF, ROBOT_STARTS,
    ROBOT_MAX_SPEED, CENTER_GOAL_ID,
)
from simulation.game_objects import C_RED, C_BLUE, C_YELLOW
from training.rnd import RNDModule
from config.hyperparameters import RND_ENABLED, RND_UPDATE_EVERY

AGENT_IDS = ["red1", "red2", "blue1", "blue2"]
ROBOT_PAIRS = list(itertools.combinations(AGENT_IDS, 2))


class OverrideEnv:
    """Single-instance VEX Override Gym-style environment."""

    def __init__(self, headless: bool = True, seed: int = 0,
                 late_start_prob: float = 0.0):
        self.headless        = headless
        self.late_start_prob = late_start_prob
        self.rng             = np.random.default_rng(seed)
        self._step_count     = 0

        self.sim = OverrideSimulator(headless=headless)

        self._prev_red_score  = 0
        self._prev_blue_score = 0
        self._prev_carrying_pin  = {rid: None for rid in AGENT_IDS}
        self._prev_carrying_cup  = {rid: None for rid in AGENT_IDS}
        self._prev_toggle_owners = {}

        self._cooldowns: Dict[str, Dict[str, int]] = {
            rid: {"intake": 0, "score_pin": 0, "score_cup": 0,
                  "toggle": 0, "flip_pin": 0, "flip_cup": 0, "match_load": 0}
            for rid in AGENT_IDS
        }

        self._carry_steps: Dict[str, int]           = {rid: 0 for rid in AGENT_IDS}
        self._contact_steps: Dict[Tuple, int]       = {p: 0 for p in ROBOT_PAIRS}
        self._start_positions: Dict[str, Tuple]     = {}
        self._prev_target_dist: Dict[str, float]    = {rid: 0.0 for rid in AGENT_IDS}
        self._prev_scores: Dict[str, int]           = {rid: 0 for rid in AGENT_IDS}

        # v8.3: full-cycle bonus tracking
        # _steps_since_last_score: steps elapsed since this robot last scored.
        # Initialised large so first-episode pickups earn no cycle bonus.
        # _last_drop_step: global step count of last drop event, used to
        # guard against the drop→re-intake cycle-bonus exploit.
        self._steps_since_last_score: Dict[str, int] = {rid: MAX_EPISODE_STEPS + 1
                                                         for rid in AGENT_IDS}
        self._last_drop_step: Dict[str, int]          = {rid: -1 for rid in AGENT_IDS}

        # v7: per-component reward sum tracker (summed across all robots and
        # all steps since last drain).  drain_reward_components() returns the
        # current dict and resets it — used by training loops for TB-style
        # per-signal logging.
        self._reward_components: Dict[str, float]   = {}

        # v9 SC5b: visible yellow halves placed in the CENTER goal this episode.
        # Live reward is suppressed; payout occurs in the terminal block
        # based on midfield robot majority at match end.
        self._pending_center_yellow_halves: int = 0

        # v9.1: per-robot midfield state from the PREVIOUS step.
        # Used to detect exits (was_in=True, is_in=False) for the exit penalty.
        # Updated every step (not just during the park window) so the state
        # is accurate when the window opens.
        self._prev_midfield: Dict[str, bool] = {rid: False for rid in AGENT_IDS}

        # v8: toggle grace tracker — {toggle_id: steps_remaining} so camping
        # penalty doesn't fire immediately after a successful flip.
        self._toggle_grace: Dict[int, int] = {}

        self.rnd = RNDModule(obs_dim=OBS_DIM, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) if RND_ENABLED else None

    # -------------------------------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        self.sim.reset()
        self._step_count     = 0
        self._prev_red_score  = 0
        self._prev_blue_score = 0
        self._prev_carrying_pin  = {rid: None for rid in AGENT_IDS}
        self._prev_carrying_cup  = {rid: None for rid in AGENT_IDS}
        self._prev_toggle_owners = {t.toggle_id: t.owner for t in self.sim.toggles}
        self._carry_steps        = {rid: 0 for rid in AGENT_IDS}
        self._toggle_grace = {}
        self._contact_steps      = {p: 0 for p in ROBOT_PAIRS}
        self._prev_scores        = {rid: 0 for rid in AGENT_IDS}
        self._steps_since_last_score = {rid: MAX_EPISODE_STEPS + 1 for rid in AGENT_IDS}
        self._last_drop_step         = {rid: -1 for rid in AGENT_IDS}
        self._pending_center_yellow_halves = 0  # v9 SC5b deferred-yellow counter
        self._prev_midfield = {rid: False for rid in AGENT_IDS}  # v9.1 exit-penalty tracker
        # NOTE: do NOT clear self._reward_components here.  Per-component
        # sums persist across episodes within a rollout so the training loop
        # can drain them once per update cycle for logging.

        self._start_positions = {
            rid: (float(ROBOT_STARTS[rid]["pos"][0]),
                  float(ROBOT_STARTS[rid]["pos"][1]))
            for rid in AGENT_IDS
        }

        for rid in AGENT_IDS:
            for k in self._cooldowns[rid]:
                self._cooldowns[rid][k] = 0

        robot_map = {r.robot_id: r for r in self.sim.robots}
        for rid in AGENT_IDS:
            self._prev_target_dist[rid] = self._empty_target_dist(rid, robot_map)

        if self.late_start_prob > 0 and self.rng.random() < self.late_start_prob:
            self.sim.match_phase    = "driver"
            self.sim.time_remaining = 20.0
            self.sim.time_elapsed   = TOTAL_SECONDS - 20.0
            self.sim.timer_started  = True
            self.sim.rules_engine.endgame_active = True

        self.sim.timer_started = True
        return build_all_observations(self.sim)

    # -------------------------------------------------------------------------
    def step(
        self,
        actions: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[Dict, Dict, bool, Dict]:
        pre_red  = self.sim.rules_engine.red_score
        pre_blue = self.sim.rules_engine.blue_score
        pre_goal_stacks   = {g.goal_id: list(g.stack) for g in self.sim.goals}
        pre_toggle_owners = {t.toggle_id: t.owner    for t in self.sim.toggles}
        pre_carrying_pin  = {r.robot_id: r.carrying_pin for r in self.sim.robots}
        pre_carrying_cup  = {r.robot_id: r.carrying_cup for r in self.sim.robots}
        pre_positions     = {r.robot_id: (float(r.body.position.x),
                                          float(r.body.position.y))
                             for r in self.sim.robots}

        robot_map = {r.robot_id: r for r in self.sim.robots}

        pre_empty_dist = {
            rid: self._empty_target_dist(rid, robot_map) for rid in AGENT_IDS
        }

        sim_actions      = [None] * 4
        flips_fired      = {rid: False for rid in AGENT_IDS}
        score_attempted  = {rid: False for rid in AGENT_IDS}

        for i, rid in enumerate(AGENT_IDS):
            if rid not in actions:
                sim_actions[i] = _zero_action()
                continue

            cont, disc = actions[rid]
            if hasattr(cont, "detach"): cont = cont.detach().cpu().numpy()
            if hasattr(disc, "detach"): disc = disc.detach().cpu().numpy()

            cd  = self._cooldowns[rid]
            rob = robot_map[rid]

            def _gate(cd_name, raw_bit):
                return bool(raw_bit > 0.5) and cd[cd_name] == 0

            intake     = _gate("intake",     disc[0])
            score_pin  = _gate("score_pin",  disc[1])
            score_cup  = _gate("score_cup",  disc[2])
            flip_pin   = _gate("flip_pin",   disc[4])
            flip_cup   = _gate("flip_cup",   disc[5])
            match_load = _gate("match_load", disc[6])

            toggle_raw = bool(disc[3] > 0.5) and cd["toggle"] == 0
            toggle = toggle_raw and self._toggle_is_legal(rob)

            sim_actions[i] = {
                "left":      float(np.clip(cont[0], -1.0, 1.0)),
                "right":     float(np.clip(cont[1], -1.0, 1.0)),
                "intake":    intake,
                "score_pin": score_pin,
                "score_cup": score_cup,
                "toggle":    toggle,
                "flip_pin":  flip_pin,
                "flip_cup":  flip_cup,
            }

            if flip_pin or flip_cup:
                flips_fired[rid] = True
            if score_pin or score_cup:
                score_attempted[rid] = True

            if intake:     cd["intake"]     = COOLDOWN_INTAKE
            if score_pin:  cd["score_pin"]  = COOLDOWN_SCORE
            if score_cup:  cd["score_cup"]  = COOLDOWN_SCORE
            if toggle:     cd["toggle"]     = COOLDOWN_TOGGLE
            if flip_pin:   cd["flip_pin"]   = COOLDOWN_FLIP
            if flip_cup:   cd["flip_cup"]   = COOLDOWN_FLIP
            if match_load: cd["match_load"] = COOLDOWN_MATCH_LOAD

        for rid in AGENT_IDS:
            for k in self._cooldowns[rid]:
                if self._cooldowns[rid][k] > 0:
                    self._cooldowns[rid][k] -= 1

        # Decrement toggle grace counters each step
        for tid in list(self._toggle_grace.keys()):
            if self._toggle_grace[tid] > 1:
                self._toggle_grace[tid] -= 1
            else:
                del self._toggle_grace[tid]

        # v8.1: snapshot per-robot carry-step counters BEFORE sim.step() can
        # reset them (a successful score sets carrying_pin=None during
        # sim.step(), which would then zero the counter before
        # _compute_rewards reads it).  time_to_score_bonus needs the
        # pre-step value to know how long the carry actually lasted.
        pre_step_carry = dict(self._carry_steps)

        # v8.1: nearest robot to each unowned element BEFORE this step's
        # actions resolve.  Used by resource_denial_bonus: a robot that
        # intakes an element an opponent was closer to gets a strategic
        # one-time bonus.
        pre_element_nearest_rid: Dict[int, str] = {}
        for _p in self.sim.pins:
            if _p.scored or _p.carried_by is not None:
                continue
            _px = float(_p.body.position.x); _py = float(_p.body.position.y)
            _best_rid = None; _best_d = float('inf')
            for _r in self.sim.robots:
                _d = math.hypot(float(_r.body.position.x) - _px,
                                float(_r.body.position.y) - _py)
                if _d < _best_d:
                    _best_d, _best_rid = _d, _r.robot_id
            if _best_rid is not None:
                pre_element_nearest_rid[id(_p)] = _best_rid
        for _c in self.sim.cups:
            if _c.scored or _c.carried_by is not None:
                continue
            _cx = float(_c.body.position.x); _cy = float(_c.body.position.y)
            _best_rid = None; _best_d = float('inf')
            for _r in self.sim.robots:
                _d = math.hypot(float(_r.body.position.x) - _cx,
                                float(_r.body.position.y) - _cy)
                if _d < _best_d:
                    _best_d, _best_rid = _d, _r.robot_id
            if _best_rid is not None:
                pre_element_nearest_rid[id(_c)] = _best_rid

        self.sim.step(CONTROL_DT, sim_actions)
        self._step_count += 1

        # v8.3: advance cycle counter for all robots; reset to 0 in
        # _compute_rewards when a score is detected for that robot.
        for rid in AGENT_IDS:
            self._steps_since_last_score[rid] += 1

        post_red  = self.sim.rules_engine.red_score
        post_blue = self.sim.rules_engine.blue_score
        robot_map = {r.robot_id: r for r in self.sim.robots}

        post_empty_dist = {
            rid: self._empty_target_dist(rid, robot_map) for rid in AGENT_IDS
        }

        # Update carry steps
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is not None or r.carrying_cup is not None:
                self._carry_steps[rid] += 1
            else:
                self._carry_steps[rid] = 0

        # v8.1: expose carry_step counter to observation_builder via a
        # public attribute on each robot.  observation_builder reads
        # `robot._carry_steps`; default 0 if env_wrapper has not set it yet.
        for rid in AGENT_IDS:
            r = robot_map[rid]
            r._carry_steps = self._carry_steps[rid]

        # Update contact steps
        for (a, b) in ROBOT_PAIRS:
            ra = robot_map[a]; rb = robot_map[b]
            ax, ay = float(ra.body.position.x), float(ra.body.position.y)
            bx, by = float(rb.body.position.x), float(rb.body.position.y)
            dist   = math.hypot(ax - bx, ay - by)
            if dist < PINNING_CONTACT_DIST:
                self._contact_steps[(a, b)] += 1
            else:
                self._contact_steps[(a, b)] = 0

        # Build observations once; pass to reward fn so RND avoids redundant builds
        obs = build_all_observations(self.sim)

        rewards = self._compute_rewards(
            pre_red, pre_blue, post_red, post_blue,
            pre_goal_stacks, pre_toggle_owners,
            pre_carrying_pin, pre_carrying_cup,
            sim_actions, robot_map,
            pre_empty_dist, post_empty_dist,
            flips_fired, score_attempted,
            pre_positions,
            pre_carry_steps=pre_step_carry,
            pre_element_nearest_rid=pre_element_nearest_rid,
            post_obs=obs,
        )

        self._prev_red_score     = post_red
        self._prev_blue_score    = post_blue
        self._prev_toggle_owners = {t.toggle_id: t.owner for t in self.sim.toggles}
        for r in self.sim.robots:
            self._prev_carrying_pin[r.robot_id] = r.carrying_pin
            self._prev_carrying_cup[r.robot_id] = r.carrying_cup
        for rid in AGENT_IDS:
            self._prev_target_dist[rid] = post_empty_dist[rid]

        done = self.sim.match_over or self._step_count >= MAX_EPISODE_STEPS

        if done:
            # v9 SC5b: deferred payout for yellow halves placed in the center
            # goal.  Count robots in the Midfield right now; alliance with
            # strict majority claims all visible yellow halves; ties (0-0,
            # 1-1, 2-2) leave them unclaimed.  Use the same shaping weight
            # the live non-center yellow path uses so reward magnitude is
            # comparable across goal types.
            n_yellow = self._pending_center_yellow_halves
            if n_yellow > 0:
                rc_mid = sum(1 for r in self.sim.robots
                             if r.alliance == "red" and
                             self.sim.rules_engine.is_robot_in_midfield(r))
                bc_mid = sum(1 for r in self.sim.robots
                             if r.alliance == "blue" and
                             self.sim.rules_engine.is_robot_in_midfield(r))
                yval = REWARD_WEIGHTS["score_yellow_owned"] * n_yellow
                if rc_mid > bc_mid:
                    for rid in ["red1",  "red2"]: rewards[rid] += yval
                    for rid in ["blue1", "blue2"]: rewards[rid] += REWARD_WEIGHTS["score_opp_half"] * n_yellow
                elif bc_mid > rc_mid:
                    for rid in ["blue1", "blue2"]: rewards[rid] += yval
                    for rid in ["red1",  "red2"]: rewards[rid] += REWARD_WEIGHTS["score_opp_half"] * n_yellow
                # tie → 0 reward (yellows unclaimed)
            self._pending_center_yellow_halves = 0

            diff = post_red - post_blue
            wt   = REWARD_WEIGHTS["win_terminal"]
            if diff > 0:
                for rid in ["red1",  "red2"]:  rewards[rid] += wt * (diff / 80.0)
                for rid in ["blue1", "blue2"]: rewards[rid] -= wt * (diff / 80.0)
            elif diff < 0:
                for rid in ["blue1", "blue2"]: rewards[rid] += wt * (-diff / 80.0)
                for rid in ["red1",  "red2"]:  rewards[rid] -= wt * (-diff / 80.0)

        info = {
            "red_score":   post_red,
            "blue_score":  post_blue,
            "match_phase": self.sim.match_phase,
            "step":        self._step_count,
        }
        return obs, rewards, done, info

    # =========================================================================
    # REWARD COMPUTATION (v8)
    # =========================================================================
    def _compute_rewards(
        self,
        pre_red, pre_blue, post_red, post_blue,
        pre_goal_stacks, pre_toggle_owners,
        pre_carrying_pin, pre_carrying_cup,
        sim_actions, robot_map,
        pre_empty_dist, post_empty_dist,
        flips_fired, score_attempted,
        pre_positions,
        pre_carry_steps: Dict[str, int] = None,
        pre_element_nearest_rid: Dict[int, str] = None,
        post_obs: Dict[str, np.ndarray] = None,
    ) -> Dict[str, float]:
        rw      = REWARD_WEIGHTS
        rewards = {rid: 0.0 for rid in AGENT_IDS}
        if pre_carry_steps is None:
            pre_carry_steps = dict(self._carry_steps)
        if pre_element_nearest_rid is None:
            pre_element_nearest_rid = {}

        # v7 per-component tracking: snapshot total reward at each section
        # boundary so we can attribute the delta to that section's signal.
        rc = self._reward_components
        _prev_total = 0.0
        def _track(name):
            nonlocal _prev_total
            cur = sum(rewards.values())
            rc[name] = rc.get(name, 0.0) + (cur - _prev_total)
            _prev_total = cur

        # 1. Score delta
        red_delta  = post_red  - pre_red
        blue_delta = post_blue - pre_blue
        for rid in ["red1",  "red2"]:  rewards[rid] += rw["score_delta"] * (red_delta  - blue_delta)
        for rid in ["blue1", "blue2"]: rewards[rid] += rw["score_delta"] * (blue_delta - red_delta)
        _track("score_delta")

        # Per-alliance breakdown (the combined sum is always 0; these are diagnostic)
        rc["score_delta_red"]  = rc.get("score_delta_red",  0.0) + rw["score_delta"] * (red_delta  - blue_delta) * 2
        rc["score_delta_blue"] = rc.get("score_delta_blue", 0.0) + rw["score_delta"] * (blue_delta - red_delta)  * 2

        # 1b. Absolute own-score reward (v9) — unconditional own-alliance score
        # increase signal so there is always a positive gradient for "put points on
        # the board", regardless of opponent score and self-play symmetry.
        for rid in ["red1",  "red2"]:  rewards[rid] += rw["own_score_abs"] * red_delta
        for rid in ["blue1", "blue2"]: rewards[rid] += rw["own_score_abs"] * blue_delta
        _track("own_score_abs")

        # 1c. Win-threshold bonus (v9): one-time reward when lead first crosses +15 pts.
        # Gives the policy a discrete "win state" target beyond incremental score_delta.
        pre_r_lead  = pre_red  - pre_blue
        post_r_lead = post_red - post_blue
        if post_r_lead >= 15 and pre_r_lead < 15:
            for rid in ["red1", "red2"]:  rewards[rid] += rw["win_threshold_bonus"]
        pre_b_lead  = pre_blue  - pre_red
        post_b_lead = post_blue - post_red
        if post_b_lead >= 15 and pre_b_lead < 15:
            for rid in ["blue1", "blue2"]: rewards[rid] += rw["win_threshold_bonus"]
        _track("win_threshold")

        # 2. Intake / drop  (+ v7 time-to-score bonus, v8.1 resource denial,
        #                    v8.3 intake_cycle_bonus)
        # v8.1 FIX: time_to_score_bonus reads from pre_carry_steps, NOT
        # self._carry_steps.  The latter is reset to 0 in step() before we
        # run if the robot just scored (carrying_pin became None during
        # sim.step()).  Reading the post-update value gave full bonus
        # every score regardless of actual carry duration.
        for rid in AGENT_IDS:
            r       = robot_map[rid]
            had_pin = pre_carrying_pin[rid] is not None
            had_cup = pre_carrying_cup[rid] is not None
            now_pin = r.carrying_pin is not None
            now_cup = r.carrying_cup is not None

            new_pin_in = now_pin and not had_pin
            new_cup_in = now_cup and not had_cup

            if new_pin_in or new_cup_in:
                rewards[rid] += rw["intake_success"]
                # v8.1 resource denial bonus: did we beat an opponent to it?
                newly_carried = r.carrying_pin if new_pin_in else r.carrying_cup
                if newly_carried is not None:
                    nearest_rid = pre_element_nearest_rid.get(id(newly_carried))
                    if nearest_rid is not None and nearest_rid != rid:
                        nearest_robot = robot_map.get(nearest_rid)
                        if nearest_robot is not None and nearest_robot.alliance != r.alliance:
                            rewards[rid] += rw["resource_denial_bonus"]

                # v8.3: intake_cycle_bonus — rewards fast return from last score
                # to next pickup (fetch phase of the cycle).
                # Guard: only fires when the most recent relevant event was a
                # SCORE, not a DROP.  Prevents drop→re-intake exploit where the
                # bonus would fire on re-intake after a deliberate drop.
                # last_score_step approximated as (step_count - steps_since_last_score).
                last_score_step = self._step_count - self._steps_since_last_score[rid]
                if last_score_step > self._last_drop_step[rid]:
                    cycle_steps = self._steps_since_last_score[rid]
                    cycle_ratio = max(0.0, 1.0 - cycle_steps / float(INTAKE_CYCLE_TARGET))
                    if cycle_ratio > 0.0:
                        rewards[rid] += rw["intake_cycle_bonus"] * cycle_ratio

            scored = r.successful_scores > self._prev_scores.get(rid, 0)
            if (had_pin and not now_pin and not scored) or \
               (had_cup and not now_cup and not scored):
                rewards[rid] += rw["drop_penalty"]
                self._last_drop_step[rid] = self._step_count  # v8.3: track for exploit guard

            if scored:
                # v8.1 FIX: use PRE-step carry counter — post-update value
                # is 0 because the score caused carrying_pin/cup to become None.
                carry_steps = pre_carry_steps.get(rid, 0)
                ratio = max(0.0, 1.0 - carry_steps / float(TIME_TO_SCORE_TARGET))
                rewards[rid] += rw["time_to_score_bonus"] * ratio
                # v8.3: reset cycle counter so next intake measures return speed
                self._steps_since_last_score[rid] = 0
                # v9: score-under-pressure bonus — reward scoring while being contested
                srx = float(robot_map[rid].body.position.x)
                sry = float(robot_map[rid].body.position.y)
                under_pressure = any(
                    robot_map[oid].alliance != robot_map[rid].alliance and
                    math.hypot(float(robot_map[oid].body.position.x) - srx,
                               float(robot_map[oid].body.position.y) - sry) < PINNING_CONTACT_DIST
                    for oid in AGENT_IDS
                )
                if under_pressure:
                    rewards[rid] += rw["score_under_pressure"]

        for r in self.sim.robots:
            self._prev_scores[r.robot_id] = r.successful_scores
        _track("intake_drop_and_time_to_score")

        # 3. Carrying proximity reward — only when a legal score exists at a nearby goal.
        # A robot holding the wrong element for every reachable goal earns nothing here,
        # preventing the "park at goal unable to score" exploit.
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is None and r.carrying_cup is None:
                continue
            # v8: hard cut-off — no proximity reward after PROX_CARRY_DECAY_STEPS
            if self._carry_steps[rid] >= PROX_CARRY_DECAY_STEPS:
                continue
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            best = None
            for g in self.sim.goals:
                if g.alliance not in ("neutral", r.alliance):
                    continue
                can_pin = r.carrying_pin is not None and not _stack_top_is_pin(g.stack)
                can_cup = r.carrying_cup is not None and     _stack_top_is_pin(g.stack)
                if can_pin or can_cup:
                    d = math.hypot(rx - g.x, ry - g.y)
                    if best is None or d < best:
                        best = d
            if best is not None:
                prox = rw["carrying_proximity_scale"] / (1.0 + best / GOAL_PROXIMITY_NORM)
                rewards[rid] += prox
        _track("carrying_proximity")

        # 3b. Fetch-needed-element redirect reward.
        # When a robot holds an element it cannot currently score anywhere, reward it
        # for approaching the specific element type it is missing so it can complete
        # the stack and return.
        for rid in AGENT_IDS:
            r = robot_map[rid]
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            # Determine whether this robot can score anywhere right now.
            can_score_anywhere = any(
                g.alliance in ("neutral", r.alliance) and (
                    (r.carrying_pin is not None and not _stack_top_is_pin(g.stack)) or
                    (r.carrying_cup is not None and     _stack_top_is_pin(g.stack))
                )
                for g in self.sim.goals
            )
            if can_score_anywhere:
                continue  # proximity reward above already pulls this robot to the goal
            needs_pin = r.carrying_cup is not None and r.carrying_pin is None
            needs_cup = r.carrying_pin is not None and r.carrying_cup is None
            if not needs_pin and not needs_cup:
                continue
            best = None
            if needs_pin:
                for p in self.sim.pins:
                    if p.scored or p.carried_by is not None:
                        continue
                    d = math.hypot(rx - float(p.body.position.x),
                                   ry - float(p.body.position.y))
                    if best is None or d < best:
                        best = d
            else:
                for c in self.sim.cups:
                    if c.scored or c.carried_by is not None:
                        continue
                    d = math.hypot(rx - float(c.body.position.x),
                                   ry - float(c.body.position.y))
                    if best is None or d < best:
                        best = d
            if best is not None:
                rewards[rid] += rw["fetch_needed_scale"] / (1.0 + best / GOAL_PROXIMITY_NORM)
        _track("fetch_needed")

        # 3c. Wrong-element loitering penalty.
        # Small per-step cost when a robot lingers within scoring radius of a goal
        # where it cannot make a legal score.  This breaks the stable equilibrium of
        # parking at an unreachable goal while the holding-timeout ramp is still low.
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is None and r.carrying_cup is None:
                continue
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            for g in self.sim.goals:
                if g.alliance not in ("neutral", r.alliance):
                    continue
                if math.hypot(rx - g.x, ry - g.y) > SCORING_RADIUS * 1.5:
                    continue
                can_pin = r.carrying_pin is not None and not _stack_top_is_pin(g.stack)
                can_cup = r.carrying_cup is not None and     _stack_top_is_pin(g.stack)
                if not can_pin and not can_cup:
                    rewards[rid] += rw["wrong_element_loiter"]
                    break
        _track("wrong_element_loiter")

        # 3d. v7 — Teammate-goal overlap penalty.
        # Penalise the situation where both alliance robots are inside
        # SCORING_RADIUS of the same goal.  Encourages division of labour:
        # one robot scores, the other goes back to fetch the next element.
        for alliance in ("red", "blue"):
            allies = [robot_map[r] for r in AGENT_IDS if robot_map[r].alliance == alliance]
            if len(allies) < 2:
                continue
            a, b = allies[0], allies[1]
            ax, ay = float(a.body.position.x), float(a.body.position.y)
            bx, by = float(b.body.position.x), float(b.body.position.y)
            for g in self.sim.goals:
                if g.alliance not in ("neutral", alliance):
                    continue
                a_in = math.hypot(ax - g.x, ay - g.y) < SCORING_RADIUS
                b_in = math.hypot(bx - g.x, by - g.y) < SCORING_RADIUS
                if a_in and b_in:
                    # Split the penalty between both crowders so we don't
                    # arbitrarily blame one robot.
                    rewards[a.robot_id] += 0.5 * rw["teammate_overlap_penalty"]
                    rewards[b.robot_id] += 0.5 * rw["teammate_overlap_penalty"]
                    break  # one penalty per pair per step
        _track("teammate_overlap")

        # 3e. v9.1 — Ally separation gradient bonus (PROBLEM 64).
        # v7 used a binary threshold (>= TARGET → full bonus, else 0).
        # v9.1 replaces this with a linear gradient:
        #   reward = ally_separation_bonus × min(1, d / ALLY_SEPARATION_TARGET)
        # This provides a smooth pull away from the teammate at all distances
        # (0 when touching, full bonus at TARGET, capped beyond TARGET).
        for alliance in ("red", "blue"):
            allies = [robot_map[r] for r in AGENT_IDS if robot_map[r].alliance == alliance]
            if len(allies) < 2:
                continue
            a, b = allies[0], allies[1]
            d = math.hypot(float(a.body.position.x) - float(b.body.position.x),
                           float(a.body.position.y) - float(b.body.position.y))
            ratio = min(1.0, d / ALLY_SEPARATION_TARGET)
            rewards[a.robot_id] += rw["ally_separation_bonus"] * ratio
            rewards[b.robot_id] += rw["ally_separation_bonus"] * ratio
        _track("ally_separation")

        # 3e.1 v9.1 — Ally contact penalty (PROBLEM 63).
        # Direct per-step penalty when two allied robots are within
        # ALLY_CONTACT_DIST inches (physical contact zone).  Stops ally-on-ally
        # pinning/pushing that blocks both robots from efficiently cycling.
        # Complements the separation gradient: gradient pulls apart at range,
        # contact penalty punishes actual collisions.
        for alliance in ("red", "blue"):
            allies = [robot_map[r] for r in AGENT_IDS if robot_map[r].alliance == alliance]
            if len(allies) < 2:
                continue
            a, b = allies[0], allies[1]
            d = math.hypot(float(a.body.position.x) - float(b.body.position.x),
                           float(a.body.position.y) - float(b.body.position.y))
            if d < ALLY_CONTACT_DIST:
                rewards[a.robot_id] += rw["ally_contact_penalty"]
                rewards[b.robot_id] += rw["ally_contact_penalty"]
        _track("ally_contact")

        # 3g. v9.2 — Cup orientation pre-placement shaping (PROBLEM 66).
        # Fires per-step when a robot is carrying a cup within SCORING_RADIUS×3
        # of any legal goal whose top element is a pin.  Rewards the robot for
        # already holding the cup in the correct orientation for that goal:
        #   correct = eff_clear_up=False (clear side down) when preserving own pin
        #           = eff_clear_up=True  (dark side down)  when denying opp pin
        #           = eff_clear_up=False (clear side down) when own alliance owns the yellow toggle
        #           = eff_clear_up=True  (dark side down)  when opp alliance owns the yellow toggle
        # Proximity-weighted (same kernel as carrying_proximity).
        # Shapes pre-placement behaviour so robots flip the cup BEFORE arriving
        # at the goal, not after — by then it's too late.
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_cup is None:
                continue
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            cup_eff_clear = _eff_clear_up(r.carrying_cup)
            best_goal_reward = 0.0
            for g in self.sim.goals:
                if g.alliance not in ("neutral", r.alliance):
                    continue
                if not g.stack or not g.stack[-1][1]:  # top must be a pin
                    continue
                d = math.hypot(rx - g.x, ry - g.y)
                if d > SCORING_RADIUS * 3:
                    continue
                top_obj, _ = g.stack[-1]
                top_up_col = top_obj.get_up_color()
                # Determine the correct cup orientation for this goal
                if top_up_col == C_RED or top_up_col == C_BLUE:
                    # Solid color: deny opp (dark-down=True), preserve own (clear-down=False)
                    top_is_own = (top_up_col == C_RED and r.alliance == "red") or \
                                 (top_up_col == C_BLUE and r.alliance == "blue")
                    correct_clear_up = not top_is_own
                elif top_up_col == C_YELLOW:
                    # Yellow: depends on toggle ownership
                    towner = _get_toggle_for_goal(g, list(self.sim.toggles))
                    if towner == r.alliance:
                        correct_clear_up = False  # preserve own yellow: clear side down
                    elif towner is not None:
                        correct_clear_up = True   # deny opponent yellow: dark side down
                    else:
                        continue  # unowned toggle — no orientation preference yet
                else:
                    continue
                prox = 1.0 / (1.0 + d / GOAL_PROXIMITY_NORM)
                if cup_eff_clear == correct_clear_up:
                    goal_r = rw["cup_orient_correct"] * prox
                else:
                    goal_r = rw["cup_orient_wrong"] * prox
                # Only count the single most-relevant goal per robot
                if abs(goal_r) > abs(best_goal_reward):
                    best_goal_reward = goal_r
            rewards[rid] += best_goal_reward
        _track("cup_orientation")

        # 3f. Yellow-pin approach reward (v7 + v8.1 strategic extension).
        # v7: empty-handed robot whose alliance owns ANY toggle gets a
        # pull toward the nearest yellow-sided pin (so pickup priority
        # exists BEFORE the score moment, not just at it).
        # v8.1: also fires at reduced scale when the alliance owns NO
        # toggle, encouraging the "grab yellow → flip toggle → score"
        # strategic chain.
        any_yellow_pin = any(
            (not p.scored and p.carried_by is None) and
            (p.get_up_color() == C_YELLOW or p.get_down_color() == C_YELLOW)
            for p in self.sim.pins
        )
        if any_yellow_pin:
            for rid in AGENT_IDS:
                r = robot_map[rid]
                if r.carrying_pin is not None or r.carrying_cup is not None:
                    continue
                my_owns_toggle = any(t.owner == r.alliance for t in self.sim.toggles)
                scale = (rw["yellow_approach_scale"] if my_owns_toggle
                         else rw["yellow_approach_unowned"])
                if scale <= 0.0:
                    continue
                rx, ry = float(r.body.position.x), float(r.body.position.y)
                best_d = None
                for p in self.sim.pins:
                    if p.scored or p.carried_by is not None:
                        continue
                    if not (p.get_up_color() == C_YELLOW or p.get_down_color() == C_YELLOW):
                        continue
                    d = math.hypot(rx - float(p.body.position.x),
                                   ry - float(p.body.position.y))
                    if best_d is None or d < best_d:
                        best_d = d
                if best_d is not None:
                    rewards[rid] += scale / (1.0 + best_d / GOAL_PROXIMITY_NORM)
        _track("yellow_approach")

        # 4. Score attempt reward — only fires when the attempt is actually legal
        # (correct element for the goal's current stack state).  Prevents rewarding
        # robots for repeatedly pressing score on a goal they can never fill.
        for rid in AGENT_IDS:
            if not score_attempted[rid]:
                continue
            r = robot_map[rid]
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            for g in self.sim.goals:
                if g.alliance not in ("neutral", r.alliance):
                    continue
                if math.hypot(rx - g.x, ry - g.y) > SCORING_RADIUS + 4.0:
                    continue
                can_pin = r.carrying_pin is not None and not _stack_top_is_pin(g.stack)
                can_cup = r.carrying_cup is not None and     _stack_top_is_pin(g.stack)
                if can_pin or can_cup:
                    rewards[rid] += rw["score_attempt_in_zone"]
                    break
        _track("score_attempt")

        # 5. Holding timeout penalty — quadratic ramp after HOLDING_TIMEOUT_STEPS.
        # v9: Speed-gated — only fires when robot speed < IDLE_SPEED_THRESHOLD.
        # Actively-moving carriers are exempt; the penalty exclusively targets
        # stationary/parked carries.  Fixes the drop-cycle exploit where moving
        # carriers were penalised identically to parked ones, making dropping
        # cheaper than continuing to the goal (PROBLEM 57).
        #
        # Diagnostics (v9, no reward effect): track how often the gate fires
        # vs. how often it exempts a moving carrier.  Visible in [Rwd] log as
        # `holding_gate_fired` (penalty applied) and `holding_gate_exempt`
        # (over-timeout carrier was moving and skipped the penalty).
        n_fired = 0
        n_exempt = 0
        for rid in AGENT_IDS:
            cs = self._carry_steps[rid]
            if cs > HOLDING_TIMEOUT_STEPS:
                r   = robot_map[rid]
                spd = math.hypot(float(r.body.velocity.x), float(r.body.velocity.y))
                if spd < IDLE_SPEED_THRESHOLD:
                    overshoot = cs - HOLDING_TIMEOUT_STEPS
                    ratio     = min((overshoot / HOLDING_RAMP_STEPS) ** 2,
                                    HOLDING_RAMP_SQ_CAP)
                    rewards[rid] += rw["holding_penalty_rate"] * ratio
                    n_fired += 1
                else:
                    n_exempt += 1
        # Per-step fractions averaged across n_component_rollouts at log time.
        rc["holding_gate_fired"]  = rc.get("holding_gate_fired",  0.0) + (n_fired  / float(len(AGENT_IDS)))
        rc["holding_gate_exempt"] = rc.get("holding_gate_exempt", 0.0) + (n_exempt / float(len(AGENT_IDS)))
        _track("holding_timeout")

        # 6. Empty-hand approach delta
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is None and r.carrying_cup is None:
                delta_d = pre_empty_dist[rid] - post_empty_dist[rid]
                if delta_d > 0:
                    rewards[rid] += rw["approach_scale"] * delta_d / FIELD_DIAGONAL
        _track("approach_delta")

        # 7. Flip penalty (neutral — no reward or punishment)
        for rid in AGENT_IDS:
            if flips_fired[rid]:
                rewards[rid] += rw["flip_penalty"]
        _track("flip_penalty")

        # 8. Idle + start-zone penalty
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is not None or r.carrying_cup is not None:
                continue
            vx    = float(r.body.velocity.x)
            vy    = float(r.body.velocity.y)
            speed = math.hypot(vx, vy)
            if speed < IDLE_SPEED_THRESHOLD:
                rewards[rid] += rw["idle_penalty"]
            sx, sy = self._start_positions.get(rid, (0.0, 0.0))
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            if math.hypot(rx - sx, ry - sy) < START_ZONE_RADIUS:
                rewards[rid] += rw["start_zone_penalty"]
        _track("idle_start_zone")

        # 9. Pinning violation
        for (a, b) in ROBOT_PAIRS:
            if self._contact_steps[(a, b)] <= PINNING_STEPS_LIMIT:
                continue
            ra = robot_map[a]; rb = robot_map[b]
            if ra.alliance == rb.alliance:
                continue
            sp_a = math.hypot(float(ra.body.velocity.x), float(ra.body.velocity.y))
            sp_b = math.hypot(float(rb.body.velocity.x), float(rb.body.velocity.y))
            if sp_a >= sp_b:
                rewards[a] += rw["pinning_violation"]
            else:
                rewards[b] += rw["pinning_violation"]
        _track("pinning")

        # 10. Goal-level causal scoring events
        # v8.1: scale causal scoring events during endgame so last-minute
        # scoring is worth substantially more.
        endgame_mult = rw["endgame_score_multiplier"] if self.sim.rules_engine.endgame_active else 1.0
        for goal in self.sim.goals:
            pre_stack  = pre_goal_stacks.get(goal.goal_id, [])
            post_stack = list(goal.stack)
            if len(post_stack) <= len(pre_stack):
                continue

            new_obj, new_is_pin = post_stack[-1]
            scoring_rid = None
            for rid in AGENT_IDS:
                if pre_carrying_pin.get(rid) is new_obj and new_is_pin:
                    scoring_rid = rid; break
                if pre_carrying_cup.get(rid) is new_obj and not new_is_pin:
                    scoring_rid = rid; break
            if scoring_rid is None:
                continue

            scorer_alliance = robot_map[scoring_rid].alliance
            ally_rids = [r for r in AGENT_IDS if robot_map[r].alliance == scorer_alliance]

            if new_is_pin:
                pin_idx = len(pre_stack)  # 0-based position of the new pin in the stack

                # Determine which halves are actually visible at placement time.
                # UP half: always visible — it is now the top of the stack.
                # DOWN half: hidden by the goal post for the first pin (index 0);
                #            for deeper pins, visible only if the cup below has its
                #            clear side facing up (dark side down = blocks nothing).
                up_vis = True
                if pin_idx == 0:
                    down_vis = False  # goal post always hides the bottom-most pin's DOWN half
                else:
                    prev_obj, prev_is_pin = post_stack[pin_idx - 1]
                    down_vis = _eff_clear_up(prev_obj) if not prev_is_pin else True

                towner = _get_toggle_for_goal(goal, list(self.sim.toggles))
                is_center_goal = (goal.goal_id == CENTER_GOAL_ID)

                for visible, col in ((up_vis, new_obj.get_up_color()),
                                     (down_vis, new_obj.get_down_color())):
                    if not visible:
                        continue  # hidden half scores 0 pts; don't signal for it
                    if col == C_RED:
                        r_val = rw["score_own_pin"] if scorer_alliance == "red" else rw["score_opp_half"]
                        b_val = rw["score_opp_half"] if scorer_alliance == "red" else rw["score_own_pin"]
                    elif col == C_BLUE:
                        r_val = rw["score_opp_half"] if scorer_alliance == "red" else rw["score_own_pin"]
                        b_val = rw["score_own_pin"] if scorer_alliance == "blue" else rw["score_opp_half"]
                    elif col == C_YELLOW:
                        # SC5b: yellow halves placed in the CENTER goal do not
                        # award live reward — their ownership is decided by
                        # midfield robot majority at match end.  Track the
                        # visible half count for deferred payout in the
                        # terminal block below; skip the live reward here.
                        if is_center_goal:
                            self._pending_center_yellow_halves += 1
                            continue
                        # Non-center yellow: toggle-based reward (live).
                        if towner == "red":
                            r_val = rw["score_yellow_owned"]
                            b_val = rw["score_opp_half"]
                        elif towner == "blue":
                            r_val = rw["score_opp_half"]
                            b_val = rw["score_yellow_owned"]
                        else:
                            r_val = rw["score_yellow_neutral"]
                            b_val = rw["score_yellow_neutral"]
                    else:
                        r_val = b_val = 0.0
                    for rid in ["red1", "red2"]: rewards[rid] += r_val * endgame_mult
                    for rid in ["blue1", "blue2"]: rewards[rid] += b_val * endgame_mult
            else:
                # Cup placed on top of a pin.
                # Visibility rule (mirrors game_objects.py get_score):
                #   eff_clear_up=True  → clear side up, dark side DOWN → dark bottom
                #                        faces the pin's UP half → pin UP is BLOCKED.
                #   eff_clear_up=False → dark side up, clear side DOWN → clear bottom
                #                        faces the pin's UP half → pin UP is VISIBLE.
                eff_clear = _eff_clear_up(new_obj)
                cup_idx = len(pre_stack)
                if cup_idx > 0:
                    below_obj, below_is_pin = post_stack[cup_idx - 1]
                    if below_is_pin:
                        pin_up_col = below_obj.get_up_color()
                        if eff_clear:
                            # Dark bottom faces pin — pin's UP half is BLOCKED (denied).
                            if _is_opponent_color(pin_up_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_success"] * endgame_mult
                            elif _is_own_color(pin_up_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_own"] * endgame_mult
                            elif pin_up_col == C_YELLOW:
                                # v9.2 (PROBLEM 65): yellow not caught by _is_own/opp_color.
                                # Dark side blocks yellow UP half — evaluate via toggle.
                                towner = _get_toggle_for_goal(goal, list(self.sim.toggles))
                                if towner == scorer_alliance:
                                    # Blocking OWN alliance's yellow — strong self-cancel penalty
                                    for rid in ally_rids: rewards[rid] += rw["yellow_self_cancel"] * endgame_mult
                                elif towner is not None:
                                    # Blocking OPPONENT's yellow — good denial!
                                    for rid in ally_rids: rewards[rid] += rw["yellow_deny_bonus"] * endgame_mult
                                # towner is None (unowned toggle): no causal yellow reward
                        else:
                            # Clear bottom faces pin — pin's UP half stays VISIBLE.
                            if _is_opponent_color(pin_up_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_preserved_opp"] * endgame_mult
                            elif pin_up_col == C_YELLOW:
                                # v9.2 (PROBLEM 65): clear side preserves yellow UP half.
                                towner = _get_toggle_for_goal(goal, list(self.sim.toggles))
                                if towner == scorer_alliance:
                                    # Correctly preserving OWN alliance's yellow — bonus!
                                    for rid in ally_rids: rewards[rid] += rw["yellow_preserve_bonus"] * endgame_mult
                                elif towner is not None:
                                    # Preserving OPPONENT's yellow — same as denial_preserved_opp
                                    for rid in ally_rids: rewards[rid] += rw["denial_preserved_opp"] * endgame_mult
                if cup_idx >= 1:
                    for rid in ally_rids: rewards[rid] += rw["stack_bonus"] * endgame_mult
        _track("causal_scoring")

        # 11. Toggle events
        for toggle in self.sim.toggles:
            prev_owner = pre_toggle_owners.get(toggle.toggle_id)
            curr_owner = toggle.owner
            if prev_owner != curr_owner:
                if curr_owner == "red":
                    for rid in ["red1", "red2"]: rewards[rid] += rw["toggle_gain"]
                    for rid in ["blue1", "blue2"]: rewards[rid] += rw["toggle_loss"]
                elif curr_owner == "blue":
                    for rid in ["blue1", "blue2"]: rewards[rid] += rw["toggle_gain"]
                    for rid in ["red1", "red2"]: rewards[rid] += rw["toggle_loss"]
                # v8: grant grace window so robots aren't immediately penalised
                # for toggle_camping right after a successful flip.
                self._toggle_grace[toggle.toggle_id] = TOGGLE_LEAVE_GRACE_STEPS
        _track("toggle_events")

        # 12. Midfield park bonus — v9.1: extended to 8-s window with progress
        # ramp.  rate = midfield_endgame × progress, where
        # progress = (PARK_WINDOW_SECONDS − tr) / PARK_WINDOW_SECONDS.
        # → 0/step at t=8 s (window open), 1.0/step at t=0 s (match end).
        # Being in midfield later is ALWAYS worth more per step.
        # Total max (stay full 8 s): ∑ progress × steps ≈ +80 per robot.
        # obs[ptr+19] urgency ramp is now aligned to the same 8-s horizon
        # (ENDGAME_RAMP_SECONDS = 8) so the policy has a clean contextual cue.
        tr = float(self.sim.time_remaining)
        if self.sim.rules_engine.endgame_active and tr <= PARK_WINDOW_SECONDS:
            progress = max(0.0, (PARK_WINDOW_SECONDS - tr) / PARK_WINDOW_SECONDS)
            step_rate = rw["midfield_endgame"] * progress
            mc_x = mc_y = 72.0
            for r in self.sim.robots:
                rx = float(r.body.position.x)
                ry = float(r.body.position.y)
                if abs(rx - mc_x) + abs(ry - mc_y) <= MIDFIELD_HALF + 10.0:
                    rewards[r.robot_id] += step_rate
        _track("midfield_endgame")

        # 12b. v9.1 SC5b park bonus — same 8-s progress ramp as midfield_endgame.
        # Rate = sc5b_park_bonus × progress: 0/step at t=8 s, 1.5/step at t=0 s.
        # Fires unconditionally (no yellow requirement) so the policy is
        # incentivised both to CLAIM yellows and to DENY them via tie-forcing.
        # Combined peak: (1.0 + 1.5) × 1.0 = 2.5/step at match end.
        if self.sim.rules_engine.endgame_active and tr <= PARK_WINDOW_SECONDS:
            progress = max(0.0, (PARK_WINDOW_SECONDS - tr) / PARK_WINDOW_SECONDS)
            for r in self.sim.robots:
                if self.sim.rules_engine.is_robot_in_midfield(r):
                    rewards[r.robot_id] += rw["sc5b_park_bonus"] * progress
        _track("sc5b_park_bonus")

        # 12c. v9.1 — Midfield exit penalty (PROBLEM 62).
        # Fires ONCE when a robot LEAVES the midfield during the park window.
        # Penalty = midfield_exit_penalty × progress²  (quadratic, always ≤ 0).
        #
        # Anti-exploit proof:
        #   At any progress p, the per-step park reward rate = 2.5 × p.
        #   An exit at progress p costs: |−40| × p² = 40p² reward units.
        #   Any "scoring run" outside midfield yields at most ~8-10 reward units.
        #   Since 40p² + (forgone step rewards during gap) >> 8-10 for all p≥0.25,
        #   leaving is NEVER net-positive once the robot is meaningfully into
        #   the window.  Near the window start (p≈0), both reward and penalty
        #   are near zero, so leaving early is not punished harshly (correct —
        #   there is still time to score before committing).
        #
        # _prev_midfield is updated EVERY step (below) so the state is correct
        # whether or not the park window is active.
        if self.sim.rules_engine.endgame_active and tr <= PARK_WINDOW_SECONDS:
            progress = max(0.0, (PARK_WINDOW_SECONDS - tr) / PARK_WINDOW_SECONDS)
            for r in self.sim.robots:
                was_in = self._prev_midfield.get(r.robot_id, False)
                is_in  = self.sim.rules_engine.is_robot_in_midfield(r)
                if was_in and not is_in:
                    # quadratic scale: harsher the later the exit
                    rewards[r.robot_id] += rw["midfield_exit_penalty"] * (progress ** 2)
        _track("midfield_exit")

        # Update _prev_midfield every step (outside the window gate) so the
        # transition from pre-window to in-window is captured correctly.
        for r in self.sim.robots:
            self._prev_midfield[r.robot_id] = (
                self.sim.rules_engine.is_robot_in_midfield(r)
            )

        # 13a. Spin penalty — penalise in-place spinning.
        # Fires when angular velocity is high AND translational speed is low,
        # distinguishing deliberate tight turns (ang_vel high, trans_speed moderate)
        # from pointless in-place spinning (ang_vel high, trans_speed ≈ 0).
        for rid in AGENT_IDS:
            r = robot_map[rid]
            ang_vel     = abs(float(r.body.angular_velocity))
            trans_speed = math.hypot(float(r.body.velocity.x),
                                     float(r.body.velocity.y))
            if ang_vel > SPIN_ANG_VEL_THRESHOLD and trans_speed < SPIN_TRANS_THRESHOLD:
                rewards[rid] += rw["spin_penalty"]
        _track("spin_penalty")

        # 13b. Toggle-camping penalty — penalise lingering near a toggle the
        # robot's own alliance already owns, unless inside the post-flip grace
        # window (v8: robots get TOGGLE_LEAVE_GRACE_STEPS steps to physically
        # leave after claiming a toggle before the penalty resumes).
        for rid in AGENT_IDS:
            r = robot_map[rid]
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            for t in self.sim.toggles:
                tid = getattr(t, "toggle_id", None)
                if self._toggle_grace.get(tid, 0) > 0:
                    continue  # grace window — don't penalise yet
                tx = float(t.body.position.x) if hasattr(t, "body") else t.x
                ty = float(t.body.position.y) if hasattr(t, "body") else t.y
                if (math.hypot(rx - tx, ry - ty) < TOGGLE_CAMP_RADIUS
                        and t.owner == r.alliance):
                    rewards[rid] += rw["toggle_camping"]
                    break   # only penalise once per robot per step
        _track("toggle_camping")

        # 13c. Forward-speed reward — reward the component of velocity directed
        # toward the robot's current target.  Empty-handed: nearest uncollected
        # element.  Carrying: nearest goal where scoring is legal.
        # This gives a dense, continuous incentive for fast purposeful movement
        # and complements the sparser approach-delta and proximity signals.
        for rid in AGENT_IDS:
            r   = robot_map[rid]
            rx  = float(r.body.position.x)
            ry  = float(r.body.position.y)
            vx  = float(r.body.velocity.x)
            vy  = float(r.body.velocity.y)
            spd = math.hypot(vx, vy)
            if spd < 1.0:
                continue   # stationary — skip to avoid division noise

            target_x = target_y = None
            best_d   = None

            if r.carrying_pin is not None or r.carrying_cup is not None:
                # Carrying: aim for nearest goal where a legal score exists.
                for g in self.sim.goals:
                    if g.alliance not in ("neutral", r.alliance):
                        continue
                    can_pin = r.carrying_pin is not None and not _stack_top_is_pin(g.stack)
                    can_cup = r.carrying_cup is not None and     _stack_top_is_pin(g.stack)
                    if not (can_pin or can_cup):
                        continue
                    d = math.hypot(rx - g.x, ry - g.y)
                    if best_d is None or d < best_d:
                        best_d, target_x, target_y = d, g.x, g.y
            else:
                # Empty-handed: aim for nearest uncollected element.
                for pin in self.sim.pins:
                    if pin.scored or pin.carried_by is not None:
                        continue
                    px = float(pin.body.position.x)
                    py = float(pin.body.position.y)
                    d  = math.hypot(rx - px, ry - py)
                    if best_d is None or d < best_d:
                        best_d, target_x, target_y = d, px, py
                for cup in self.sim.cups:
                    if cup.scored or cup.carried_by is not None:
                        continue
                    cx = float(cup.body.position.x)
                    cy = float(cup.body.position.y)
                    d  = math.hypot(rx - cx, ry - cy)
                    if best_d is None or d < best_d:
                        best_d, target_x, target_y = d, cx, cy

            if target_x is None or best_d < 1.0:
                continue   # no valid target or already at target

            # Unit vector toward target; dot with normalised velocity → [-1, 1]
            inv_d = 1.0 / best_d
            ux, uy = (target_x - rx) * inv_d, (target_y - ry) * inv_d
            dot = (vx * ux + vy * uy) / spd
            if dot > 0.0:
                rewards[rid] += rw["forward_speed_scale"] * dot
        _track("forward_speed")

        # 13d. v8.3 — Carrying speed bonus.
        # Direction-agnostic per-step reward proportional to raw speed magnitude
        # while carrying.  Complements forward_speed_scale (which requires
        # heading alignment with target) by creating a global "go fast with
        # the goods" incentive.  Safe: max 0.015/step; holding_timeout cap
        # (-1.8/step) dominates from step ~45 onward, preventing circling.
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is None and r.carrying_cup is None:
                continue
            spd = math.hypot(float(r.body.velocity.x), float(r.body.velocity.y))
            if spd < 1.0:
                continue
            spd_frac = min(1.0, spd / float(ROBOT_MAX_SPEED))
            rewards[rid] += rw["carrying_speed_scale"] * spd_frac
        _track("carrying_speed")

        # 13f. v8.1 — Defensive position bonus.
        # Reward empty-handed robots for positioning themselves on the line
        # between a carrying enemy and that enemy's nearest scorable goal.
        # Encourages defensive blocking play without requiring contact,
        # giving robots a positive (vs purely penalty-driven) reason to play
        # defense.
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is not None or r.carrying_cup is not None:
                continue   # only empty-handed robots play defense
            rx = float(r.body.position.x); ry = float(r.body.position.y)
            blocked = False
            for opp_id in AGENT_IDS:
                if blocked:
                    break
                opp = robot_map[opp_id]
                if opp.alliance == r.alliance:
                    continue
                if opp.carrying_pin is None and opp.carrying_cup is None:
                    continue
                ox = float(opp.body.position.x); oy = float(opp.body.position.y)
                # opp's nearest scorable goal
                best_g = None; best_d = None
                for g in self.sim.goals:
                    if g.alliance not in ("neutral", opp.alliance):
                        continue
                    can_pin = opp.carrying_pin is not None and not _stack_top_is_pin(g.stack)
                    can_cup = opp.carrying_cup is not None and     _stack_top_is_pin(g.stack)
                    if not (can_pin or can_cup):
                        continue
                    d = math.hypot(ox - g.x, oy - g.y)
                    if best_d is None or d < best_d:
                        best_d, best_g = d, g
                if best_g is None:
                    continue
                gx, gy = best_g.x, best_g.y
                dx, dy = gx - ox, gy - oy
                L2 = dx * dx + dy * dy
                if L2 < 1.0:
                    continue
                # Parameter t: 0 at opp, 1 at goal.  Only count if defender
                # is between them (0.1 ≤ t ≤ 0.9).
                t_param = ((rx - ox) * dx + (ry - oy) * dy) / L2
                if t_param < 0.1 or t_param > 0.9:
                    continue
                # Perpendicular distance from (rx,ry) to line opp→goal
                px_on_line = ox + t_param * dx
                py_on_line = oy + t_param * dy
                perp_d = math.hypot(rx - px_on_line, ry - py_on_line)
                if perp_d > DEFENSIVE_LINE_PERP_DIST:
                    continue
                rewards[rid] += rw["defensive_position_bonus"]
                blocked = True
        _track("defensive_position")

        # 14. RND Intrinsic Reward — inference only in workers (no backward pass).
        # Predictor training skipped here: 16 simultaneous backward+Adam steps
        # across workers exhaust VRAM. The fixed target network still drives a
        # meaningful novelty signal without per-step predictor updates.
        if self.rnd is not None:
            obs_cache = post_obs if post_obs is not None else build_all_observations(self.sim)
            for rid in AGENT_IDS:
                obs_t     = torch.FloatTensor(obs_cache[rid]).to(self.rnd.device)
                intrinsic = self.rnd.compute_intrinsic_reward(obs_t).item()
                rewards[rid] += intrinsic
            _track("rnd_intrinsic")

        # Diagnostic: fraction of robots currently over the holding timeout.
        # 0 = all robots scoring fast; 1 = every robot is timed out.
        # Reported as per-rollout average (same units as other signals after
        # the training loop divides by n_component_rollouts).
        n_over = sum(1 for cs in self._carry_steps.values()
                     if cs > HOLDING_TIMEOUT_STEPS)
        rc["frac_over_timeout"] = rc.get("frac_over_timeout", 0.0) + (
            n_over / float(len(AGENT_IDS))
        )

        return rewards

    # =========================================================================
    # DISTANCE HELPERS
    # =========================================================================
    def _empty_target_dist(self, rid: str, robot_map: dict) -> float:
        r = robot_map.get(rid)
        if r is None:
            return FIELD_DIAGONAL
        if r.carrying_pin is not None or r.carrying_cup is not None:
            return FIELD_DIAGONAL
        rx = float(r.body.position.x)
        ry = float(r.body.position.y)
        best = FIELD_DIAGONAL
        for pin in self.sim.pins:
            if pin.scored or pin.carried_by is not None:
                continue
            d = math.hypot(rx - float(pin.body.position.x),
                           ry - float(pin.body.position.y))
            if d < best: best = d
        for cup in self.sim.cups:
            if cup.scored or cup.carried_by is not None:
                continue
            d = math.hypot(rx - float(cup.body.position.x),
                           ry - float(cup.body.position.y))
            if d < best: best = d
        return best

    def _carrying_target_dist(self, rid: str, robot_map: dict) -> float:
        r = robot_map.get(rid)
        if r is None:
            return FIELD_DIAGONAL
        rx = float(r.body.position.x)
        ry = float(r.body.position.y)
        best = FIELD_DIAGONAL
        for g in self.sim.goals:
            if g.alliance not in ("neutral", r.alliance):
                continue
            d = math.hypot(rx - g.x, ry - g.y)
            if d < best: best = d
        return best

    def _toggle_is_legal(self, robot) -> bool:
        rx = float(robot.body.position.x)
        ry = float(robot.body.position.y)
        for t in self.sim.toggles:
            tx = float(t.body.position.x) if hasattr(t, "body") else t.x
            ty = float(t.body.position.y) if hasattr(t, "body") else t.y
            if math.hypot(rx - tx, ry - ty) < 30.0 and t.owner != robot.alliance:
                return True
        return False

    # =========================================================================
    # PUBLIC API
    # =========================================================================
    def drain_reward_components(self) -> Dict[str, float]:
        """Return the running per-component reward sums and reset the tracker.

        Sums are aggregated across all four robots and every step since the
        last drain.  Use the returned dict to populate per-signal TensorBoard
        (or text-log) metrics in the training loop.
        """
        out = dict(self._reward_components)
        self._reward_components.clear()
        return out

    def get_action_masks(self) -> Dict[str, np.ndarray]:
        return {
            r.robot_id: get_action_mask(r, self.sim.goals)
            for r in self.sim.robots
        }

    def render(self):
        self.sim.render()

    def close(self):
        if not self.headless:
            import pygame
            pygame.quit()

    @property
    def obs_dim(self):         return OBS_DIM
    @property
    def action_cont_dim(self): return 2
    @property
    def action_disc_dim(self): return 7


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================

def _zero_action():
    return {"left": 0.0, "right": 0.0, "intake": False, "score_pin": False,
            "score_cup": False, "toggle": False, "flip_pin": False, "flip_cup": False}


def _stack_top_is_pin(stack) -> bool:
    """True if the topmost element in a goal stack is a pin (not a cup).

    Centralises the ``bool(stack) and stack[-1][1]`` idiom used throughout
    the reward logic so that any future stack-format change only needs one fix.
    """
    return bool(stack) and bool(stack[-1][1])


def _eff_clear_up(cup) -> bool:
    flipped = getattr(cup, "flipped", False)
    return (not cup.clear_on_top) if flipped else cup.clear_on_top


def _get_toggle_for_goal(goal, toggles) -> Optional[str]:
    dx = goal.x - 72.0
    dy = goal.y - 72.0
    tid = (1 if dx <= 0 else 2) if abs(dx) >= abs(dy) else (3 if dy <= 0 else 4)
    for t in toggles:
        if t.toggle_id == tid:
            return t.owner if t.owner in ("red", "blue") else None
    return None


def _is_opponent_color(color, alliance: str) -> bool:
    return color == C_BLUE if alliance == "red" else color == C_RED


def _is_own_color(color, alliance: str) -> bool:
    return color == C_RED if alliance == "red" else color == C_BLUE