"""
training/env_wrapper.py  (v8)
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
    ALLY_SEPARATION_TARGET, TIME_TO_SCORE_TARGET,
    ENDGAME_RAMP_SECONDS, ENDGAME_RAMP_MAX_MULT,
    PROX_CARRY_DECAY_STEPS, TOGGLE_LEAVE_GRACE_STEPS,
    DEFENSIVE_LINE_PERP_DIST,
)
from config.game_rules import (
    SCORING_RADIUS, ENDGAME_SECONDS, TOTAL_SECONDS,
    AUTONOMOUS_SECONDS, MIDFIELD_HALF, ROBOT_STARTS,
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

        # v7: per-component reward sum tracker (summed across all robots and
        # all steps since last drain).  drain_reward_components() returns the
        # current dict and resets it — used by training loops for TB-style
        # per-signal logging.
        self._reward_components: Dict[str, float]   = {}

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
    # REWARD COMPUTATION (v7)
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

        # 2. Intake / drop  (+ v7 time-to-score bonus, v8.1 resource denial)
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

            scored = r.successful_scores > self._prev_scores.get(rid, 0)
            if (had_pin and not now_pin and not scored) or \
               (had_cup and not now_cup and not scored):
                rewards[rid] += rw["drop_penalty"]

            if scored:
                # v8.1 FIX: use PRE-step carry counter — post-update value
                # is 0 because the score caused carrying_pin/cup to become None.
                carry_steps = pre_carry_steps.get(rid, 0)
                ratio = max(0.0, 1.0 - carry_steps / float(TIME_TO_SCORE_TARGET))
                rewards[rid] += rw["time_to_score_bonus"] * ratio

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

        # 3e. v7 — Ally separation bonus.
        # Small per-step reward when the two alliance robots are at least
        # ALLY_SEPARATION_TARGET inches apart.  Pulls them out of "share one
        # element" loops without explicitly assigning roles.
        for alliance in ("red", "blue"):
            allies = [robot_map[r] for r in AGENT_IDS if robot_map[r].alliance == alliance]
            if len(allies) < 2:
                continue
            a, b = allies[0], allies[1]
            d = math.hypot(float(a.body.position.x) - float(b.body.position.x),
                           float(a.body.position.y) - float(b.body.position.y))
            if d >= ALLY_SEPARATION_TARGET:
                rewards[a.robot_id] += rw["ally_separation_bonus"]
                rewards[b.robot_id] += rw["ally_separation_bonus"]
        _track("ally_separation")

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
        # Quadratic means each extra HOLDING_RAMP_STEPS of overshoot squares the cost,
        # making prolonged carrying catastrophic rather than merely annoying.
        for rid in AGENT_IDS:
            cs = self._carry_steps[rid]
            if cs > HOLDING_TIMEOUT_STEPS:
                overshoot = cs - HOLDING_TIMEOUT_STEPS
                ratio     = (overshoot / HOLDING_RAMP_STEPS) ** 2
                rewards[rid] += rw["holding_penalty_rate"] * ratio
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
                        # Yellow reward belongs to whichever alliance owns the toggle,
                        # regardless of who placed the pin.  The opposing alliance
                        # effectively gifted points, so they receive score_opp_half.
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
                        else:
                            # Clear bottom faces pin — pin's UP half stays VISIBLE.
                            if _is_opponent_color(pin_up_col, scorer_alliance):
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

        # 12. Midfield endgame bonus  (v7: ramping multiplier)
        # Base rate `midfield_endgame` is paid every step in endgame.  In the
        # final ENDGAME_RAMP_SECONDS, multiply by a linear ramp 1 → ENDGAME_RAMP_MAX_MULT
        # so robots only get the BIG reward by parking very late.  Earlier parking
        # still earns the base rate (so it isn't punished), but the marginal
        # gain from staying until the buzzer is much larger.
        if self.sim.rules_engine.endgame_active:
            tr = float(self.sim.time_remaining)
            if tr <= ENDGAME_RAMP_SECONDS:
                # Linear ramp: tr=ENDGAME_RAMP_SECONDS → mult=1
                #              tr=0                   → mult=ENDGAME_RAMP_MAX_MULT
                ramp = 1.0 + (ENDGAME_RAMP_MAX_MULT - 1.0) * (
                    (ENDGAME_RAMP_SECONDS - tr) / ENDGAME_RAMP_SECONDS)
            else:
                ramp = 1.0
            mc_x = mc_y = 72.0
            for r in self.sim.robots:
                rx = float(r.body.position.x)
                ry = float(r.body.position.y)
                if abs(rx - mc_x) + abs(ry - mc_y) <= MIDFIELD_HALF + 10.0:
                    rewards[r.robot_id] += rw["midfield_endgame"] * ramp
        _track("midfield_endgame")

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

        # 13d. v8.1 — Defensive position bonus.
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

        # 14. RND Intrinsic Reward — use pre-built post_obs to avoid redundant builds
        if self.rnd is not None:
            # Update predictor once per update cadence (not once per agent)
            do_update = self.rnd.should_update(self._step_count)
            obs_cache = post_obs if post_obs is not None else build_all_observations(self.sim)
            for rid in AGENT_IDS:
                obs_t     = torch.FloatTensor(obs_cache[rid]).to(self.rnd.device)
                intrinsic = self.rnd.compute_intrinsic_reward(obs_t).item()
                rewards[rid] += intrinsic
                if do_update:
                    self.rnd.update(obs_t.unsqueeze(0))
            _track("rnd_intrinsic")

        # Diagnostic: average carry steps across robots (tracks cycle speed improvement)
        rc["avg_carry_steps"] = rc.get("avg_carry_steps", 0.0) + (
            sum(self._carry_steps.values()) / len(AGENT_IDS)
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
            r.robot_id: get_action_mask(r, self.sim.goals, self.sim.rules_engine)
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