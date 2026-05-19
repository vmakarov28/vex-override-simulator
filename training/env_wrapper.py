"""
training/env_wrapper.py  (v4 - Upgraded)
==========================================================================
Major improvements:
  - Holding penalty now starts AFTER 60 steps (~3 seconds)
  - Then ramps up gradually to discourage stalling
  - Stronger anti-reward-hacking measures
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
        self._contact_steps      = {p: 0 for p in ROBOT_PAIRS}
        self._prev_scores        = {rid: 0 for rid in AGENT_IDS}

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
    # REWARD COMPUTATION (v4 - Upgraded)
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
        post_obs: Dict[str, np.ndarray] = None,
    ) -> Dict[str, float]:
        rw      = REWARD_WEIGHTS
        rewards = {rid: 0.0 for rid in AGENT_IDS}

        # 1. Score delta
        red_delta  = post_red  - pre_red
        blue_delta = post_blue - pre_blue
        for rid in ["red1",  "red2"]:  rewards[rid] += rw["score_delta"] * (red_delta  - blue_delta)
        for rid in ["blue1", "blue2"]: rewards[rid] += rw["score_delta"] * (blue_delta - red_delta)

        # 2. Intake / drop
        for rid in AGENT_IDS:
            r       = robot_map[rid]
            had_pin = pre_carrying_pin[rid] is not None
            had_cup = pre_carrying_cup[rid] is not None
            now_pin = r.carrying_pin is not None
            now_cup = r.carrying_cup is not None

            if (now_pin and not had_pin) or (now_cup and not had_cup):
                rewards[rid] += rw["intake_success"]

            scored = r.successful_scores > self._prev_scores.get(rid, 0)
            if (had_pin and not now_pin and not scored) or \
               (had_cup and not now_cup and not scored):
                rewards[rid] += rw["drop_penalty"]

        for r in self.sim.robots:
            self._prev_scores[r.robot_id] = r.successful_scores

        # 3. Carrying proximity reward — only when a legal score exists at a nearby goal.
        # A robot holding the wrong element for every reachable goal earns nothing here,
        # preventing the "park at goal unable to score" exploit.
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is None and r.carrying_cup is None:
                continue
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            best = None
            for g in self.sim.goals:
                if g.alliance not in ("neutral", r.alliance):
                    continue
                can_pin = (r.carrying_pin is not None and
                           (not g.stack or not g.stack[-1][1]))
                can_cup = (r.carrying_cup is not None and
                           bool(g.stack) and g.stack[-1][1])
                if can_pin or can_cup:
                    d = math.hypot(rx - g.x, ry - g.y)
                    if best is None or d < best:
                        best = d
            if best is not None:
                prox = rw["carrying_proximity_scale"] / (1.0 + best / GOAL_PROXIMITY_NORM)
                rewards[rid] += prox

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
                    (r.carrying_pin is not None and (not g.stack or not g.stack[-1][1])) or
                    (r.carrying_cup is not None and bool(g.stack) and g.stack[-1][1])
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
                can_pin = (r.carrying_pin is not None and
                           (not g.stack or not g.stack[-1][1]))
                can_cup = (r.carrying_cup is not None and
                           bool(g.stack) and g.stack[-1][1])
                if not can_pin and not can_cup:
                    rewards[rid] += rw["wrong_element_loiter"]
                    break

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
                can_pin = (r.carrying_pin is not None and
                           (not g.stack or not g.stack[-1][1]))
                can_cup = (r.carrying_cup is not None and
                           bool(g.stack) and g.stack[-1][1])
                if can_pin or can_cup:
                    rewards[rid] += rw["score_attempt_in_zone"]
                    break

        # 5. Holding timeout penalty — ramps up after HOLDING_TIMEOUT_STEPS
        for rid in AGENT_IDS:
            cs = self._carry_steps[rid]
            if cs > HOLDING_TIMEOUT_STEPS:
                overshoot = cs - HOLDING_TIMEOUT_STEPS
                penalty   = rw["holding_penalty_rate"] * (overshoot / HOLDING_RAMP_STEPS)
                rewards[rid] += penalty

        # 6. Empty-hand approach delta
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is None and r.carrying_cup is None:
                delta_d = pre_empty_dist[rid] - post_empty_dist[rid]
                if delta_d > 0:
                    rewards[rid] += rw["approach_scale"] * delta_d / FIELD_DIAGONAL

        # 7. Flip penalty (neutral — no reward or punishment)
        for rid in AGENT_IDS:
            if flips_fired[rid]:
                rewards[rid] += rw["flip_penalty"]

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

        # 10. Goal-level causal scoring events
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
                    for rid in ["red1", "red2"]: rewards[rid] += r_val
                    for rid in ["blue1", "blue2"]: rewards[rid] += b_val
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
                                for rid in ally_rids: rewards[rid] += rw["denial_success"]
                            elif _is_own_color(pin_up_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_own"]
                        else:
                            # Clear bottom faces pin — pin's UP half stays VISIBLE.
                            if _is_opponent_color(pin_up_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_preserved_opp"]
                if cup_idx >= 1:
                    for rid in ally_rids: rewards[rid] += rw["stack_bonus"]

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

        # 12. Midfield endgame bonus
        if self.sim.rules_engine.endgame_active:
            mc_x = mc_y = 72.0
            for r in self.sim.robots:
                rx = float(r.body.position.x)
                ry = float(r.body.position.y)
                if abs(rx - mc_x) + abs(ry - mc_y) <= MIDFIELD_HALF + 10.0:
                    rewards[r.robot_id] += rw["midfield_endgame"]

        # 13. RND Intrinsic Reward — use pre-built post_obs to avoid redundant builds
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