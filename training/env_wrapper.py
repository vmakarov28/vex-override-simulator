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

        # 3. Carrying proximity reward
        for rid in AGENT_IDS:
            r = robot_map[rid]
            if r.carrying_pin is not None or r.carrying_cup is not None:
                dist = self._carrying_target_dist(rid, robot_map)
                prox = rw["carrying_proximity_scale"] / (1.0 + dist / GOAL_PROXIMITY_NORM)
                rewards[rid] += prox

        # 4. Score attempt reward
        for rid in AGENT_IDS:
            if not score_attempted[rid]:
                continue
            r = robot_map[rid]
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            for g in self.sim.goals:
                if g.alliance not in ("neutral", r.alliance):
                    continue
                if math.hypot(rx - g.x, ry - g.y) <= SCORING_RADIUS + 4.0:
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
                for col in [new_obj.get_up_color(), new_obj.get_down_color()]:
                    if col == C_RED:
                        r_val = rw["score_own_pin"] if scorer_alliance == "red" else rw["score_opp_half"]
                        b_val = rw["score_opp_half"] if scorer_alliance == "red" else rw["score_own_pin"]
                    elif col == C_BLUE:
                        r_val = rw["score_opp_half"] if scorer_alliance == "red" else rw["score_own_pin"]
                        b_val = rw["score_own_pin"] if scorer_alliance == "blue" else rw["score_opp_half"]
                    elif col == C_YELLOW:
                        towner = _get_toggle_for_goal(goal, list(self.sim.toggles))
                        if towner == scorer_alliance:
                            r_val = rw["score_yellow_owned"] if scorer_alliance == "red" else 0.0
                            b_val = rw["score_yellow_owned"] if scorer_alliance == "blue" else 0.0
                        else:
                            r_val = rw["score_yellow_neutral"] if scorer_alliance == "red" else 0.0
                            b_val = rw["score_yellow_neutral"] if scorer_alliance == "blue" else 0.0
                    else:
                        r_val = b_val = 0.0
                    for rid in ["red1", "red2"]: rewards[rid] += r_val
                    for rid in ["blue1", "blue2"]: rewards[rid] += b_val
            else:
                eff_clear = _eff_clear_up(new_obj)
                cup_idx = len(pre_stack)
                if cup_idx > 0:
                    below_obj, below_is_pin = post_stack[cup_idx - 1]
                    if below_is_pin:
                        blocked_col = below_obj.get_up_color()
                        if not eff_clear:
                            if _is_opponent_color(blocked_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_success"]
                            elif _is_own_color(blocked_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_own"]
                        else:
                            if _is_opponent_color(blocked_col, scorer_alliance):
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