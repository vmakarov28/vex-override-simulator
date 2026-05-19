"""
environment/override_env.py

PettingZoo Multi-Agent Environment for Override
===============================================

This wraps the OverrideSimulator into a PettingZoo ParallelEnv so that
four neural network agents can play against each other cleanly.

Key features:
- 4 agents: red1, red2, blue1, blue2
- Full match lifecycle (auto → driver → settle)
- 551-dim observations built by observation_builder (matches training env)
- 9-dim action vector: [left, right, intake, score_pin, score_cup, toggle,
                         flip_pin, flip_cup, match_load]
- Reward shaping aligned with training/env_wrapper.py v5 (legality-gated
  proximity, fetch-needed redirect, wrong-element loiter penalty)
"""

import math
import numpy as np
from typing import Dict, Any

from pettingzoo import ParallelEnv
from gymnasium import spaces

from simulation.simulator import OverrideSimulator
from utils.observation_builder import build_all_observations, OBS_DIM
from config.hyperparameters import (
    REWARD_WEIGHTS, GOAL_PROXIMITY_NORM,
    HOLDING_TIMEOUT_STEPS, HOLDING_RAMP_STEPS,
    IDLE_SPEED_THRESHOLD, START_ZONE_RADIUS,
    PINNING_STEPS_LIMIT, PINNING_CONTACT_DIST,
    FIELD_DIAGONAL,
)
from config.game_rules import SCORING_RADIUS, MIDFIELD_CENTER, MIDFIELD_HALF

AGENT_IDS = ["red1", "red2", "blue1", "blue2"]


class OverrideEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "override_v0"
    }

    def __init__(self, render_mode=None, headless=True):
        super().__init__()
        self.render_mode = render_mode
        self.headless = headless

        self.sim = OverrideSimulator(headless=headless)
        self.agents = list(AGENT_IDS)
        self.possible_agents = list(AGENT_IDS)

        self.action_spaces      = {a: self._get_action_space()      for a in self.agents}
        self.observation_spaces = {a: self._get_observation_space() for a in self.agents}

        # Per-agent state for reward computation
        self._prev_red_score  = 0
        self._prev_blue_score = 0
        self._carry_steps: Dict[str, int] = {rid: 0 for rid in AGENT_IDS}
        # Per-robot, per-opponent contact step tracking  (key = frozenset of two rids)
        self._contact_steps: Dict[frozenset, int] = {
            frozenset({a, b}): 0
            for i, a in enumerate(AGENT_IDS) for b in AGENT_IDS[i+1:]
        }
        self._start_positions: Dict[str, tuple] = {}
        self._prev_carrying_pin  = {rid: None for rid in AGENT_IDS}
        self._prev_carrying_cup  = {rid: None for rid in AGENT_IDS}
        self._prev_scores_count  = {rid: 0    for rid in AGENT_IDS}

    # -------------------------------------------------------------------------
    def _get_action_space(self):
        # [left, right, intake, score_pin, score_cup, toggle, flip_pin, flip_cup, match_load]
        return spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    def _get_observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    # -------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.sim.reset()
        self.agents = list(AGENT_IDS)

        self._prev_red_score  = 0
        self._prev_blue_score = 0
        self._carry_steps     = {rid: 0 for rid in AGENT_IDS}
        self._contact_steps   = {frozenset({a, b}): 0
                                 for i, a in enumerate(AGENT_IDS)
                                 for b in AGENT_IDS[i+1:]}
        self._prev_carrying_pin = {rid: None for rid in AGENT_IDS}
        self._prev_carrying_cup = {rid: None for rid in AGENT_IDS}
        self._prev_scores_count = {rid: 0    for rid in AGENT_IDS}

        robot_map = {r.robot_id: r for r in self.sim.robots}
        self._start_positions = {
            rid: (float(robot_map[rid].body.position.x),
                  float(robot_map[rid].body.position.y))
            for rid in AGENT_IDS if rid in robot_map
        }

        self.sim.timer_started = True
        observations = build_all_observations(self.sim)
        infos = {a: {} for a in self.agents}
        return observations, infos

    # -------------------------------------------------------------------------
    def step(self, actions: Dict[str, np.ndarray]):
        try:
            robot_map = {r.robot_id: r for r in self.sim.robots}
            pre_red   = self.sim.rules_engine.red_score
            pre_blue  = self.sim.rules_engine.blue_score
            pre_carrying_pin = {r.robot_id: r.carrying_pin for r in self.sim.robots}
            pre_carrying_cup = {r.robot_id: r.carrying_cup for r in self.sim.robots}

            # Build simulator action list
            sim_actions = []
            flips_fired     = {rid: False for rid in AGENT_IDS}
            score_attempted = {rid: False for rid in AGENT_IDS}

            for rid in AGENT_IDS:
                if rid in actions:
                    act = np.asarray(actions[rid], dtype=np.float32)
                    left       = float(np.clip(act[0], -1.0, 1.0))
                    right      = float(np.clip(act[1], -1.0, 1.0))
                    intake     = bool(act[2] > 0.5)
                    score_pin  = bool(act[3] > 0.5)
                    score_cup  = bool(act[4] > 0.5)
                    toggle     = bool(act[5] > 0.5)
                    flip_pin   = bool(act[6] > 0.5)
                    flip_cup   = bool(act[7] > 0.5)
                    # match_load (act[8]) is not passed directly; handled via sim API
                    if flip_pin or flip_cup:
                        flips_fired[rid] = True
                    if score_pin or score_cup:
                        score_attempted[rid] = True
                    sim_actions.append({
                        "left":      left,
                        "right":     right,
                        "intake":    intake,
                        "score_pin": score_pin,
                        "score_cup": score_cup,
                        "toggle":    toggle,
                        "flip_pin":  flip_pin,
                        "flip_cup":  flip_cup,
                    })
                else:
                    sim_actions.append({
                        "left": 0.0, "right": 0.0,
                        "intake": False, "score_pin": False, "score_cup": False,
                        "toggle": False, "flip_pin": False, "flip_cup": False,
                    })

            dt = 1.0 / 20.0
            self.sim.step(dt, sim_actions)

            robot_map = {r.robot_id: r for r in self.sim.robots}
            post_red  = self.sim.rules_engine.red_score
            post_blue = self.sim.rules_engine.blue_score

            # Update carry steps
            for rid in AGENT_IDS:
                r = robot_map.get(rid)
                if r and (r.carrying_pin is not None or r.carrying_cup is not None):
                    self._carry_steps[rid] += 1
                else:
                    self._carry_steps[rid] = 0

            # Update contact steps
            for key in self._contact_steps:
                a_id, b_id = tuple(key)
                ra = robot_map.get(a_id)
                rb = robot_map.get(b_id)
                if ra and rb:
                    d = math.hypot(
                        float(ra.body.position.x) - float(rb.body.position.x),
                        float(ra.body.position.y) - float(rb.body.position.y),
                    )
                    if d < PINNING_CONTACT_DIST:
                        self._contact_steps[key] += 1
                    else:
                        self._contact_steps[key] = 0

            observations = build_all_observations(self.sim)
            rewards = self._compute_rewards(
                pre_red, pre_blue, post_red, post_blue,
                robot_map, pre_carrying_pin, pre_carrying_cup,
                flips_fired, score_attempted,
            )

            self._prev_red_score  = post_red
            self._prev_blue_score = post_blue
            for r in self.sim.robots:
                self._prev_carrying_pin[r.robot_id] = r.carrying_pin
                self._prev_carrying_cup[r.robot_id] = r.carrying_cup
                self._prev_scores_count[r.robot_id] = r.successful_scores

            done = self.sim.match_over
            terminations = {a: done for a in self.agents}
            truncations  = {a: False for a in self.agents}
            infos = {a: {"phase": self.sim.match_phase,
                         "red_score": post_red, "blue_score": post_blue}
                     for a in self.agents}

            if done:
                self.agents = []

            return observations, rewards, terminations, truncations, infos

        except Exception as e:
            print(f"[OverrideEnv] Step error: {e}")
            import traceback; traceback.print_exc()
            observations = {a: np.zeros(OBS_DIM, dtype=np.float32) for a in self.agents}
            rewards      = {a: 0.0  for a in self.agents}
            terminations = {a: True for a in self.agents}
            truncations  = {a: True for a in self.agents}
            infos        = {a: {}   for a in self.agents}
            self.agents  = []
            return observations, rewards, terminations, truncations, infos

    # -------------------------------------------------------------------------
    def _compute_rewards(
        self,
        pre_red, pre_blue, post_red, post_blue,
        robot_map,
        pre_carrying_pin, pre_carrying_cup,
        flips_fired, score_attempted,
    ) -> Dict[str, float]:
        rw = REWARD_WEIGHTS
        rewards = {rid: 0.0 for rid in AGENT_IDS}

        # 1. Score delta
        red_delta  = post_red  - pre_red
        blue_delta = post_blue - pre_blue
        for rid in ["red1",  "red2"]:  rewards[rid] += rw["score_delta"] * (red_delta  - blue_delta)
        for rid in ["blue1", "blue2"]: rewards[rid] += rw["score_delta"] * (blue_delta - red_delta)

        # 2. Intake / drop
        for rid in AGENT_IDS:
            r       = robot_map.get(rid)
            if not r:
                continue
            had_pin = pre_carrying_pin.get(rid) is not None
            had_cup = pre_carrying_cup.get(rid) is not None
            now_pin = r.carrying_pin is not None
            now_cup = r.carrying_cup is not None
            if (now_pin and not had_pin) or (now_cup and not had_cup):
                rewards[rid] += rw["intake_success"]
            prev_sc = self._prev_scores_count.get(rid, 0)
            scored  = r.successful_scores > prev_sc
            if (had_pin and not now_pin and not scored) or \
               (had_cup and not now_cup and not scored):
                rewards[rid] += rw["drop_penalty"]

        # 3. Carrying proximity — only fires when a legal score exists at some goal.
        for rid in AGENT_IDS:
            r = robot_map.get(rid)
            if not r:
                continue
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
                rewards[rid] += rw["carrying_proximity_scale"] / (1.0 + best / GOAL_PROXIMITY_NORM)

        # 3b. Fetch-needed redirect: approach missing element when unable to score anywhere.
        for rid in AGENT_IDS:
            r = robot_map.get(rid)
            if not r:
                continue
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            can_score_anywhere = any(
                g.alliance in ("neutral", r.alliance) and (
                    (r.carrying_pin is not None and (not g.stack or not g.stack[-1][1])) or
                    (r.carrying_cup is not None and bool(g.stack) and g.stack[-1][1])
                )
                for g in self.sim.goals
            )
            if can_score_anywhere:
                continue
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

        # 3c. Wrong-element loiter penalty.
        for rid in AGENT_IDS:
            r = robot_map.get(rid)
            if not r:
                continue
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

        # 4. Score attempt reward — only for legal attempts.
        for rid in AGENT_IDS:
            if not score_attempted[rid]:
                continue
            r = robot_map.get(rid)
            if not r:
                continue
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

        # 5. Holding timeout penalty.
        for rid in AGENT_IDS:
            cs = self._carry_steps[rid]
            if cs > HOLDING_TIMEOUT_STEPS:
                overshoot = cs - HOLDING_TIMEOUT_STEPS
                rewards[rid] += rw["holding_penalty_rate"] * (overshoot / HOLDING_RAMP_STEPS)

        # 6. Flip penalty.
        for rid in AGENT_IDS:
            if flips_fired[rid]:
                rewards[rid] += rw["flip_penalty"]

        # 7. Idle + start-zone penalty.
        for rid in AGENT_IDS:
            r = robot_map.get(rid)
            if not r:
                continue
            if r.carrying_pin is not None or r.carrying_cup is not None:
                continue
            speed = math.hypot(float(r.body.velocity.x), float(r.body.velocity.y))
            if speed < IDLE_SPEED_THRESHOLD:
                rewards[rid] += rw["idle_penalty"]
            sx, sy = self._start_positions.get(rid, (0.0, 0.0))
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            if math.hypot(rx - sx, ry - sy) < START_ZONE_RADIUS:
                rewards[rid] += rw["start_zone_penalty"]

        # 8. Pinning violation.
        for key in self._contact_steps:
            if self._contact_steps[key] <= PINNING_STEPS_LIMIT:
                continue
            a_id, b_id = tuple(key)
            ra = robot_map.get(a_id)
            rb = robot_map.get(b_id)
            if not ra or not rb or ra.alliance == rb.alliance:
                continue
            sp_a = math.hypot(float(ra.body.velocity.x), float(ra.body.velocity.y))
            sp_b = math.hypot(float(rb.body.velocity.x), float(rb.body.velocity.y))
            pinner = a_id if sp_a >= sp_b else b_id
            rewards[pinner] += rw["pinning_violation"]

        # 9. Terminal win/loss.
        if self.sim.match_over:
            diff = post_red - post_blue
            wt   = rw["win_terminal"]
            if diff > 0:
                for rid in ["red1",  "red2"]:  rewards[rid] += wt * (diff / 80.0)
                for rid in ["blue1", "blue2"]: rewards[rid] -= wt * (diff / 80.0)
            elif diff < 0:
                for rid in ["blue1", "blue2"]: rewards[rid] += wt * (-diff / 80.0)
                for rid in ["red1",  "red2"]:  rewards[rid] -= wt * (-diff / 80.0)

        # 10. Midfield endgame bonus.
        if self.sim.rules_engine.endgame_active:
            mc_x, mc_y = MIDFIELD_CENTER
            for r in self.sim.robots:
                rx = float(r.body.position.x)
                ry = float(r.body.position.y)
                if abs(rx - mc_x) + abs(ry - mc_y) <= MIDFIELD_HALF + 10.0:
                    rewards[r.robot_id] += rw["midfield_endgame"]

        return rewards

    # -------------------------------------------------------------------------
    def render(self):
        if not self.headless:
            self.sim.render()

    def close(self):
        pass


def make_override_env(render_mode=None):
    """Factory function to create a wrapped Override environment."""
    return OverrideEnv(render_mode=render_mode)
