"""
environment/override_env.py

PettingZoo Multi-Agent Environment for Override
===============================================

This wraps the OverrideSimulator into a PettingZoo ParallelEnv so that
four neural network agents can play against each other cleanly.

Key features:
- 4 agents: red1, red2, blue1, blue2
- Full match lifecycle (auto → driver → settle)
- Rich observations (positions, velocities, carried objects, toggle states, scores, time)
- Continuous or discrete action space per robot
- Proper reward shaping that encourages legal, strategic play
- Fouls are penalized

This is the bridge between the physics/rules simulation and modern RL libraries
(Stable-Baselines3, CleanRL, RLlib, etc.).
"""

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
import numpy as np
from typing import Dict, Any

from simulation.simulator import OverrideSimulator
from config.hyperparameters import TRAINING_CONFIG


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
        self.agents = ["red1", "red2", "blue1", "blue2"]
        self.possible_agents = self.agents[:]

        # Action and observation spaces (will be refined)
        self.action_spaces = {agent: self._get_action_space() for agent in self.agents}
        self.observation_spaces = {agent: self._get_observation_space() for agent in self.agents}

    def _get_action_space(self):
        # Continuous drive + discrete intake/score buttons
        from gymnasium import spaces
        # [left, right, intake (0/1), score (0/1)]
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _get_observation_space(self):
        from gymnasium import spaces
        # Rich observation: ~40 values per agent for now
        return spaces.Box(low=-1.0, high=1.0, shape=(48,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.sim.reset()
        self.agents = self.possible_agents[:]
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        try:
            action_list = []
            for agent in ["red1", "red2", "blue1", "blue2"]:
                if agent in actions:
                    act = actions[agent]
                    action_list.append({
                        "left": float(act[0]),
                        "right": float(act[1]),
                        "intake": bool(act[2] > 0.5),
                        "score": bool(act[3] > 0.5),
                        "toggle": bool(act[3] > 0.7)
                    })
                else:
                    action_list.append(None)

            dt = 1.0 / 60.0
            self.sim.step(dt, action_list)

            observations = self._get_observations()
            rewards = self._get_rewards()
            terminations = {agent: self.sim.match_over for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
            infos = {agent: {"phase": self.sim.match_phase} for agent in self.agents}

            if self.sim.match_over:
                self.agents = []

            return observations, rewards, terminations, truncations, infos

        except Exception as e:
            print(f"[OverrideEnv] Step error: {e}")
            observations = self._get_observations()
            rewards = {agent: 0.0 for agent in self.agents}
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: True for agent in self.agents}
            infos = {agent: {} for agent in self.agents}
            return observations, rewards, terminations, truncations, infos

    def _get_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        for i, agent in enumerate(["red1", "red2", "blue1", "blue2"]):
            robot = self.sim.robots[i] if i < len(self.sim.robots) else None
            if robot:
                state = robot.get_state()
                # Normalized simple observation
                o = np.array([
                    state["position"].x / FIELD_WIDTH,
                    state["position"].y / FIELD_HEIGHT,
                    state["velocity"].x / 100,
                    state["velocity"].y / 100,
                    np.sin(state["angle"]),
                    np.cos(state["angle"]),
                    1.0 if state["carrying"]["pin"] else 0.0,
                    1.0 if state["carrying"]["cup"] else 0.0,
                    self.sim.time_elapsed / 120.0,
                    1.0 if self.sim.match_phase == "autonomous" else 0.0,
                    # Mechanism config (one-hot style)
                    1.0 if state.get("mechanism_config") == "front_intake_back_score" else 0.0,
                ], dtype=np.float32)
                # Pad to fixed size
                obs[agent] = np.pad(o, (0, 48 - len(o)))[:48]
            else:
                obs[agent] = np.zeros(48, dtype=np.float32)
        return obs

    def _get_rewards(self) -> Dict[str, float]:
        """
        Full layered reward system for Override:
        - Score Delta (primary)
        - Dense shaping (possession, toggles, approach)
        - Event rewards
        - Terminal rewards + margin bonus
        - Anti-degenerate behavior encouraged via penalties in rules_engine
        """
        rewards = {}
        for i, agent_name in enumerate(["red1", "red2", "blue1", "blue2"]):
            alliance = "red" if "red" in agent_name else "blue"
            robot = self.sim.robots[i]

            my_score = self.sim.red_score if alliance == "red" else self.sim.blue_score
            opp_score = self.sim.blue_score if alliance == "red" else self.sim.red_score

            reward = 0.0

            # LAYER 1: Score Delta
            prev_my = getattr(self, f'prev_{alliance}_score', my_score)
            prev_opp = getattr(self, f'prev_opp_score', opp_score)
            reward += (my_score - prev_my) * 1.1
            reward -= (opp_score - prev_opp) * 1.0

            # LAYER 2: Dense Shaping
            held_value = 0.0
            if robot.carrying_pin and robot.carrying_cup:
                held_value = 2.3
            elif robot.carrying_cup:
                held_value = 1.7
            elif robot.carrying_pin:
                held_value = 1.1
            reward += held_value * 0.16

            # Toggle dominance
            my_toggles = sum(1 for t in getattr(self.sim, 'toggles', []) if getattr(t, 'owner', None) == alliance)
            opp_toggles = sum(1 for t in getattr(self.sim, 'toggles', []) if getattr(t, 'owner', None) and getattr(t, 'owner') != alliance)
            reward += (my_toggles - opp_toggles) * 0.25

            # LAYER 3: Event
            if hasattr(robot, 'successful_scores'):
                prev_scores = getattr(robot, 'prev_successful_scores', 0)
                if robot.successful_scores > prev_scores:
                    reward += 5.8
                robot.prev_successful_scores = robot.successful_scores

            # LAYER 4: Terminal
            if self.sim.match_over:
                if my_score > opp_score:
                    reward += 30
                elif my_score < opp_score:
                    reward -= 27
                margin = abs(my_score - opp_score)
                reward += min(margin * 0.11, 11) * (1 if my_score > opp_score else -1)

            # Store for next step
            setattr(self, f'prev_{alliance}_score', my_score)
            setattr(self, 'prev_opp_score', opp_score)

            rewards[agent_name] = float(np.clip(reward, -2.0, 2.0))

        return rewards

    def render(self):
        if not self.headless:
            self.sim.render()

    def close(self):
        pass


def make_override_env(render_mode=None):
    """Factory function to create a wrapped Override environment."""
    env = OverrideEnv(render_mode=render_mode)
    # You can add wrappers here (e.g. for normalization)
    return env