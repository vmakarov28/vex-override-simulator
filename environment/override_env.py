"""
environment/override_env.py

PettingZoo Multi-Agent Environment for Override
===============================================

Reward parity with training/env_wrapper.py v8 (core sections):
- Legality-gated proximity reward with PROX_CARRY_DECAY_STEPS cut-off (3)
- Fetch-needed redirect (3b)
- Wrong-element loiter penalty (3c)
- Quadratic + capped holding-timeout (5)
- Goal-level causal scoring: corrected cup denial direction, pin DOWN-half
  visibility check, toggle-aware yellow pin reward (10)
- Toggle gain/loss events (11)
- Pinning violation via deterministic sorted-tuple contact tracking (8)

Note: v8-only rewards (resource_denial_bonus, defensive_position_bonus,
endgame_score_multiplier, toggle grace window, ally separation, teammate
overlap) are implemented in training/env_wrapper.py and are NOT replicated
here — the PettingZoo wrapper is used for external evaluation only.

9-dim action: [left, right, intake, score_pin, score_cup, toggle,
               flip_pin, flip_cup, match_load]
588-dim observation built by utils/observation_builder.
"""

import math
import itertools
import numpy as np
from typing import Dict, Optional

from pettingzoo import ParallelEnv
from gymnasium import spaces

from simulation.simulator import OverrideSimulator
from simulation.game_objects import C_RED, C_BLUE, C_YELLOW
from utils.observation_builder import build_all_observations, OBS_DIM
from config.hyperparameters import (
    REWARD_WEIGHTS, GOAL_PROXIMITY_NORM,
    HOLDING_TIMEOUT_STEPS, HOLDING_RAMP_STEPS, HOLDING_RAMP_SQ_CAP,
    IDLE_SPEED_THRESHOLD, START_ZONE_RADIUS,
    PINNING_STEPS_LIMIT, PINNING_CONTACT_DIST,
    FIELD_DIAGONAL, PROX_CARRY_DECAY_STEPS,
    PARK_WINDOW_SECONDS,
)
from config.game_rules import SCORING_RADIUS, MIDFIELD_CENTER, MIDFIELD_HALF

AGENT_IDS   = ["red1", "red2", "blue1", "blue2"]
ROBOT_PAIRS = list(itertools.combinations(AGENT_IDS, 2))  # 6 deterministic sorted pairs


class OverrideEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "override_v0"}

    def __init__(self, render_mode=None, headless=True):
        super().__init__()
        self.render_mode = render_mode
        self.headless    = headless

        self.sim             = OverrideSimulator(headless=headless)
        self.agents          = list(AGENT_IDS)
        self.possible_agents = list(AGENT_IDS)

        self.action_spaces      = {a: self._get_action_space()      for a in self.agents}
        self.observation_spaces = {a: self._get_observation_space() for a in self.agents}

        # Carry / contact counters
        self._carry_steps:   Dict[str, int]   = {rid: 0 for rid in AGENT_IDS}
        self._contact_steps: Dict[tuple, int] = {p: 0   for p in ROBOT_PAIRS}

        # Previous-step state for delta / causal rewards
        self._prev_red_score     = 0
        self._prev_blue_score    = 0
        self._prev_carrying_pin  = {rid: None for rid in AGENT_IDS}
        self._prev_carrying_cup  = {rid: None for rid in AGENT_IDS}
        self._prev_scores_count  = {rid: 0    for rid in AGENT_IDS}
        self._prev_goal_stacks:   Dict[int, list] = {}
        self._prev_toggle_owners: Dict[int, str]  = {}
        self._start_positions:    Dict[str, tuple] = {}

    # -------------------------------------------------------------------------
    def _get_action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    def _get_observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    # -------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.sim.reset()
        self.agents = list(AGENT_IDS)

        self._carry_steps        = {rid: 0   for rid in AGENT_IDS}
        self._contact_steps      = {p: 0     for p in ROBOT_PAIRS}
        self._prev_red_score     = 0
        self._prev_blue_score    = 0
        self._prev_carrying_pin  = {rid: None for rid in AGENT_IDS}
        self._prev_carrying_cup  = {rid: None for rid in AGENT_IDS}
        self._prev_scores_count  = {rid: 0    for rid in AGENT_IDS}
        self._prev_goal_stacks   = {g.goal_id:   list(g.stack) for g in self.sim.goals}
        self._prev_toggle_owners = {t.toggle_id: t.owner       for t in self.sim.toggles}

        robot_map = {r.robot_id: r for r in self.sim.robots}
        self._start_positions = {
            rid: (float(robot_map[rid].body.position.x),
                  float(robot_map[rid].body.position.y))
            for rid in AGENT_IDS if rid in robot_map
        }

        self.sim.timer_started = True
        observations = build_all_observations(self.sim)
        return observations, {a: {} for a in self.agents}

    # -------------------------------------------------------------------------
    def step(self, actions: Dict[str, np.ndarray]):
        try:
            robot_map = {r.robot_id: r for r in self.sim.robots}

            # Capture all pre-step state needed for reward computation
            pre_red          = self.sim.rules_engine.red_score
            pre_blue         = self.sim.rules_engine.blue_score
            pre_carrying_pin = {r.robot_id: r.carrying_pin for r in self.sim.robots}
            pre_carrying_cup = {r.robot_id: r.carrying_cup for r in self.sim.robots}
            pre_goal_stacks  = {g.goal_id:   list(g.stack) for g in self.sim.goals}
            pre_toggle_owners= {t.toggle_id: t.owner       for t in self.sim.toggles}

            sim_actions     = []
            flips_fired     = {rid: False for rid in AGENT_IDS}
            score_attempted = {rid: False for rid in AGENT_IDS}

            for rid in AGENT_IDS:
                if rid in actions:
                    act        = np.asarray(actions[rid], dtype=np.float32)
                    left       = float(np.clip(act[0], -1.0, 1.0))
                    right      = float(np.clip(act[1], -1.0, 1.0))
                    intake     = bool(act[2] > 0.5)
                    score_pin  = bool(act[3] > 0.5)
                    score_cup  = bool(act[4] > 0.5)
                    toggle     = bool(act[5] > 0.5)
                    flip_pin   = bool(act[6] > 0.5)
                    flip_cup   = bool(act[7] > 0.5)
                    match_load = bool(act[8] > 0.5) if act.shape[0] > 8 else False
                    if flip_pin or flip_cup:   flips_fired[rid]     = True
                    if score_pin or score_cup: score_attempted[rid] = True
                    sim_actions.append({
                        "left": left, "right": right, "intake": intake,
                        "score_pin": score_pin, "score_cup": score_cup,
                        "toggle": toggle, "flip_pin": flip_pin, "flip_cup": flip_cup,
                        "match_load": match_load,
                    })
                else:
                    sim_actions.append({
                        "left": 0.0, "right": 0.0, "intake": False,
                        "score_pin": False, "score_cup": False,
                        "toggle": False, "flip_pin": False, "flip_cup": False,
                        "match_load": False,
                    })

            self.sim.step(1.0 / 20.0, sim_actions)

            robot_map = {r.robot_id: r for r in self.sim.robots}
            post_red  = self.sim.rules_engine.red_score
            post_blue = self.sim.rules_engine.blue_score

            # Update carry-step counters
            for rid in AGENT_IDS:
                r = robot_map.get(rid)
                if r and (r.carrying_pin is not None or r.carrying_cup is not None):
                    self._carry_steps[rid] += 1
                else:
                    self._carry_steps[rid] = 0
                # v8.1: expose to observation_builder
                if r is not None:
                    r._carry_steps = self._carry_steps[rid]

            # Update contact-step counters — deterministic order from ROBOT_PAIRS
            for (a_id, b_id) in ROBOT_PAIRS:
                ra = robot_map.get(a_id)
                rb = robot_map.get(b_id)
                if ra and rb:
                    d = math.hypot(
                        float(ra.body.position.x) - float(rb.body.position.x),
                        float(ra.body.position.y) - float(rb.body.position.y),
                    )
                    if d < PINNING_CONTACT_DIST:
                        self._contact_steps[(a_id, b_id)] += 1
                    else:
                        self._contact_steps[(a_id, b_id)] = 0

            observations = build_all_observations(self.sim)
            rewards = self._compute_rewards(
                pre_red, pre_blue, post_red, post_blue,
                robot_map,
                pre_carrying_pin, pre_carrying_cup,
                pre_goal_stacks, pre_toggle_owners,
                flips_fired, score_attempted,
            )

            # Advance previous-step trackers
            self._prev_goal_stacks   = {g.goal_id:   list(g.stack) for g in self.sim.goals}
            self._prev_toggle_owners = {t.toggle_id: t.owner       for t in self.sim.toggles}
            for r in self.sim.robots:
                self._prev_carrying_pin[r.robot_id] = r.carrying_pin
                self._prev_carrying_cup[r.robot_id] = r.carrying_cup
                self._prev_scores_count[r.robot_id] = r.successful_scores

            done         = self.sim.match_over
            terminations = {a: done  for a in self.agents}
            truncations  = {a: False for a in self.agents}
            infos        = {a: {"phase": self.sim.match_phase,
                                "red_score": post_red, "blue_score": post_blue}
                            for a in self.agents}
            if done:
                self.agents = []

            return observations, rewards, terminations, truncations, infos

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[OverrideEnv] Step error: {e}")
            observations = {a: np.zeros(OBS_DIM, dtype=np.float32) for a in self.agents}
            rewards      = {a: 0.0  for a in self.agents}
            terminations = {a: True for a in self.agents}
            truncations  = {a: True for a in self.agents}
            infos        = {a: {}   for a in self.agents}
            self.agents  = []
            return observations, rewards, terminations, truncations, infos

    # =========================================================================
    # REWARD COMPUTATION — full parity with training/env_wrapper.py v5
    # =========================================================================
    def _compute_rewards(
        self,
        pre_red, pre_blue, post_red, post_blue,
        robot_map,
        pre_carrying_pin, pre_carrying_cup,
        pre_goal_stacks, pre_toggle_owners,
        flips_fired, score_attempted,
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
            r = robot_map.get(rid)
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
        # A robot holding the wrong element for every reachable goal earns nothing,
        # preventing the "park at goal unable to score" exploit.
        # v8: proximity cut-off after PROX_CARRY_DECAY_STEPS (matches training wrapper).
        for rid in AGENT_IDS:
            r = robot_map.get(rid)
            if not r or (r.carrying_pin is None and r.carrying_cup is None):
                continue
            if self._carry_steps[rid] >= PROX_CARRY_DECAY_STEPS:
                continue   # hard deadline: no proximity reward after 1.75 s carrying
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
                rewards[rid] += rw["carrying_proximity_scale"] / (1.0 + best / GOAL_PROXIMITY_NORM)

        # 3b. Fetch-needed-element redirect.
        # When a robot cannot score anywhere, reward it for approaching the missing
        # element type so it can complete the stack and return to a goal.
        for rid in AGENT_IDS:
            r = robot_map.get(rid)
            if not r:
                continue
            rx, ry = float(r.body.position.x), float(r.body.position.y)
            can_score_anywhere = any(
                g.alliance in ("neutral", r.alliance) and (
                    (r.carrying_pin is not None and not _stack_top_is_pin(g.stack)) or
                    (r.carrying_cup is not None and     _stack_top_is_pin(g.stack))
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
        # Per-step penalty for lingering within scoring radius of a goal where the
        # robot cannot make a legal score.  Breaks the stable wrong-element-camping
        # equilibrium while the holding-timeout ramp is still low.
        for rid in AGENT_IDS:
            r = robot_map.get(rid)
            if not r or (r.carrying_pin is None and r.carrying_cup is None):
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
                    break   # one penalty per robot per step

        # 4. Score attempt — only fires for stack-legal attempts.
        # Prevents rewarding robots for button-mashing on a goal they can never fill.
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
                can_pin = r.carrying_pin is not None and not _stack_top_is_pin(g.stack)
                can_cup = r.carrying_cup is not None and     _stack_top_is_pin(g.stack)
                if can_pin or can_cup:
                    rewards[rid] += rw["score_attempt_in_zone"]
                    break

        # 5. Holding timeout penalty — quadratic + capped ramp (v8, PROBLEM 47).
        # Matches training/env_wrapper.py: ratio = min((overshoot/ramp)^2, CAP).
        for rid in AGENT_IDS:
            cs = self._carry_steps[rid]
            if cs > HOLDING_TIMEOUT_STEPS:
                overshoot = cs - HOLDING_TIMEOUT_STEPS
                ratio = min((overshoot / HOLDING_RAMP_STEPS) ** 2,
                            HOLDING_RAMP_SQ_CAP)
                rewards[rid] += rw["holding_penalty_rate"] * ratio

        # 6. Flip penalty.
        for rid in AGENT_IDS:
            if flips_fired[rid]:
                rewards[rid] += rw["flip_penalty"]

        # 7. Idle + start-zone penalty.
        for rid in AGENT_IDS:
            r = robot_map.get(rid)
            if not r or (r.carrying_pin is not None or r.carrying_cup is not None):
                continue
            speed = math.hypot(float(r.body.velocity.x), float(r.body.velocity.y))
            if speed < IDLE_SPEED_THRESHOLD:
                rewards[rid] += rw["idle_penalty"]
            sx, sy = self._start_positions.get(rid, (0.0, 0.0))
            if math.hypot(float(r.body.position.x) - sx,
                          float(r.body.position.y) - sy) < START_ZONE_RADIUS:
                rewards[rid] += rw["start_zone_penalty"]

        # 8. Pinning violation — deterministic: ROBOT_PAIRS gives fixed (a, b) order.
        for (a_id, b_id) in ROBOT_PAIRS:
            if self._contact_steps[(a_id, b_id)] <= PINNING_STEPS_LIMIT:
                continue
            ra = robot_map.get(a_id)
            rb = robot_map.get(b_id)
            if not ra or not rb or ra.alliance == rb.alliance:
                continue
            sp_a = math.hypot(float(ra.body.velocity.x), float(ra.body.velocity.y))
            sp_b = math.hypot(float(rb.body.velocity.x), float(rb.body.velocity.y))
            rewards[a_id if sp_a >= sp_b else b_id] += rw["pinning_violation"]

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

        # 10. Goal-level causal scoring events.
        # Mirrors training/env_wrapper.py section 10 exactly:
        #   - Pin DOWN-half excluded when hidden by goal post (index 0) or opaque cup
        #   - Cup denial branch is correctly oriented (eff_clear=True → pin BLOCKED)
        #   - Yellow pin reward assigned to toggle owner, not pin placer
        for goal in self.sim.goals:
            pre_stack  = pre_goal_stacks.get(goal.goal_id, [])
            post_stack = list(goal.stack)
            if len(post_stack) <= len(pre_stack):
                continue

            new_obj, new_is_pin = post_stack[-1]

            # Identify scorer by object identity (same object that was in their carry slot)
            scoring_rid = None
            for rid in AGENT_IDS:
                if pre_carrying_pin.get(rid) is new_obj and new_is_pin:
                    scoring_rid = rid; break
                if pre_carrying_cup.get(rid) is new_obj and not new_is_pin:
                    scoring_rid = rid; break
            if scoring_rid is None:
                continue

            scorer_alliance = robot_map[scoring_rid].alliance
            ally_rids = [rid for rid in AGENT_IDS
                         if robot_map[rid].alliance == scorer_alliance]

            if new_is_pin:
                pin_idx = len(pre_stack)   # 0-based position of the new pin

                # Visibility of each half at placement time:
                #   UP  half: always visible — it is now the top of the stack.
                #   DOWN half: hidden by goal post for index-0 pin; for deeper pins,
                #              visible only when the cup below has its clear side up
                #              (clear top = transparent from above = DOWN half visible).
                up_vis = True
                if pin_idx == 0:
                    down_vis = False          # goal post always hides the bottom pin's DOWN face
                else:
                    prev_obj, prev_is_pin = post_stack[pin_idx - 1]
                    # If prev is a cup: visible when clear side is up (transparent top).
                    # If prev is somehow a pin (illegal stack): default True (safe).
                    down_vis = _eff_clear_up(prev_obj) if not prev_is_pin else True

                towner = _get_toggle_for_goal(goal, list(self.sim.toggles))

                for visible, col in ((up_vis,   new_obj.get_up_color()),
                                     (down_vis, new_obj.get_down_color())):
                    if not visible:
                        continue    # hidden half scores 0 pts — no signal
                    if col == C_RED:
                        r_val = rw["score_own_pin"] if scorer_alliance == "red" else rw["score_opp_half"]
                        b_val = rw["score_opp_half"] if scorer_alliance == "red" else rw["score_own_pin"]
                    elif col == C_BLUE:
                        r_val = rw["score_opp_half"] if scorer_alliance == "red" else rw["score_own_pin"]
                        b_val = rw["score_own_pin"]  if scorer_alliance == "blue" else rw["score_opp_half"]
                    elif col == C_YELLOW:
                        # Yellow reward belongs to whoever owns the toggle, not who placed the pin.
                        if towner == "red":
                            r_val = rw["score_yellow_owned"]; b_val = rw["score_opp_half"]
                        elif towner == "blue":
                            r_val = rw["score_opp_half"];    b_val = rw["score_yellow_owned"]
                        else:
                            r_val = b_val = rw["score_yellow_neutral"]
                    else:
                        r_val = b_val = 0.0
                    for rid in ["red1",  "red2"]:  rewards[rid] += r_val
                    for rid in ["blue1", "blue2"]: rewards[rid] += b_val

            else:
                # Cup placed on top of a pin.
                # Orientation key (mirrors game_objects.py get_score):
                #   eff_clear_up=True  → clear side up → dark bottom faces pin UP → BLOCKED (denial)
                #   eff_clear_up=False → dark side up  → clear bottom faces pin UP → VISIBLE
                eff_clear = _eff_clear_up(new_obj)
                cup_idx   = len(pre_stack)
                if cup_idx > 0:
                    below_obj, below_is_pin = post_stack[cup_idx - 1]
                    if below_is_pin:
                        pin_up_col = below_obj.get_up_color()
                        if eff_clear:
                            # Dark bottom faces pin — pin's UP half is blocked (denied).
                            if _is_opponent_color(pin_up_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_success"]
                            elif _is_own_color(pin_up_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_own"]
                        else:
                            # Clear bottom faces pin — pin's UP half remains visible.
                            if _is_opponent_color(pin_up_col, scorer_alliance):
                                for rid in ally_rids: rewards[rid] += rw["denial_preserved_opp"]
                if cup_idx >= 1:
                    for rid in ally_rids: rewards[rid] += rw["stack_bonus"]

        # 11. Toggle events.
        for toggle in self.sim.toggles:
            prev_owner = pre_toggle_owners.get(toggle.toggle_id)
            curr_owner = toggle.owner
            if prev_owner != curr_owner:
                if curr_owner == "red":
                    for rid in ["red1",  "red2"]:  rewards[rid] += rw["toggle_gain"]
                    for rid in ["blue1", "blue2"]: rewards[rid] += rw["toggle_loss"]
                elif curr_owner == "blue":
                    for rid in ["blue1", "blue2"]: rewards[rid] += rw["toggle_gain"]
                    for rid in ["red1",  "red2"]:  rewards[rid] += rw["toggle_loss"]

        # 12. Midfield endgame bonus — v8.3 parity fix (PROBLEM 56):
        # Only fires in the final PARK_WINDOW_SECONDS (3 s) to match
        # training/env_wrapper.py.  Previously fired for the full 20-s
        # endgame (20× too large), making evaluation scores meaningless.
        tr = float(self.sim.time_remaining)
        if self.sim.rules_engine.endgame_active and tr <= PARK_WINDOW_SECONDS:
            mc_x, mc_y = MIDFIELD_CENTER
            for r in self.sim.robots:
                if abs(float(r.body.position.x) - mc_x) + \
                   abs(float(r.body.position.y) - mc_y) <= MIDFIELD_HALF + 10.0:
                    rewards[r.robot_id] += rw["midfield_endgame"]

        return rewards

    # -------------------------------------------------------------------------
    def render(self):
        if not self.headless:
            self.sim.render()

    def close(self):
        pass


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================

def _stack_top_is_pin(stack) -> bool:
    """True if the topmost element in a goal stack is a pin (not a cup).

    Centralises the ``bool(stack) and stack[-1][1]`` idiom used throughout
    the reward logic so that any future stack-format change only needs one fix.
    """
    return bool(stack) and bool(stack[-1][1])


def _eff_clear_up(cup) -> bool:
    """True = the cup's clear/white half is UP in the stack.

    Accounts for the ``flipped`` flag that can invert the orientation after a
    flip action.  Used by both the cup-denial reward and the pin-visibility
    (DOWN-half) check.
    """
    flipped = getattr(cup, "flipped", False)
    return (not cup.clear_on_top) if flipped else cup.clear_on_top


def _get_toggle_for_goal(goal, toggles) -> Optional[str]:
    """Return the alliance ("red"/"blue") owning the toggle associated with
    ``goal``, or None if the toggle is neutral or unassigned.

    Toggle assignment uses quadrant proximity to field centre (72", 72").
    """
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


def make_override_env(render_mode=None):
    """Factory function to create a wrapped Override environment."""
    return OverrideEnv(render_mode=render_mode)
