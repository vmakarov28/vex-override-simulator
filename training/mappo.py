"""
training/mappo.py
────────────────────────────────────────────────────────────────────────────
Multi-Agent PPO (MAPPO) update logic.

Architecture
------------
- Two policy instances: red_policy and blue_policy.
- Each policy has its own centralized critic (sees both teammates' obs).
- Rollout buffer collects (obs, action, log_prob, reward, value, done, mask).
- GAE computes advantages in place.
- PPO clip update runs PPO_EPOCHS passes over minibatches.

All tensors on the specified device.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional

from config.hyperparameters import (
    GAMMA, GAE_LAMBDA, CLIP_EPS,
    VALUE_LOSS_COEF, ENTROPY_COEF, ENTROPY_COEF_MIN, ENTROPY_ANNEAL_STEPS,
    MAX_GRAD_NORM, PPO_EPOCHS, MINIBATCH_SIZE,
    ROLLOUT_STEPS,
    OBS_DIM, ACTION_CONT, ACTION_DISC,
    LEARNING_RATE, CRITIC_LR,
)
from training.network import Policy, CentralizedCritic


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    """
    Stores one rollout worth of transitions for a SINGLE alliance (2 robots).

    Per-agent fields (indexed by robot slot 0 or 1 within the alliance):
      obs, cont_act, disc_act, log_prob, mask

    Alliance-level fields:
      reward (alliance-shared), value (centralized), done
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.reset()

    def reset(self):
        self.ptr  = 0
        self.full = False

        # Accumulate as Python lists of numpy arrays during rollout.
        # This eliminates ~426K individual CUDA indexed writes per update cycle
        # (previously: 13 GPU tensor writes × 32 envs × 2 alliances × 512 steps).
        # All data is bulk-transferred to GPU in a single pass at compute_gae().
        self._obs_l   = [[], []]
        self._cont_l  = [[], []]
        self._disc_l  = [[], []]
        self._lp_l    = [[], []]
        self._mask_l  = [[], []]
        self._rew_l   = []
        self._val_l   = []
        self._done_l  = []

        # These are populated by compute_gae() and consumed by merge_buffers()
        # and the PPO update — they do not exist during rollout collection.
        self.obs      = [None, None]
        self.cont_act = [None, None]
        self.disc_act = [None, None]
        self.log_prob = [None, None]
        self.mask     = [None, None]
        self.reward   = None
        self.value    = None
        self.done     = None
        self.returns    = None
        self.advantages = None
        self.last_value = 0.0

    @staticmethod
    def _to_np(x):
        """Convert a GPU tensor, CPU tensor, or numpy array to a numpy array."""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def add(self, obs0, obs1, cont0, cont1, disc0, disc1,
            lp0, lp1, reward, value, done, mask0, mask1):
        """
        Append one transition.
        Accepts numpy arrays, CPU tensors, or GPU tensors — all are normalized
        to numpy immediately so the list stores only plain CPU arrays.
        """
        _n = self._to_np
        self._obs_l[0].append(_n(obs0))
        self._obs_l[1].append(_n(obs1))
        self._cont_l[0].append(_n(cont0))
        self._cont_l[1].append(_n(cont1))
        self._disc_l[0].append(_n(disc0))
        self._disc_l[1].append(_n(disc1))
        self._lp_l[0].append(float(_n(lp0)))
        self._lp_l[1].append(float(_n(lp1)))
        self._mask_l[0].append(_n(mask0))
        self._mask_l[1].append(_n(mask1))
        self._rew_l.append(float(reward))
        self._val_l.append(float(value))
        self._done_l.append(bool(done))

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def compute_gae(self):
        """
        Compute GAE on CPU (numpy), then bulk-transfer all data to GPU.

        Running the GAE recurrence on numpy avoids 16K individual CUDA
        element-accesses that a Python for-loop over GPU tensors would issue.
        The final torch.from_numpy().to(device) calls are a handful of bulk
        transfers instead of hundreds of thousands of small indexed writes.
        """
        N = self.capacity

        rewards = np.array(self._rew_l,  dtype=np.float32)
        values  = np.array(self._val_l,  dtype=np.float32)
        dones   = np.array(self._done_l, dtype=np.bool_)

        # GAE recurrence — pure numpy, no CUDA kernel dispatches
        advantages = np.zeros(N, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(N)):
            not_done = 0.0 if dones[t] else 1.0
            next_val = values[t + 1] if t < N - 1 else float(self.last_value)
            delta    = rewards[t] + GAMMA * next_val * not_done - values[t]
            gae      = delta + GAMMA * GAE_LAMBDA * not_done * gae
            advantages[t] = gae

        returns  = advantages + values
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        adv_norm = (advantages - adv_mean) / adv_std

        # Bulk GPU transfer — O(fields) transfers instead of O(fields × steps)
        D = self.device
        def _t(arr):  return torch.from_numpy(np.ascontiguousarray(arr)).to(D)
        def _s(lst):  return _t(np.stack(lst))

        self.obs      = [_s(self._obs_l[0]),  _s(self._obs_l[1])]
        self.cont_act = [_s(self._cont_l[0]), _s(self._cont_l[1])]
        self.disc_act = [_s(self._disc_l[0]), _s(self._disc_l[1])]
        self.log_prob = [_s(self._lp_l[0]),   _s(self._lp_l[1])]
        self.mask     = [_s(self._mask_l[0]), _s(self._mask_l[1])]
        self.reward   = _t(rewards)
        self.value    = _t(values)
        self.done     = torch.from_numpy(dones).to(D)
        self.returns    = _t(returns)
        self.advantages = _t(adv_norm)

    def get_minibatches(self, batch_size: int):
        N = self.capacity
        idx = torch.randperm(N, device=self.device)
        for start in range(0, N - batch_size + 1, batch_size):
            yield idx[start:start + batch_size]

    @staticmethod
    def merge_buffers(bufs: list) -> "RolloutBuffer":
        """
        Merge per-env buffers into one for a joint PPO update.
        GAE must already be computed on each buffer before calling this.
        Advantages are re-normalized globally after merging.
        """
        total  = sum(b.capacity for b in bufs)
        merged = RolloutBuffer(total, bufs[0].device)

        merged.obs[0]      = torch.cat([b.obs[0]      for b in bufs], dim=0)
        merged.obs[1]      = torch.cat([b.obs[1]      for b in bufs], dim=0)
        merged.cont_act[0] = torch.cat([b.cont_act[0] for b in bufs], dim=0)
        merged.cont_act[1] = torch.cat([b.cont_act[1] for b in bufs], dim=0)
        merged.disc_act[0] = torch.cat([b.disc_act[0] for b in bufs], dim=0)
        merged.disc_act[1] = torch.cat([b.disc_act[1] for b in bufs], dim=0)
        merged.log_prob[0] = torch.cat([b.log_prob[0] for b in bufs], dim=0)
        merged.log_prob[1] = torch.cat([b.log_prob[1] for b in bufs], dim=0)
        merged.mask[0]     = torch.cat([b.mask[0]     for b in bufs], dim=0)
        merged.mask[1]     = torch.cat([b.mask[1]     for b in bufs], dim=0)
        merged.reward      = torch.cat([b.reward      for b in bufs], dim=0)
        merged.value       = torch.cat([b.value       for b in bufs], dim=0)
        merged.done        = torch.cat([b.done        for b in bufs], dim=0)
        merged.returns     = torch.cat([b.returns     for b in bufs], dim=0)

        # Re-normalize advantages globally across all envs
        all_adv            = torch.cat([b.advantages  for b in bufs], dim=0)
        merged.advantages  = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

        merged.ptr  = total
        merged.full = True
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# MAPPO Trainer
# ─────────────────────────────────────────────────────────────────────────────
class MAPPOTrainer:
    """
    Manages two alliance policies (red, blue) with their centralized critics.

    Single-env usage
    ----------------
    trainer = MAPPOTrainer(device)
    trainer.reset_buffers()
    for each step:
        agent_data = trainer.get_actions(obs, masks)
        obs, rew, done, info = env.step(...)
        trainer.store_transition(agent_data, rew, done)
    trainer.set_last_values(obs)
    trainer.update()

    Multi-env usage
    ---------------
    red_bufs  = [RolloutBuffer(ROLLOUT_STEPS, device) for _ in range(N)]
    blue_bufs = [RolloutBuffer(ROLLOUT_STEPS, device) for _ in range(N)]
    # ... fill buffers manually ...
    for i in range(N):
        rv, bv = trainer.compute_last_values(last_obs[i])
        red_bufs[i].last_value  = rv
        blue_bufs[i].last_value = bv
        red_bufs[i].compute_gae()
        blue_bufs[i].compute_gae()
    trainer.update_multi_env(red_bufs, blue_bufs)
    """

    def __init__(self, device: torch.device):
        self.device = device

        self.red_policy  = Policy().to(device)
        self.blue_policy = Policy().to(device)

        self.red_critic  = CentralizedCritic().to(device)
        self.blue_critic = CentralizedCritic().to(device)

        self.red_optimizer = optim.Adam(
            list(self.red_policy.parameters()) +
            list(self.red_critic.parameters()),
            lr=LEARNING_RATE, eps=1e-5)
        self.blue_optimizer = optim.Adam(
            list(self.blue_policy.parameters()) +
            list(self.blue_critic.parameters()),
            lr=LEARNING_RATE, eps=1e-5)

        self.red_buf  = RolloutBuffer(ROLLOUT_STEPS, device)
        self.blue_buf = RolloutBuffer(ROLLOUT_STEPS, device)

        self.total_updates   = 0
        self.total_env_steps = 0

    # ── Entropy schedule ──────────────────────────────────────────────────
    def _current_entropy_coef(self) -> float:
        frac = min(1.0, self.total_env_steps / ENTROPY_ANNEAL_STEPS)
        return ENTROPY_COEF + frac * (ENTROPY_COEF_MIN - ENTROPY_COEF)

    # ── Action sampling ───────────────────────────────────────────────────
    @torch.no_grad()
    def get_actions(
        self,
        obs: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        deterministic: bool = False,
        red_policy_override: Optional[Policy] = None,
        blue_policy_override: Optional[Policy] = None,
    ) -> Dict:
        """
        Sample actions for all four agents.

        Supports both single-env and batched-env inputs:
          Single : obs[rid] shape (OBS_DIM,)      → values are scalars / 1-D tensors
          Batched: obs[rid] shape (N, OBS_DIM)    → values are N-D tensors

        red_policy_override / blue_policy_override allow injecting a frozen
        pool policy for self-play opponent sampling without touching the
        trained policy objects.
        """
        obs_t  = {rid: torch.FloatTensor(obs[rid]).to(self.device)
                  for rid in obs}
        mask_t = {rid: torch.BoolTensor(masks[rid]).to(self.device)
                  for rid in masks}

        # Detect whether obs are batched (N, OBS_DIM) or single (OBS_DIM,)
        batched = next(iter(obs_t.values())).dim() == 2

        results = {}
        for alliance, policy, critic, rids in [
            ("red",  red_policy_override  or self.red_policy,
                     self.red_critic,  ["red1",  "red2"]),
            ("blue", blue_policy_override or self.blue_policy,
                     self.blue_critic, ["blue1", "blue2"]),
        ]:
            r0, r1 = rids
            o0, o1 = obs_t[r0], obs_t[r1]
            m0, m1 = mask_t[r0], mask_t[r1]

            c0, d0, lp0, _ = policy.get_action(o0, m0, deterministic)
            c1, d1, lp1, _ = policy.get_action(o1, m1, deterministic)

            if batched:
                # o0/o1 are (N, OBS_DIM) — critic handles the batch directly
                val = critic(o0, o1).squeeze(-1)   # (N,)
            else:
                val = critic(o0.unsqueeze(0), o1.unsqueeze(0)).squeeze()  # scalar

            results[r0] = {"cont": c0, "disc": d0, "log_prob": lp0,
                           "obs": o0, "mask": m0}
            results[r1] = {"cont": c1, "disc": d1, "log_prob": lp1,
                           "obs": o1, "mask": m1}
            results[f"{alliance}_value"] = val   # (N,) or scalar

        return results

    def store_transition(
        self,
        agent_data: Dict,
        rewards: Dict[str, float],
        done: bool,
    ):
        """Store one environment step in both alliance buffers."""
        red_reward  = (rewards.get("red1",  0.0) + rewards.get("red2",  0.0)) / 2.0
        blue_reward = (rewards.get("blue1", 0.0) + rewards.get("blue2", 0.0)) / 2.0

        self.red_buf.add(
            obs0=agent_data["red1"]["obs"],
            obs1=agent_data["red2"]["obs"],
            cont0=agent_data["red1"]["cont"],
            cont1=agent_data["red2"]["cont"],
            disc0=agent_data["red1"]["disc"],
            disc1=agent_data["red2"]["disc"],
            lp0=agent_data["red1"]["log_prob"],
            lp1=agent_data["red2"]["log_prob"],
            reward=red_reward,
            value=agent_data["red_value"],
            done=done,
            mask0=agent_data["red1"]["mask"],
            mask1=agent_data["red2"]["mask"],
        )

        self.blue_buf.add(
            obs0=agent_data["blue1"]["obs"],
            obs1=agent_data["blue2"]["obs"],
            cont0=agent_data["blue1"]["cont"],
            cont1=agent_data["blue2"]["cont"],
            disc0=agent_data["blue1"]["disc"],
            disc1=agent_data["blue2"]["disc"],
            lp0=agent_data["blue1"]["log_prob"],
            lp1=agent_data["blue2"]["log_prob"],
            reward=blue_reward,
            value=agent_data["blue_value"],
            done=done,
            mask0=agent_data["blue1"]["mask"],
            mask1=agent_data["blue2"]["mask"],
        )

        self.total_env_steps += 1

    def reset_buffers(self):
        self.red_buf.reset()
        self.blue_buf.reset()

    @torch.no_grad()
    def set_last_values(self, obs: Dict[str, np.ndarray]):
        """Bootstrap value at end of single-env rollout."""
        rv, bv = self.compute_last_values(obs)
        self.red_buf.last_value  = rv
        self.blue_buf.last_value = bv

    @torch.no_grad()
    def compute_last_values(self, obs: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Return (red_value, blue_value) bootstrap estimates for one env's last obs."""
        o_r0 = torch.FloatTensor(obs["red1"]).to(self.device).unsqueeze(0)
        o_r1 = torch.FloatTensor(obs["red2"]).to(self.device).unsqueeze(0)
        o_b0 = torch.FloatTensor(obs["blue1"]).to(self.device).unsqueeze(0)
        o_b1 = torch.FloatTensor(obs["blue2"]).to(self.device).unsqueeze(0)
        red_v  = self.red_critic(o_r0,  o_r1).item()
        blue_v = self.blue_critic(o_b0, o_b1).item()
        return red_v, blue_v

    # ── PPO core ──────────────────────────────────────────────────────────
    def _run_ppo_epochs(
        self,
        buf: RolloutBuffer,
        policy: Policy,
        critic: CentralizedCritic,
        optimizer: optim.Optimizer,
        ent_coef: float,
        stats: Dict,
        prefix: str,
    ):
        """Run PPO_EPOCHS update passes on a pre-filled buffer. Mutates stats."""
        p_losses, v_losses, entropies = [], [], []

        for _ in range(PPO_EPOCHS):
            for mb_idx in buf.get_minibatches(MINIBATCH_SIZE):
                obs0_mb  = buf.obs[0][mb_idx]
                obs1_mb  = buf.obs[1][mb_idx]
                c0_mb    = buf.cont_act[0][mb_idx]
                c1_mb    = buf.cont_act[1][mb_idx]
                d0_mb    = buf.disc_act[0][mb_idx]
                d1_mb    = buf.disc_act[1][mb_idx]
                lp0_old  = buf.log_prob[0][mb_idx]
                lp1_old  = buf.log_prob[1][mb_idx]
                mask0_mb = buf.mask[0][mb_idx]
                mask1_mb = buf.mask[1][mb_idx]
                adv_mb   = buf.advantages[mb_idx]
                ret_mb   = buf.returns[mb_idx]

                lp0_new, ent0 = policy.evaluate_actions(
                    obs0_mb, c0_mb, d0_mb, mask0_mb)
                lp1_new, ent1 = policy.evaluate_actions(
                    obs1_mb, c1_mb, d1_mb, mask1_mb)

                lp_old  = (lp0_old + lp1_old) / 2.0
                lp_new  = (lp0_new + lp1_new) / 2.0
                entropy = (ent0 + ent1) / 2.0

                ratio = (lp_new - lp_old).exp()
                clip  = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS)

                policy_loss = -torch.min(
                    ratio * adv_mb, clip * adv_mb).mean()

                values_mb = critic(obs0_mb, obs1_mb).squeeze(-1)
                old_vals  = buf.value[mb_idx]
                v_clipped = old_vals + (values_mb - old_vals).clamp(
                    -CLIP_EPS, CLIP_EPS)
                v_loss = torch.max(
                    (values_mb - ret_mb).pow(2),
                    (v_clipped  - ret_mb).pow(2)).mean()

                # KL regularization to prevent policy collapse
                kl_loss = 0.5 * ((lp_old - lp_new).pow(2)).mean()

                loss = (policy_loss
                        + VALUE_LOSS_COEF * v_loss
                        - ent_coef * entropy.mean()
                        + 0.01 * kl_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(critic.parameters()),
                    MAX_GRAD_NORM)
                optimizer.step()

                p_losses.append(policy_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.mean().item())

        stats[f"{prefix}_policy_loss"] = np.mean(p_losses) if p_losses else 0.0
        stats[f"{prefix}_value_loss"]  = np.mean(v_losses) if v_losses else 0.0
        stats[f"{prefix}_entropy"]     = np.mean(entropies) if entropies else 0.0

    def update(self) -> Dict:
        """Single-env PPO update. Computes GAE internally."""
        ent_coef = self._current_entropy_coef()
        stats = {
            "red_policy_loss": 0.0, "red_value_loss": 0.0, "red_entropy": 0.0,
            "blue_policy_loss": 0.0, "blue_value_loss": 0.0, "blue_entropy": 0.0,
        }

        self.red_buf.compute_gae()
        self._run_ppo_epochs(self.red_buf, self.red_policy, self.red_critic,
                             self.red_optimizer, ent_coef, stats, "red")

        self.blue_buf.compute_gae()
        self._run_ppo_epochs(self.blue_buf, self.blue_policy, self.blue_critic,
                             self.blue_optimizer, ent_coef, stats, "blue")

        self.total_updates += 1
        return stats

    def update_multi_env(
        self,
        red_bufs: List[RolloutBuffer],
        blue_bufs: List[RolloutBuffer],
    ) -> Dict:
        """
        Multi-env PPO update.  Per-env GAE must already be computed before
        calling this (call buf.compute_gae() on each buffer first).
        Merges all per-env buffers, re-normalizes advantages globally,
        then runs PPO_EPOCHS on the joint batch.
        """
        ent_coef = self._current_entropy_coef()
        stats = {
            "red_policy_loss": 0.0, "red_value_loss": 0.0, "red_entropy": 0.0,
            "blue_policy_loss": 0.0, "blue_value_loss": 0.0, "blue_entropy": 0.0,
        }

        merged_red  = RolloutBuffer.merge_buffers(red_bufs)
        merged_blue = RolloutBuffer.merge_buffers(blue_bufs)

        self._run_ppo_epochs(merged_red,  self.red_policy,  self.red_critic,
                             self.red_optimizer,  ent_coef, stats, "red")
        self._run_ppo_epochs(merged_blue, self.blue_policy, self.blue_critic,
                             self.blue_optimizer, ent_coef, stats, "blue")

        self.total_updates += 1
        return stats

    # ── Checkpointing ─────────────────────────────────────────────────────
    def save(self, path: str, extra: dict = None):
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        checkpoint = {
            "red_policy":      self.red_policy.state_dict(),
            "blue_policy":     self.blue_policy.state_dict(),
            "red_critic":      self.red_critic.state_dict(),
            "blue_critic":     self.blue_critic.state_dict(),
            "red_opt":         self.red_optimizer.state_dict(),
            "blue_opt":        self.blue_optimizer.state_dict(),
            "total_updates":   self.total_updates,
            "total_env_steps": self.total_env_steps,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        print(f"[MAPPO] Saved checkpoint → {path}")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.red_policy.load_state_dict(ck["red_policy"])
        self.blue_policy.load_state_dict(ck["blue_policy"])
        self.red_critic.load_state_dict(ck["red_critic"])
        self.blue_critic.load_state_dict(ck["blue_critic"])
        self.red_optimizer.load_state_dict(ck["red_opt"])
        self.blue_optimizer.load_state_dict(ck["blue_opt"])
        self.total_updates   = ck.get("total_updates", 0)
        self.total_env_steps = ck.get("total_env_steps", 0)
        print(f"[MAPPO] Loaded checkpoint ← {path} "
              f"(step {self.total_env_steps:,})")
        return ck
