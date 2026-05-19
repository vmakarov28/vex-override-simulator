"""
utils/opponent_pool.py
────────────────────────────────────────────────────────────────────────────
Manages a rotating pool of past policy checkpoints for league-style self-play.

Design
------
- Maintains up to POOL_SIZE policy snapshots (state dicts).
- The "current" policy (being trained) is always added every CHECKPOINT_EVERY
  gradient updates.
- When choosing an opponent, we either use the latest policy (self-play)
  with probability (1 - POOL_SAMPLE_PROB) or sample uniformly from the pool
  with probability POOL_SAMPLE_PROB.
- Oldest checkpoints are evicted when the pool is full (FIFO).
"""

import os
import copy
import random
import torch
from typing import Optional, List
from config.hyperparameters import POOL_SIZE, POOL_SAMPLE_PROB


class OpponentPool:
    """
    Circular buffer of past policy checkpoints.

    Usage
    -----
    pool = OpponentPool(policy_cls, device)
    pool.add(current_policy)        # after every CHECKPOINT_EVERY updates
    opp_policy = pool.sample()      # returns a frozen Policy instance
    """

    def __init__(self, policy_cls, device: torch.device,
                 max_size: int = POOL_SIZE,
                 sample_prob: float = POOL_SAMPLE_PROB):
        """
        Parameters
        ----------
        policy_cls  : class — the Policy class (from training/network.py)
        device      : torch.device
        max_size    : int — max number of checkpoints to keep
        sample_prob : float — probability of sampling from pool vs. latest
        """
        self.policy_cls  = policy_cls
        self.device      = device
        self.max_size    = max_size
        self.sample_prob = sample_prob

        self._pool: List[dict] = []          # list of state_dict snapshots
        self._latest_state: Optional[dict] = None

    def add(self, policy) -> None:
        """Snapshot the current policy's weights and add to the pool."""
        state = copy.deepcopy(policy.state_dict())
        self._latest_state = state
        self._pool.append(state)
        if len(self._pool) > self.max_size:
            self._pool.pop(0)   # evict oldest

    def sample(self, policy_instance=None):
        """
        Return a policy instance loaded with a pooled checkpoint.

        If pool is empty or random draw favours "latest", returns the current
        policy weights (self-play against most recent).  Otherwise returns a
        frozen copy loaded with a random historical checkpoint.

        Parameters
        ----------
        policy_instance : Policy — optional reusable instance (avoids re-allocation).
                          If None, a new instance is created (slower).

        Returns
        -------
        Policy — a frozen (no-grad) policy ready for inference.
        """
        if not self._pool or random.random() > self.sample_prob:
            # Return a frozen copy of the latest weights
            state = self._latest_state if self._latest_state else None
        else:
            state = random.choice(self._pool)

        if policy_instance is None:
            policy_instance = self.policy_cls().to(self.device)

        if state is not None:
            policy_instance.load_state_dict(state)

        policy_instance.eval()
        for param in policy_instance.parameters():
            param.requires_grad_(False)
        return policy_instance

    def save(self, path: str) -> None:
        """Persist the entire pool to disk (for resuming training)."""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        torch.save({
            "pool": self._pool,
            "latest": self._latest_state,
            "max_size": self.max_size,
            "sample_prob": self.sample_prob,
        }, path)
        print(f"[Pool] Saved {len(self._pool)} checkpoints → {path}")

    def load(self, path: str) -> None:
        """Restore pool from disk."""
        if not os.path.exists(path):
            print(f"[Pool] No pool file at {path}. Starting fresh.")
            return
        data = torch.load(path, map_location=self.device)
        self._pool          = data.get("pool", [])
        self._latest_state  = data.get("latest", None)
        self.max_size       = data.get("max_size", self.max_size)
        self.sample_prob    = data.get("sample_prob", self.sample_prob)
        print(f"[Pool] Loaded {len(self._pool)} checkpoints from {path}")

    def __len__(self):
        return len(self._pool)

    def is_empty(self):
        return len(self._pool) == 0
