"""
training/network.py
────────────────────────────────────────────────────────────────────────────
Neural network architecture for the VEX Override MAPPO system.

Components
----------
Policy (actor)
  - Shared MLP backbone: 524 → [512, 256, 128]
  - Continuous head: 128 → 2  (mean + log_std for left/right motors)
  - Discrete head:   128 → 7  (Bernoulli logit per button action)

CentralizedCritic
  - Takes concatenated obs from both robots on the same alliance: 1048 → 1
  - Hidden: [512, 256, 128] → 1
  - Used ONLY during training; discarded at inference.

Both networks use LayerNorm for training stability (no batch-norm artefacts
with small minibatches).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli

from config.hyperparameters import (
    OBS_DIM, ACTION_CONT, ACTION_DISC,
    ACTOR_HIDDEN, CRITIC_HIDDEN,
    LOG_STD_MIN, LOG_STD_MAX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mlp(dims: list, activation=nn.ELU, output_activation=None) -> nn.Sequential:
    """Build a fully-connected MLP with LayerNorm after each hidden layer."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:          # hidden layers only
            layers.append(nn.LayerNorm(dims[i + 1]))
            act = activation()
            layers.append(act)
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


def _init_weights(module, gain=1.0):
    """Orthogonal init — standard practice for PPO."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Policy (actor)
# ─────────────────────────────────────────────────────────────────────────────
class Policy(nn.Module):
    """
    Decentralised actor — uses only local 524-dim observation.

    Forward returns
    ---------------
    cont_mean   : (B, 2)   — mean of left/right motor Gaussian
    cont_log_std: (B, 2)   — log-std (clamped)
    disc_logits : (B, 7)   — raw logits for Bernoulli button heads
    """

    def __init__(self,
                 obs_dim: int = OBS_DIM,
                 cont_dim: int = ACTION_CONT,
                 disc_dim: int = ACTION_DISC,
                 hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = ACTOR_HIDDEN

        self.backbone = _mlp([obs_dim] + hidden)

        feat_dim = hidden[-1]
        # Continuous: mean and log_std
        self.cont_mean    = nn.Linear(feat_dim, cont_dim)
        self.cont_log_std = nn.Linear(feat_dim, cont_dim)
        # Discrete: one logit per button
        self.disc_head    = nn.Linear(feat_dim, disc_dim)

        # Init: small gain for output layers (standard PPO recipe)
        self.backbone.apply(lambda m: _init_weights(m, gain=math.sqrt(2)))
        _init_weights(self.cont_mean,    gain=0.01)
        _init_weights(self.cont_log_std, gain=0.01)
        _init_weights(self.disc_head,    gain=0.01)

    def forward(self, obs: torch.Tensor):
        """obs : (B, 524)  →  cont_mean, cont_log_std, disc_logits"""
        feats = self.backbone(obs)
        mean    = self.cont_mean(feats)
        log_std = self.cont_log_std(feats).clamp(LOG_STD_MIN, LOG_STD_MAX)
        logits  = self.disc_head(feats)
        return mean, log_std, logits

    def get_action(self, obs: torch.Tensor, action_mask: torch.Tensor = None,
                   deterministic: bool = False):
        """
        Sample (or argmax) an action from the current policy.

        Parameters
        ----------
        obs          : (B, 524) or (524,) — current observation(s)
        action_mask  : (B, 7) or (7,) bool tensor — True = action legal
        deterministic: bool — if True, use mode instead of sampling

        Returns
        -------
        cont_action : (B, 2)  — motors in [-1, 1]
        disc_action : (B, 7)  — binary button presses
        log_prob    : (B,)    — total log-probability of the action
        entropy     : (B,)    — total entropy
        """
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)

        mean, log_std, logits = self.forward(obs)

        # ── Apply action masking: set logits of illegal actions to -∞ ──
        if action_mask is not None:
            mask_f = action_mask.float()
            logits = logits + (1.0 - mask_f) * (-1e9)

        # ── Continuous (diagonal Gaussian) ──
        std = log_std.exp()
        cont_dist = Normal(mean, std)
        if deterministic:
            cont_action_raw = mean
        else:
            cont_action_raw = cont_dist.rsample()
        cont_action = torch.tanh(cont_action_raw)

        # log prob with tanh squashing correction
        cont_lp = cont_dist.log_prob(cont_action_raw) \
                  - torch.log(1.0 - cont_action.pow(2) + 1e-6)
        cont_lp = cont_lp.sum(-1)   # (B,)

        # ── Discrete (independent Bernoulli per button) ──
        disc_dist = Bernoulli(logits=logits)
        if deterministic:
            disc_action = (logits > 0).float()
        else:
            disc_action = disc_dist.sample()
        disc_lp = disc_dist.log_prob(disc_action).sum(-1)   # (B,)

        log_prob = cont_lp + disc_lp
        entropy  = cont_dist.entropy().sum(-1) + disc_dist.entropy().sum(-1)

        if squeeze:
            cont_action = cont_action.squeeze(0)
            disc_action = disc_action.squeeze(0)
            log_prob    = log_prob.squeeze(0)
            entropy     = entropy.squeeze(0)

        return cont_action, disc_action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor,
                         cont_action: torch.Tensor,
                         disc_action: torch.Tensor,
                         action_mask: torch.Tensor = None):
        """
        Evaluate log-probabilities and entropy of stored actions.
        Used during the PPO update phase.
        """
        mean, log_std, logits = self.forward(obs)

        if action_mask is not None:
            logits = logits + (1.0 - action_mask.float()) * (-1e9)

        std = log_std.exp()
        cont_dist = Normal(mean, std)

        # Recover pre-tanh action
        cont_raw = torch.atanh(cont_action.clamp(-0.9999, 0.9999))
        cont_lp  = cont_dist.log_prob(cont_raw) \
                   - torch.log(1.0 - cont_action.pow(2) + 1e-6)
        cont_lp  = cont_lp.sum(-1)
        cont_ent = cont_dist.entropy().sum(-1)

        disc_dist = Bernoulli(logits=logits)
        disc_lp   = disc_dist.log_prob(disc_action).sum(-1)
        disc_ent  = disc_dist.entropy().sum(-1)

        log_prob = cont_lp + disc_lp
        entropy  = cont_ent + disc_ent
        return log_prob, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Centralized Critic (MAPPO — sees both teammates' observations)
# ─────────────────────────────────────────────────────────────────────────────
class CentralizedCritic(nn.Module):
    """
    Centralised value function.
    Input  : concatenated observations of both alliance robots → (B, 1048)
    Output : scalar value estimate → (B, 1)
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = CRITIC_HIDDEN
        joint_dim = obs_dim * 2   # both robots
        self.net = _mlp([joint_dim] + hidden + [1])
        self.net.apply(lambda m: _init_weights(m, gain=math.sqrt(2)))
        # Last layer: small init for stable early value estimates
        _init_weights(list(self.net.children())[-1], gain=1.0)

    def forward(self, obs1: torch.Tensor, obs2: torch.Tensor) -> torch.Tensor:
        """
        obs1, obs2 : (B, 524) — observations of robot 0 and robot 1 (same alliance)
        Returns    : (B, 1)   — value estimates
        """
        joint = torch.cat([obs1, obs2], dim=-1)
        return self.net(joint)
