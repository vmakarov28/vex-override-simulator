"""
training/network.py  (v9.2 — goal attention, 610-dim obs)
────────────────────────────────────────────────────────────────────────────
Neural network architecture for the VEX Override MAPPO system.

v7 change
---------
A shared ObsEncoder is now used by both Policy and Critic.  Rather than
flattening all 9 goal slots into one big MLP input, the encoder:
  1. Splits the 592-dim observation into (non-goal, goal-slots).
  2. Embeds each goal slot (17 dims) into a small vector via a shared MLP.
  3. Projects the non-goal context to a query vector.
  4. Uses dot-product attention over the 9 embedded goal slots, with
     softmax weights and weighted pooling.
  5. Concatenates [non-goal features, attended goals] and runs the result
     through the main MLP backbone.

This lets the policy *selectively focus* on the goal(s) most relevant to
the current situation (e.g. the goal it's about to score on, or the goal
it's trying to deny) rather than blindly attending to all nine equally.

v8 change
---------
OBS_DIM expanded from 554 → 564 (+10 self-awareness / defensive-intel
features).  The encoder is fully dimension-agnostic — non_goal_dim is
derived at init time from obs_dim minus the fixed goal block (153 dims),
so no architectural change is needed beyond the updated OBS_DIM constant.

v8.3 change
-----------
OBS_DIM expanded from 564 → 588 (+20 per-pin nearest-goal distances,
+4 movement-intelligence features: own speed, teammate carry_steps, opp
speed magnitudes).  GOAL_OFFSET=56 and GOAL_BLOCK_END=209 are unchanged,
so the attention mechanism requires no modification.
non_goal_dim = 588 - 9×17 = 435 (was 411); handled dynamically at init.

v9 change
---------
OBS_DIM expanded from 588 → 592 (+4 pressure features: in_scoring_range,
being_pinned_frac, score_lead_tight, dist_to_nearest_scorable).
GOAL_OFFSET=56 and GOAL_BLOCK_END=209 unchanged; new dims land in the
non-goal slice.  non_goal_dim = 592 - 9×17 = 439 (was 435).

v9.2 change
-----------
OBS_DIM expanded from 592 → 610 (+18 = 9 goals × 2 new per-goal features:
  [17] yellow_toggle_mine — 1 if my alliance owns the yellow-controlling toggle
  [18] cup_place_quality  — +1/−1/0 cup orientation benefit for this goal).
GOAL_FEATS 17→19; GOAL_BLOCK_END 209→227.
non_goal_dim = 610 - 9×19 = 439 (unchanged — new dims absorbed into goal block).
Fresh training run required (obs_dim mismatch with v9 checkpoints).

Components
----------
ObsEncoder
  - obs_dim (=610) → feat_dim (=128) per observation
  - non_goal_dim = 610 - 9×19 = 439; goal block = obs[:, 56:227]
  - Used internally by Policy (1× per obs) and Critic (2× then merged).

Policy (actor)
  - Decentralised: sees only one robot's 592-dim observation.
  - Heads: continuous (mean + log_std for left/right motors) + discrete (7).

CentralizedCritic
  - Sees both teammates' observations independently (via ObsEncoder),
    concatenates their encoded features, then maps to a scalar value.
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
# Observation layout constants — must match utils/observation_builder.py
# ─────────────────────────────────────────────────────────────────────────────
GOAL_OFFSET     = 56                          # start of goal slots
N_GOALS         = 9
GOAL_FEATS      = 19                          # v9.2: 17 + 2 (yellow_toggle_mine, cup_place_quality)
GOAL_BLOCK_END  = GOAL_OFFSET + N_GOALS * GOAL_FEATS   # 56 + 9×19 = 227
GOAL_EMBED_DIM  = 32                          # per-goal embedding size
GOAL_HIDDEN_DIM = 64                          # internal width of the per-goal MLP


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
# Observation encoder with goal attention (v7)
# ─────────────────────────────────────────────────────────────────────────────
class ObsEncoder(nn.Module):
    """
    Encodes a single (B, OBS_DIM) observation into a (B, feat_dim) feature vector.

    Architecture
    ------------
      non_goal  = obs[:, :56] ++ obs[:, 209:]              # (B, OBS_DIM - 153)
      goals     = obs[:, 56:209].view(B, 9, 17)             # (B, 9, 17)

      goal_emb  = goal_embed_mlp(goals)                     # (B, 9, GOAL_EMBED_DIM)
      query     = query_proj(non_goal)                      # (B, GOAL_EMBED_DIM)
      scores    = <goal_emb, query> / sqrt(d)               # (B, 9)
      weights   = softmax(scores, dim=-1)                   # (B, 9)
      attended  = sum_g weights[g] * goal_emb[g]            # (B, GOAL_EMBED_DIM)

      feats     = backbone( cat[non_goal, attended] )       # (B, feat_dim)

    With OBS_DIM=564: non_goal_dim = 564 - 153 = 411.
    The goal block [56:209] is fixed; the 10 v8 features appended at [554:564]
    land in the non_goal slice and require no architectural change.
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = ACTOR_HIDDEN
        self.obs_dim     = obs_dim
        self.non_goal_dim = obs_dim - N_GOALS * GOAL_FEATS    # e.g. 564 - 153 = 411

        # Per-goal embedding (shared across the 9 slots).
        self.goal_embed = nn.Sequential(
            nn.Linear(GOAL_FEATS, GOAL_HIDDEN_DIM),
            nn.LayerNorm(GOAL_HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(GOAL_HIDDEN_DIM, GOAL_EMBED_DIM),
        )

        # Query projection from non-goal context.
        self.query_proj = nn.Linear(self.non_goal_dim, GOAL_EMBED_DIM)

        # Main backbone consumes [non_goal, attended_goals].
        self.backbone = _mlp([self.non_goal_dim + GOAL_EMBED_DIM] + hidden)
        self.feat_dim = hidden[-1]

        # Init
        self.goal_embed.apply(lambda m: _init_weights(m, gain=math.sqrt(2)))
        _init_weights(self.query_proj, gain=math.sqrt(2))
        self.backbone.apply(lambda m: _init_weights(m, gain=math.sqrt(2)))

        self._inv_sqrt_d = 1.0 / math.sqrt(GOAL_EMBED_DIM)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs : (B, OBS_DIM)  →  (B, feat_dim)"""
        # Slice goal block out of the observation.
        pre  = obs[:, :GOAL_OFFSET]                          # (B, 56)
        gflat = obs[:, GOAL_OFFSET:GOAL_BLOCK_END]            # (B, 153)
        post = obs[:, GOAL_BLOCK_END:]                        # (B, 554 - 209 = 345)
        non_goal = torch.cat([pre, post], dim=-1)             # (B, non_goal_dim)

        # Per-goal embedding.
        goals = gflat.view(-1, N_GOALS, GOAL_FEATS)           # (B, 9, 17)
        goal_emb = self.goal_embed(goals)                     # (B, 9, GOAL_EMBED_DIM)

        # Query vector from the non-goal context.
        query = self.query_proj(non_goal)                     # (B, GOAL_EMBED_DIM)

        # Dot-product attention scores.
        # scores[b, g] = <goal_emb[b, g, :], query[b, :]> / sqrt(d)
        scores = torch.einsum("bgd,bd->bg", goal_emb, query) * self._inv_sqrt_d
        weights = F.softmax(scores, dim=-1)                   # (B, 9)

        # Weighted pool of goal embeddings.
        attended = torch.einsum("bg,bgd->bd", weights, goal_emb)   # (B, GOAL_EMBED_DIM)

        # Concatenate non-goal context with attended goal features.
        feats = self.backbone(torch.cat([non_goal, attended], dim=-1))
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# Policy (actor)
# ─────────────────────────────────────────────────────────────────────────────
class Policy(nn.Module):
    """
    Decentralised actor — uses only local 554-dim observation.

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

        self.encoder = ObsEncoder(obs_dim=obs_dim, hidden=hidden)
        feat_dim = self.encoder.feat_dim

        # Continuous: mean and log_std
        self.cont_mean    = nn.Linear(feat_dim, cont_dim)
        self.cont_log_std = nn.Linear(feat_dim, cont_dim)
        # Discrete: one logit per button
        self.disc_head    = nn.Linear(feat_dim, disc_dim)

        _init_weights(self.cont_mean,    gain=0.01)
        _init_weights(self.cont_log_std, gain=0.01)
        _init_weights(self.disc_head,    gain=0.01)

    def forward(self, obs: torch.Tensor):
        """obs : (B, OBS_DIM)  →  cont_mean, cont_log_std, disc_logits"""
        feats   = self.encoder(obs)
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
        obs          : (B, OBS_DIM) or (OBS_DIM,) — current observation(s)
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

    v7: each robot's observation is independently encoded with goal attention
    (using a critic-private ObsEncoder), then the two feature vectors are
    concatenated and mapped to a scalar value through a small head MLP.

    Input  : two observations (B, OBS_DIM) each
    Output : scalar value estimate → (B, 1)
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = CRITIC_HIDDEN

        self.encoder = ObsEncoder(obs_dim=obs_dim, hidden=hidden)
        feat_dim = self.encoder.feat_dim

        # Joint head: takes 2× encoder output → 1
        head_hidden = max(64, feat_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.ELU(),
            nn.Linear(head_hidden, 1),
        )

        self.head.apply(lambda m: _init_weights(m, gain=math.sqrt(2)))
        # Last layer: small init for stable early value estimates
        _init_weights(list(self.head.children())[-1], gain=1.0)

    def forward(self, obs1: torch.Tensor, obs2: torch.Tensor) -> torch.Tensor:
        """
        obs1, obs2 : (B, OBS_DIM) — observations of robot 0 and robot 1 (same alliance)
        Returns    : (B, 1)       — value estimates
        """
        f1 = self.encoder(obs1)
        f2 = self.encoder(obs2)
        joint = torch.cat([f1, f2], dim=-1)
        return self.head(joint)
