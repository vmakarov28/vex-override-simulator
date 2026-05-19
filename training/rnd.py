"""
training/rnd.py
Random Network Distillation (RND) for intrinsic motivation / anti-collapse.

This adds a curiosity bonus: the agent gets rewarded for visiting states
that are "surprising" to a fixed random network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from config.hyperparameters import RND_HIDDEN, RND_LR, RND_REWARD_SCALE


class RNDModule:
    def __init__(self, obs_dim: int, device: torch.device):
        self.device = device
        self.scale = RND_REWARD_SCALE

        # Fixed target network (never trained)
        self.target = self._build_network(obs_dim).to(device)
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor network (trained to match target)
        self.predictor = self._build_network(obs_dim).to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=RND_LR)

        self.update_count = 0

    def _build_network(self, obs_dim: int) -> nn.Sequential:
        layers = []
        prev = obs_dim
        for h in RND_HIDDEN:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        target_out = self.target(obs)
        pred_out = self.predictor(obs)

        error = (target_out - pred_out).pow(2).mean(dim=-1)
        return error * self.scale

    def update(self, obs: torch.Tensor):
        self.optimizer.zero_grad()
        target_out = self.target(obs).detach()
        pred_out = self.predictor(obs)
        loss = (target_out - pred_out).pow(2).mean()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1

    def should_update(self, step: int) -> bool:
        from config.hyperparameters import RND_UPDATE_EVERY
        return step % RND_UPDATE_EVERY == 0