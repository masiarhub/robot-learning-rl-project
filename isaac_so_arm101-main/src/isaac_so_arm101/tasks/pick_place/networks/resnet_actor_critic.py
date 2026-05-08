# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetActorCritic(nn.Module):
    """Actor-Critic with ResNet18 vision backbone.
    
    Observation layout (concatenated by the manager):
        [0:C*H*W]          → wrist camera image (flattened, normalised to [0,1])
        [C*H*W : C*H*W+3]  → bowl bottom centre in world frame  (x, y, z)
        [C*H*W+3 : ...]    → joint_pos, joint_vel, last_action
    """
    """RSL-RL compatible actor-critic with a ResNet18 vision encoder."""


    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        image_shape: tuple[int, int, int] = (3, 72, 128),  # C, H, W
        resnet_embed_dim: int = 256,
        actor_hidden_dims: list[int] = [256, 128, 64],
        critic_hidden_dims: list[int] = [256, 128, 64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **_: dict,
    ):
        super().__init__()

        self.is_recurrent = False
        self.image_shape = image_shape
        self.image_dim = image_shape[0] * image_shape[1] * image_shape[2]
        if num_actor_obs < self.image_dim:
            raise ValueError(
                f"num_actor_obs={num_actor_obs} is smaller than image_dim={self.image_dim}. "
                "Check observation term ordering and camera shape."
            )
        self.actor_state_dim = num_actor_obs - self.image_dim
        self.critic_state_dim = num_critic_obs - self.image_dim

        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        # ── Vision backbone ──────────────────────────────────────────────
        backbone = resnet18(weights=None)
        # Replace first conv to accept any channel count (3 here = RGB)
        backbone.conv1 = nn.Conv2d(
            image_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Drop final classification head; keep up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        self.vision_proj = nn.Sequential(
            nn.Flatten(),          # → (B, 512)
            nn.Linear(512, resnet_embed_dim),
            act(),
        )

        # ── Fusion: vision embedding + scalar state ───────────────────────
        actor_fused_dim = resnet_embed_dim + self.actor_state_dim
        critic_fused_dim = resnet_embed_dim + self.critic_state_dim

        def mlp(dims):
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i + 1]), act()]
            return nn.Sequential(*layers)

        self.actor = nn.Sequential(
            mlp([actor_fused_dim] + actor_hidden_dims),
            nn.Linear(actor_hidden_dims[-1], num_actions),
        )
        self.critic = nn.Sequential(
            mlp([critic_fused_dim] + critic_hidden_dims),
            nn.Linear(critic_hidden_dims[-1], 1),
        )

        # Learnable action noise parameter expected by RSL-RL.
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        self._init_weights()

    # ── helpers ──────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def _split_obs(self, obs: torch.Tensor):
        """Split flat observation into image tensor and state vector."""
        img_flat = obs[:, : self.image_dim]                    # (B, C*H*W)
        state = obs[:, self.image_dim :]                       # (B, state_dim)

        C, H, W = self.image_shape
        img = img_flat.view(-1, C, H, W)                       # (B, C, H, W)
        # Observations are expected in [0, 1]. Keep a safe fallback for uint8-like ranges.
        if img.max() > 1.5:
            img = img / 255.0
        return img, state

    def _embed(self, obs: torch.Tensor) -> torch.Tensor:
        img, state = self._split_obs(obs)
        vis = self.vision_proj(self.backbone(img))             # (B, resnet_embed_dim)
        return torch.cat([vis, state], dim=-1)                 # (B, fused_dim)

    # ── RSL-RL actor-critic API ───────────────────────────────────────────

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(self._embed(observations))
        self.distribution = torch.distributions.Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs):
        del kwargs
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        return self.actor(self._embed(observations))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        del kwargs
        return self.critic(self._embed(critic_observations)).squeeze(-1)