#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Networks of the PPO teacher (state MLP actor-critic) and the vision student (CNN)."""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Categorical, Normal

from .env import NUM_GRIPPER_ACTIONS

# ImageNet statistics, as in the HIL-SERL policy's `dataset_stats` for both cameras
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for dim in hidden_dims:
        layers += [nn.Linear(in_dim, dim), nn.ELU()]
        in_dim = dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class EmpiricalNormalization(nn.Module):
    """Running mean / std normalization, updated explicitly with `update` (as in rsl_rl)."""

    def __init__(self, dim: int, eps: float = 1e-2, clip: float = 10.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var.sqrt() + self.eps)).clamp(-self.clip, self.clip)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        n = x.shape[0]
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
        self.count = total


class ActorCritic(nn.Module):
    """Gaussian EE delta + categorical gripper; actions are `[continuous..., gripper_index]`."""

    def __init__(
        self,
        obs_dim: int,
        num_continuous: int,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        init_noise_std: float,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.obs_normalizer = EmpiricalNormalization(obs_dim)
        self.actor = mlp(obs_dim, actor_hidden_dims, num_continuous + NUM_GRIPPER_ACTIONS)
        self.critic = mlp(obs_dim, critic_hidden_dims, 1)
        self.log_std = nn.Parameter(torch.full((num_continuous,), math.log(init_noise_std)))

    def distributions(self, obs: torch.Tensor) -> tuple[Normal, Categorical]:
        out = self.actor(self.obs_normalizer(obs))
        # The env clips actions to [-1, 1]; an unbounded mean drifts past the clip, where the
        # exploration noise is clipped away and the policy gets stuck (e.g. pinned at a workspace bound)
        mean, logits = torch.tanh(out[:, : self.num_continuous]), out[:, self.num_continuous :]
        return Normal(mean, self.log_std.exp().expand_as(mean)), Categorical(logits=logits)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.obs_normalizer(obs)).squeeze(-1)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions; returns (actions, log_prob, value)."""
        normal, categorical = self.distributions(obs)
        continuous = normal.sample()
        gripper = categorical.sample()
        log_prob = normal.log_prob(continuous).sum(-1) + categorical.log_prob(gripper)
        actions = torch.cat([continuous, gripper.unsqueeze(-1).float()], dim=-1)
        return actions, log_prob, self.value(obs)

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Normal, Categorical]:
        """Log-prob and entropy of stored actions, value, and the distributions (for the KL)."""
        normal, categorical = self.distributions(obs)
        continuous, gripper = actions[:, : self.num_continuous], actions[:, -1].long()
        log_prob = normal.log_prob(continuous).sum(-1) + categorical.log_prob(gripper)
        entropy = normal.entropy().sum(-1) + categorical.entropy()
        return log_prob, entropy, self.value(obs), normal, categorical

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action: clipped mean and most likely gripper command."""
        normal, categorical = self.distributions(obs)
        return torch.cat([normal.mean.clamp(-1, 1), categorical.probs.argmax(-1, keepdim=True).float()], -1)


def expand_actor_critic_inputs(state_dict: dict, new_obs_dim: int) -> dict:
    """Adapt an `ActorCritic` state dict to observations with extra features appended: the new inputs
    get zero weights (and identity normalization), so the policy and value start out unchanged."""
    state_dict = dict(state_dict)
    old_obs_dim = state_dict["obs_normalizer.mean"].shape[0]
    extra = new_obs_dim - old_obs_dim
    if extra <= 0:
        return state_dict
    for key, fill in (("obs_normalizer.mean", 0.0), ("obs_normalizer.var", 1.0)):
        state_dict[key] = torch.cat([state_dict[key], state_dict[key].new_full((extra,), fill)])
    for key in ("actor.0.weight", "critic.0.weight"):
        weight = state_dict[key]
        state_dict[key] = torch.cat([weight, weight.new_zeros(weight.shape[0], extra)], dim=1)
    return state_dict


class RandomShiftsAug(nn.Module):
    """DrQ-v2 random shift: replicate-pad by `pad` pixels and crop back at a random offset."""

    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad == 0:
            return x
        n, _, h, w = x.shape
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        padded_h, padded_w = h + 2 * self.pad, w + 2 * self.pad

        def coords(size: int, padded: int) -> torch.Tensor:
            eps = 1.0 / padded
            return torch.linspace(-1.0 + eps, 1.0 - eps, padded, device=x.device)[:size]

        grid_y, grid_x = torch.meshgrid(coords(h, padded_h), coords(w, padded_w), indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= torch.tensor([2.0 / padded_w, 2.0 / padded_h], device=x.device, dtype=x.dtype)
        return F.grid_sample(x, base_grid + shift, padding_mode="zeros", align_corners=False)


class ImageEncoder(nn.Module):
    """Small conv encoder (DrQ-style) for a uint8 (B, 3, H, W) image."""

    def __init__(self, image_size: tuple[int, int], feature_dim: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.convs(torch.zeros(1, 3, *image_size)).shape[-1]
        self.head = nn.Sequential(nn.Linear(flat_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.register_buffer("mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image.float() / 255.0 - self.mean) / self.std
        return self.head(self.convs(x))


class VisionStudent(nn.Module):
    """Front + wrist images and the 18-dim `observation.state` -> teacher-format action.

    Outputs the EE delta mean (tanh, in [-1, 1]) and the gripper logits.
    """

    def __init__(
        self,
        state_dim: int,
        num_continuous: int,
        image_size: tuple[int, int],
        feature_dim: int,
        head_hidden_dims: list[int],
        image_shift_pad: int = 0,
        masked_state_dims: list[int] | None = None,
        wrist_image_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.state_normalizer = EmpiricalNormalization(state_dim)
        # `image_size` is the front camera's, and the wrist camera's too unless `wrist_image_size` is given
        self.front_encoder = ImageEncoder(image_size, feature_dim)
        self.wrist_encoder = ImageEncoder(wrist_image_size or image_size, feature_dim)
        self.augment = RandomShiftsAug(image_shift_pad)
        self.head = mlp(2 * feature_dim + state_dim, head_hidden_dims, num_continuous + NUM_GRIPPER_ACTIONS)
        # State features hidden from the student (e.g. the emulated motor currents); part of the network
        # so that deployment hides them too. Not saved: it comes from the constructor arguments
        mask = torch.ones(state_dim)
        if masked_state_dims:
            mask[list(masked_state_dims)] = 0.0
        self.register_buffer("state_mask", mask, persistent=False)

    def forward(
        self, front: torch.Tensor, wrist: torch.Tensor, state: torch.Tensor, augment: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if augment:
            front = self.augment(front.float())
            wrist = self.augment(wrist.float())
        features = torch.cat(
            [self.front_encoder(front), self.wrist_encoder(wrist), self.state_normalizer(state) * self.state_mask],
            dim=-1,
        )
        out = self.head(features)
        return torch.tanh(out[:, : self.num_continuous]), out[:, self.num_continuous :]

    @torch.no_grad()
    def act_inference(self, front: torch.Tensor, wrist: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        continuous, logits = self(front, wrist, state)
        return torch.cat([continuous, logits.argmax(-1, keepdim=True).float()], dim=-1)
