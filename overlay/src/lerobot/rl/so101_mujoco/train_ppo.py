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

"""Train the privileged-state PPO teacher on the MuJoCo SO101 pick-and-place scene.

Example:
```shell
python -m lerobot.rl.so101_mujoco.train_ppo --num_envs 32 --num_workers 8 --max_iterations 3000
```
Checkpoints (`model_<it>.pt`, `model_best.pt` by rolling success rate) and `metrics.jsonl` go to
`output_dir/<timestamp>/`.
"""

import logging
import time

import draccus
import numpy as np
import torch
from torch.distributions import kl_divergence

from lerobot.utils.utils import init_logging

from .configs import TrainPPOConfig, resolve_path
from .networks import ActorCritic, expand_actor_critic_inputs
from .utils import (
    TEACHER,
    EpisodeTracker,
    MetricsLogger,
    make_run_dir,
    resolve_device,
    seed_everything,
)
from .vec_env import SubprocVecEnv


class RolloutStorage:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        shape = (num_steps, num_envs)
        self.obs = torch.zeros(*shape, obs_dim, device=device)
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.log_probs = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.dones = torch.zeros(shape, device=device)
        self.returns = torch.zeros(shape, device=device)
        self.advantages = torch.zeros(shape, device=device)

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        advantage = torch.zeros_like(last_values)
        for t in reversed(range(self.rewards.shape[0])):
            next_values = last_values if t == self.rewards.shape[0] - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            advantage = delta + gamma * lam * not_done * advantage
            self.advantages[t] = advantage
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    storage: RolloutStorage,
    cfg: TrainPPOConfig,
) -> dict[str, float]:
    ppo = cfg.ppo
    obs = storage.obs.flatten(0, 1)
    actions = storage.actions.flatten(0, 1)
    old_log_probs = storage.log_probs.flatten(0, 1)
    old_values = storage.values.flatten(0, 1)
    returns = storage.returns.flatten(0, 1)
    advantages = storage.advantages.flatten(0, 1)
    with torch.no_grad():
        old_normal, old_categorical = policy.distributions(obs)

    batch_size = obs.shape[0]
    mini_batch_size = batch_size // ppo.num_mini_batches
    stats = {"loss/surrogate": 0.0, "loss/value": 0.0, "loss/entropy": 0.0, "loss/kl": 0.0}
    num_updates = 0
    for _ in range(ppo.num_learning_epochs):
        permutation = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size - mini_batch_size + 1, mini_batch_size):
            idx = permutation[start : start + mini_batch_size]
            log_prob, entropy, value, normal, categorical = policy.evaluate(obs[idx], actions[idx])

            with torch.no_grad():
                kl = (
                    kl_divergence(
                        torch.distributions.Normal(old_normal.mean[idx], old_normal.stddev[idx]), normal
                    ).sum(-1)
                    + kl_divergence(torch.distributions.Categorical(logits=old_categorical.logits[idx]), categorical)
                ).mean()
            # Adaptive learning rate on the KL to the rollout policy (rsl_rl "adaptive" schedule)
            if ppo.desired_kl is not None:
                lr = optimizer.param_groups[0]["lr"]
                if kl > 2.0 * ppo.desired_kl:
                    lr = max(1e-5, lr / 1.5)
                elif 0.0 < kl < 0.5 * ppo.desired_kl:
                    lr = min(1e-2, lr * 1.5)
                for group in optimizer.param_groups:
                    group["lr"] = lr

            ratio = torch.exp(log_prob - old_log_probs[idx])
            surrogate = -torch.min(
                advantages[idx] * ratio,
                advantages[idx] * ratio.clamp(1.0 - ppo.clip_param, 1.0 + ppo.clip_param),
            ).mean()
            value_clipped = old_values[idx] + (value - old_values[idx]).clamp(-ppo.clip_param, ppo.clip_param)
            value_loss = torch.max((value - returns[idx]) ** 2, (value_clipped - returns[idx]) ** 2).mean()
            loss = surrogate + ppo.value_loss_coef * value_loss - ppo.entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo.max_grad_norm)
            optimizer.step()

            stats["loss/surrogate"] += surrogate.item()
            stats["loss/value"] += value_loss.item()
            stats["loss/entropy"] += entropy.mean().item()
            stats["loss/kl"] += kl.item()
            num_updates += 1
    return {k: v / num_updates for k, v in stats.items()}


def save_checkpoint(path, policy, optimizer, model_kwargs, cfg: TrainPPOConfig, iteration: int) -> None:
    torch.save(
        {
            "type": TEACHER,
            "model": policy.state_dict(),
            "model_kwargs": model_kwargs,
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "env_cfg": draccus.encode(cfg.env),
        },
        path,
    )


@draccus.wrap()
def train(cfg: TrainPPOConfig) -> None:
    init_logging()
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = make_run_dir(cfg.output_dir, cfg)
    logging.info(f"Run directory: {run_dir}")
    metrics = MetricsLogger(run_dir, cfg, cfg.wandb, cfg.wandb_project)

    envs = SubprocVecEnv(cfg.env, cfg.num_envs, cfg.num_workers, seed=cfg.seed, render=False)
    try:
        obs = torch.from_numpy(envs.reset()["teacher"]).to(device)
        num_continuous = cfg.env.num_continuous_actions
        model_kwargs = {
            "obs_dim": obs.shape[-1],
            "num_continuous": num_continuous,
            "actor_hidden_dims": list(cfg.ppo.actor_hidden_dims),
            "critic_hidden_dims": list(cfg.ppo.critic_hidden_dims),
            "init_noise_std": cfg.ppo.init_noise_std,
        }
        policy = ActorCritic(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.learning_rate)
        start_iteration = 0
        if cfg.resume:
            ckpt = torch.load(resolve_path(cfg.resume), map_location=device, weights_only=False)
            old_obs_dim = ckpt["model_kwargs"]["obs_dim"]
            if old_obs_dim == obs.shape[-1]:
                policy.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
            else:
                # New observation features (e.g. the pending actions of an actuation delay): start from
                # the same policy with zero weights on them, and a fresh optimizer for the new shapes
                policy.load_state_dict(expand_actor_critic_inputs(ckpt["model"], obs.shape[-1]))
                logging.info(f"Expanded the checkpoint's observations from {old_obs_dim} to {obs.shape[-1]}")
            start_iteration = ckpt["iteration"] + 1
            logging.info(f"Resumed from {cfg.resume} at iteration {start_iteration}")

        storage = RolloutStorage(
            cfg.ppo.num_steps_per_env, cfg.num_envs, obs.shape[-1], num_continuous + 1, device
        )
        tracker = EpisodeTracker()
        best_success = -1.0
        total_steps = 0

        for iteration in range(start_iteration, cfg.max_iterations):
            start = time.perf_counter()
            with torch.inference_mode():
                for t in range(cfg.ppo.num_steps_per_env):
                    policy.obs_normalizer.update(obs)
                    actions, log_probs, values = policy.act(obs)
                    next_obs, rewards, terminated, truncated, episodes = envs.step(actions.cpu().numpy())
                    rewards = torch.from_numpy(rewards).to(device)
                    truncated_t = torch.from_numpy(truncated).to(device).float()
                    # Bootstrap time-outs with the value estimate instead of treating them as terminal
                    rewards = rewards + cfg.ppo.gamma * values * truncated_t
                    storage.obs[t] = obs
                    storage.actions[t] = actions
                    storage.log_probs[t] = log_probs
                    storage.values[t] = values
                    storage.rewards[t] = rewards
                    storage.dones[t] = torch.from_numpy(terminated | truncated).to(device).float()
                    tracker.add(episodes)
                    obs = torch.from_numpy(next_obs["teacher"]).to(device)
                collect_time = time.perf_counter() - start
                storage.compute_returns(policy.value(obs), cfg.ppo.gamma, cfg.ppo.lam)

            update_start = time.perf_counter()
            losses = ppo_update(policy, optimizer, storage, cfg)
            learn_time = time.perf_counter() - update_start
            total_steps += cfg.ppo.num_steps_per_env * cfg.num_envs

            episode_stats = tracker.summary()
            log = {
                **episode_stats,
                **losses,
                "train/step_reward": storage.rewards.mean().item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/noise_std": policy.log_std.exp().mean().item(),
                "perf/fps": cfg.ppo.num_steps_per_env * cfg.num_envs / (collect_time + learn_time),
                "perf/collect_s": collect_time,
                "perf/learn_s": learn_time,
                "train/total_steps": float(total_steps),
            }
            metrics.log(log, iteration)
            logging.info(
                f"it {iteration:5d} | steps {total_steps:9d} | fps {log['perf/fps']:6.0f} | "
                f"return {episode_stats.get('episode/return', float('nan')):7.2f} | "
                f"success {episode_stats.get('episode/success_rate', float('nan')):.2f} | "
                f"len {episode_stats.get('episode/length', float('nan')):5.1f} | "
                f"kl {losses['loss/kl']:.4f} | lr {log['train/lr']:.1e} | std {log['train/noise_std']:.2f}"
            )

            if iteration % cfg.save_interval == 0 or iteration == cfg.max_iterations - 1:
                save_checkpoint(run_dir / f"model_{iteration}.pt", policy, optimizer, model_kwargs, cfg, iteration)
            # Best = highest rolling success rate once the window holds enough episodes
            success = episode_stats.get("episode/success_rate", -1.0)
            if len(tracker.successes) >= 50 and success > best_success:
                best_success = success
                save_checkpoint(run_dir / "model_best.pt", policy, optimizer, model_kwargs, cfg, iteration)
    finally:
        envs.close()
        metrics.close()
    logging.info(f"Done. Best rolling success rate {best_success:.2f}; checkpoints in {run_dir}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
