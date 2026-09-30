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

"""Evaluate a teacher (PPO) or student (distilled) checkpoint on the MuJoCo pick-and-place scene.

Examples:
```shell
# Success rate over 50 episodes
python -m lerobot.rl.so101_mujoco.eval --checkpoint outputs/so101_mujoco/distill/<run>/model_best.pt
# Watch it in the MuJoCo viewer (real time)
python -m lerobot.rl.so101_mujoco.eval --checkpoint <ckpt> --viewer true
# Save the student's camera view of the first episodes
python -m lerobot.rl.so101_mujoco.eval --checkpoint <ckpt> --video_path outputs/eval.mp4
```
The env settings (action space, cameras) are taken from the checkpoint; `--env.*` flags are ignored.
"""

import logging
import time

import draccus
import numpy as np
import torch

from lerobot.utils.utils import init_logging

from .configs import EvalConfig
from .env import SO101PickPlaceEnv
from .utils import STUDENT, EpisodeTracker, load_checkpoint, policy_actions, resolve_device
from .vec_env import SubprocVecEnv


def _batch(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v[None] for k, v in obs.items()}


def _eval_viewer(cfg: EvalConfig, kind, model, env_cfg, device) -> None:
    env = SO101PickPlaceEnv(env_cfg, seed=cfg.seed, render=kind == STUDENT, show_viewer=True)
    try:
        for episode in range(cfg.num_episodes):
            obs, done, ret, info = env.reset(), False, 0.0, {}
            while not done:
                start = time.perf_counter()
                action = policy_actions(kind, model, _batch(obs), device)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                ret += reward
                done = terminated or truncated
                time.sleep(max(0.0, 1.0 / env_cfg.fps - (time.perf_counter() - start)))
            logging.info(f"episode {episode}: return {ret:.2f}, success {info['success']}")
    finally:
        env.close()


def _eval_vec(cfg: EvalConfig, kind, model, env_cfg, device) -> None:
    render = kind == STUDENT or cfg.video_path is not None
    writer = None
    if cfg.video_path is not None:
        import imageio

        writer = imageio.get_writer(cfg.video_path, fps=env_cfg.fps)
    tracker = EpisodeTracker(window=cfg.num_episodes)
    video_done = 0
    with SubprocVecEnv(env_cfg, cfg.num_envs, cfg.num_workers, seed=cfg.seed, render=render) as envs:
        obs = envs.reset()
        # Only count episodes that started after the reset, `num_episodes` in total
        while tracker.total < cfg.num_episodes:
            if writer is not None and video_done < cfg.video_episodes:
                frame = np.concatenate([obs["front"][0], obs["wrist"][0]], axis=-1).transpose(1, 2, 0)
                writer.append_data(np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1))
            obs, _, terminated, truncated, episodes = envs.step(policy_actions(kind, model, obs, device))
            if terminated[0] or truncated[0]:
                video_done += 1
            tracker.add(episodes[: cfg.num_episodes - tracker.total])
    if writer is not None:
        writer.close()
        logging.info(f"Video saved to {cfg.video_path}")
    stats = tracker.summary()
    logging.info(
        f"{kind}: {tracker.total} episodes | success rate {stats['episode/success_rate']:.2%} | "
        f"return {stats['episode/return']:.2f} | length {stats['episode/length']:.1f}"
    )


@draccus.wrap()
def evaluate(cfg: EvalConfig) -> None:
    init_logging()
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required")
    device = resolve_device(cfg.device)
    kind, model, env_cfg = load_checkpoint(cfg.checkpoint, device)
    with torch.inference_mode():
        if cfg.viewer:
            _eval_viewer(cfg, kind, model, env_cfg, device)
        else:
            _eval_vec(cfg, kind, model, env_cfg, device)


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
