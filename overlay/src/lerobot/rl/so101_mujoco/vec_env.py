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

"""Process-parallel batch of `SO101PickPlaceEnv`s with auto-reset.

MuJoCo physics runs on the CPU, so envs are spread over worker processes (each steps its envs in
sequence). Workers come from a fork server that has already imported the env module (torch, lerobot:
~800 MB), so they share those pages instead of each importing their own copy, and they are not forked
from the trainer, which may hold a CUDA context. Each worker creates its own EGL context when rendering.
"""

import multiprocessing as mp
import os
from dataclasses import dataclass

import numpy as np

from .configs import SO101PickPlaceEnvConfig


@dataclass
class EpisodeStats:
    episode_return: float
    length: int
    success: bool


def _stack(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([o[key] for o in observations]) for key in observations[0]}


def _worker(remote, cfg: SO101PickPlaceEnvConfig, seeds: list[int], render: bool) -> None:
    import torch

    from .env import SO101PickPlaceEnv

    # One core per worker; torch is only used for image resizing here
    torch.set_num_threads(1)
    envs = [SO101PickPlaceEnv(cfg, seed=seed, render=render) for seed in seeds]
    returns = np.zeros(len(envs))
    lengths = np.zeros(len(envs), dtype=int)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                returns[:] = 0.0
                lengths[:] = 0
                remote.send(_stack([env.reset() for env in envs]))
            elif cmd == "step":
                observations, rewards, terminated, truncated, episodes = [], [], [], [], []
                for i, env in enumerate(envs):
                    obs, reward, term, trunc, info = env.step(data[i])
                    returns[i] += reward
                    lengths[i] += 1
                    if term or trunc:
                        episodes.append(EpisodeStats(float(returns[i]), int(lengths[i]), info["success"]))
                        returns[i] = 0.0
                        lengths[i] = 0
                        obs = env.reset()
                    observations.append(obs)
                    rewards.append(reward)
                    terminated.append(term)
                    truncated.append(trunc)
                remote.send(
                    (
                        _stack(observations),
                        np.array(rewards, dtype=np.float32),
                        np.array(terminated),
                        np.array(truncated),
                        episodes,
                    )
                )
            elif cmd == "close":
                break
    except KeyboardInterrupt:
        pass
    finally:
        for env in envs:
            env.close()
        remote.close()


class SubprocVecEnv:
    """`num_envs` envs over `num_workers` processes; finished envs are reset inside `step`.

    `step` returns the observation after the auto-reset, the reward and termination flags of the
    finished step, and the stats of the episodes that ended during it.
    """

    def __init__(
        self,
        cfg: SO101PickPlaceEnvConfig,
        num_envs: int,
        num_workers: int,
        seed: int = 0,
        render: bool = False,
    ):
        # Headless GPU rendering; must be set before the fork server starts and imports mujoco
        os.environ.setdefault("MUJOCO_GL", "egl")
        num_workers = max(1, min(num_workers, num_envs))
        self.num_envs = num_envs
        self.max_episode_steps = cfg.max_episode_steps
        splits = np.array_split(np.arange(num_envs), num_workers)
        self._slices = [slice(int(s[0]), int(s[-1]) + 1) for s in splits]

        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(["lerobot.rl.so101_mujoco.env"])
        self._remotes, self._processes = [], []
        for split in splits:
            parent, child = ctx.Pipe()
            seeds = [seed * 10_000 + int(i) for i in split]
            process = ctx.Process(target=_worker, args=(child, cfg, seeds, render), daemon=True)
            process.start()
            child.close()
            self._remotes.append(parent)
            self._processes.append(process)
        self._closed = False

    def reset(self) -> dict[str, np.ndarray]:
        for remote in self._remotes:
            remote.send(("reset", None))
        return self._concat([remote.recv() for remote in self._remotes])

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[EpisodeStats]]:
        for remote, sl in zip(self._remotes, self._slices, strict=True):
            remote.send(("step", actions[sl]))
        results = [remote.recv() for remote in self._remotes]
        obs = self._concat([r[0] for r in results])
        rewards = np.concatenate([r[1] for r in results])
        terminated = np.concatenate([r[2] for r in results])
        truncated = np.concatenate([r[3] for r in results])
        episodes = [ep for r in results for ep in r[4]]
        return obs, rewards, terminated, truncated, episodes

    @staticmethod
    def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def __enter__(self) -> "SubprocVecEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
