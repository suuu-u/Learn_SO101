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

"""Run directories, metric logging and checkpoint loading shared by the train / eval scripts."""

import dataclasses
import json
import logging
import random
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch

from .configs import SO101PickPlaceEnvConfig, resolve_path
from .networks import ActorCritic, VisionStudent
from .vec_env import EpisodeStats

TEACHER = "ppo_teacher"
STUDENT = "vision_student"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA is not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device)


def make_run_dir(output_dir: str, cfg: Any) -> Path:
    run_dir = Path(output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w") as f:
        draccus.dump(cfg, f)
    return run_dir


class EpisodeTracker:
    """Rolling window over the last finished episodes."""

    def __init__(self, window: int = 100):
        self.returns: deque[float] = deque(maxlen=window)
        self.lengths: deque[int] = deque(maxlen=window)
        self.successes: deque[float] = deque(maxlen=window)
        self.total = 0

    def add(self, episodes: list[EpisodeStats]) -> None:
        for ep in episodes:
            self.returns.append(ep.episode_return)
            self.lengths.append(ep.length)
            self.successes.append(float(ep.success))
            self.total += 1

    def summary(self) -> dict[str, float]:
        if not self.returns:
            return {}
        return {
            "episode/return": float(np.mean(self.returns)),
            "episode/length": float(np.mean(self.lengths)),
            "episode/success_rate": float(np.mean(self.successes)),
            "episode/count": float(self.total),
        }


class MetricsLogger:
    """Appends metrics to `metrics.jsonl` in the run dir, and to wandb when enabled."""

    def __init__(self, run_dir: Path, cfg: Any, use_wandb: bool, project: str):
        self._file = open(run_dir / "metrics.jsonl", "a")  # noqa: SIM115
        self._wandb = None
        if use_wandb:
            import wandb

            self._wandb = wandb.init(project=project, name=run_dir.name, config=asdict(cfg), dir=str(run_dir))

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._file.write(json.dumps({"iteration": step, **metrics}) + "\n")
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._file.close()
        if self._wandb is not None:
            self._wandb.finish()


def _drop_unknown_fields(cls: type, raw: dict) -> dict:
    """Keep only the fields `cls` (a config dataclass) still has, recursively, so checkpoints saved
    before a config option was removed keep loading (the removed options only shaped the reward)."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    kept = {}
    for key, value in raw.items():
        if key not in known:
            logging.debug(f"Ignoring removed config field {cls.__name__}.{key} of the checkpoint")
            continue
        field_type = known[key].type
        if isinstance(value, dict) and dataclasses.is_dataclass(field_type):
            value = _drop_unknown_fields(field_type, value)
        kept[key] = value
    return kept


def load_checkpoint(path: str, device: torch.device) -> tuple[str, torch.nn.Module, SO101PickPlaceEnvConfig]:
    """Rebuild the teacher or student policy stored in a checkpoint, with its env config.

    A relative `path` is looked up in the working directory, then in the repository root.
    """
    ckpt = torch.load(resolve_path(path), map_location=device, weights_only=False)
    raw_env_cfg = _drop_unknown_fields(SO101PickPlaceEnvConfig, ckpt["env_cfg"])
    env_cfg = draccus.decode(SO101PickPlaceEnvConfig, raw_env_cfg)
    if ckpt["type"] == TEACHER:
        model: torch.nn.Module = ActorCritic(**ckpt["model_kwargs"])
    elif ckpt["type"] == STUDENT:
        model = VisionStudent(**ckpt["model_kwargs"])
    else:
        raise ValueError(f"Unknown checkpoint type {ckpt['type']!r} in {path}")
    model.load_state_dict(ckpt["model"])
    return ckpt["type"], model.to(device).eval(), env_cfg


def policy_actions(
    kind: str, model: torch.nn.Module, obs: dict[str, np.ndarray], device: torch.device
) -> np.ndarray:
    """Deterministic batched actions of a loaded teacher / student for a vec-env observation."""
    if kind == TEACHER:
        actions = model.act_inference(torch.from_numpy(obs["teacher"]).to(device))
    else:
        actions = model.act_inference(
            torch.from_numpy(obs["front"]).to(device),
            torch.from_numpy(obs["wrist"]).to(device),
            torch.from_numpy(obs["state"]).to(device),
        )
    return actions.cpu().numpy()
