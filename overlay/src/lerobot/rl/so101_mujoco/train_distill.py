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

"""Distill a PPO teacher into a vision student (front + wrist cameras + 18-dim state) with DAgger.

Each iteration the envs are driven by the student (mixed with the teacher early on, see
`teacher_mix_start`), the teacher labels every visited state, the labeled samples are aggregated in
a replay buffer, and the student is trained on it: MSE on the EE delta + cross-entropy on the
gripper command. Driving with the student is what lets it learn to recover from its own mistakes.

Example:
```shell
python -m lerobot.rl.so101_mujoco.train_distill \
    --teacher_checkpoint outputs/so101_mujoco/ppo/<run>/model_best.pt --num_envs 16 --num_workers 8
```
"""

import logging
import time

import draccus
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.utils.utils import init_logging

from .configs import TrainDistillConfig, resolve_path
from .networks import VisionStudent
from .utils import (
    STUDENT,
    TEACHER,
    EpisodeTracker,
    MetricsLogger,
    load_checkpoint,
    make_run_dir,
    resolve_device,
    seed_everything,
)
from .vec_env import SubprocVecEnv


class DAggerBuffer:
    """Ring buffer of (front, wrist, state) -> teacher action samples, kept on CPU."""

    def __init__(
        self,
        capacity: int,
        front_shape: tuple[int, ...],
        wrist_shape: tuple[int, ...],
        state_dim: int,
        num_continuous: int,
    ):
        self.capacity = capacity
        self.front = np.zeros((capacity, *front_shape), dtype=np.uint8)
        self.wrist = np.zeros((capacity, *wrist_shape), dtype=np.uint8)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.continuous = np.zeros((capacity, num_continuous), dtype=np.float32)
        self.gripper = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self._ptr = 0

    def add(self, obs: dict[str, np.ndarray], teacher_actions: np.ndarray) -> None:
        n = teacher_actions.shape[0]
        idx = (self._ptr + np.arange(n)) % self.capacity
        self.front[idx] = obs["front"]
        self.wrist[idx] = obs["wrist"]
        self.state[idx] = obs["state"]
        self.continuous[idx] = teacher_actions[:, :-1]
        self.gripper[idx] = teacher_actions[:, -1].astype(np.int64)
        self._ptr = (self._ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return tuple(
            torch.from_numpy(a[idx]).to(device, non_blocking=True)
            for a in (self.front, self.wrist, self.state, self.continuous, self.gripper)
        )


def save_checkpoint(path, student, model_kwargs, cfg: TrainDistillConfig, iteration: int) -> None:
    torch.save(
        {
            "type": STUDENT,
            "model": student.state_dict(),
            "model_kwargs": model_kwargs,
            "iteration": iteration,
            "env_cfg": draccus.encode(cfg.env),
            "teacher_checkpoint": cfg.teacher_checkpoint,
        },
        path,
    )


@draccus.wrap()
def train(cfg: TrainDistillConfig) -> None:
    init_logging()
    if not cfg.teacher_checkpoint:
        raise ValueError("--teacher_checkpoint is required (a PPO checkpoint from train_ppo)")
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    kind, teacher, teacher_env_cfg = load_checkpoint(cfg.teacher_checkpoint, device)
    if kind != TEACHER:
        raise ValueError(f"{cfg.teacher_checkpoint} is a {kind} checkpoint, expected a PPO teacher")
    if teacher_env_cfg.num_continuous_actions != cfg.env.num_continuous_actions:
        raise ValueError("The teacher was trained with a different action space (env.use_wrist_roll)")

    run_dir = make_run_dir(cfg.output_dir, cfg)
    logging.info(f"Run directory: {run_dir}")
    metrics = MetricsLogger(run_dir, cfg, cfg.wandb, cfg.wandb_project)
    dcfg = cfg.distill

    envs = SubprocVecEnv(cfg.env, cfg.num_envs, cfg.num_workers, seed=cfg.seed, render=True)
    try:
        obs = envs.reset()
        num_continuous = cfg.env.num_continuous_actions
        model_kwargs = {
            "state_dim": obs["state"].shape[-1],
            "num_continuous": num_continuous,
            "image_size": tuple(obs["front"].shape[-2:]),
            "wrist_image_size": tuple(obs["wrist"].shape[-2:]),
            "feature_dim": dcfg.image_feature_dim,
            "head_hidden_dims": list(dcfg.head_hidden_dims),
            "image_shift_pad": dcfg.image_shift_pad,
            # The emulated currents are the last 6 of the 18 state features (positions, velocities, currents)
            "masked_state_dims": list(range(12, 18)) if dcfg.drop_current else None,
        }
        student = VisionStudent(**model_kwargs).to(device)
        if cfg.init_checkpoint:
            init = torch.load(resolve_path(cfg.init_checkpoint), map_location=device, weights_only=False)
            # The state mask is not a weight, so a student can be continued with another mask
            same_weights = {k: v for k, v in init["model_kwargs"].items() if k != "masked_state_dims"} == {
                k: v for k, v in model_kwargs.items() if k != "masked_state_dims"
            }
            if init["type"] != STUDENT:
                raise ValueError(f"{cfg.init_checkpoint} is not a student checkpoint")
            if same_weights:
                student.load_state_dict(init["model"])
                logging.info(f"Student initialized from {cfg.init_checkpoint}")
            else:
                # Another image size: the convolutions, the state normalizer and the head carry over,
                # the image projections whose input size changed start afresh
                own = student.state_dict()
                kept = {k: v for k, v in init["model"].items() if k in own and v.shape == own[k].shape}
                student.load_state_dict(kept, strict=False)
                fresh = sorted(set(own) - set(kept))
                logging.info(f"Student partially initialized from {cfg.init_checkpoint}; new: {fresh}")
        optimizer = torch.optim.Adam(student.parameters(), lr=dcfg.learning_rate)
        buffer = DAggerBuffer(
            dcfg.buffer_capacity,
            obs["front"].shape[1:],
            obs["wrist"].shape[1:],
            obs["state"].shape[-1],
            num_continuous,
        )
        tracker = EpisodeTracker(window=50)
        best_success = -1.0
        rng = np.random.default_rng(cfg.seed)

        for iteration in range(cfg.max_iterations):
            mix = dcfg.teacher_mix_start * max(0.0, 1.0 - iteration / max(1, dcfg.teacher_mix_decay_iterations))
            start = time.perf_counter()
            student.eval()
            for _ in range(dcfg.num_steps_per_env):
                teacher_actions = teacher.act_inference(torch.from_numpy(obs["teacher"]).to(device)).cpu().numpy()
                student_state = torch.from_numpy(obs["state"]).to(device)
                student.state_normalizer.update(student_state)
                student_actions = (
                    student.act_inference(
                        torch.from_numpy(obs["front"]).to(device),
                        torch.from_numpy(obs["wrist"]).to(device),
                        student_state,
                    )
                    .cpu()
                    .numpy()
                )
                buffer.add(obs, teacher_actions)
                use_teacher = rng.random(cfg.num_envs) < mix
                actions = np.where(use_teacher[:, None], teacher_actions, student_actions)
                obs, _, _, _, episodes = envs.step(actions)
                # Once the mix reaches 0 these are pure student episodes
                tracker.add(episodes)
            collect_time = time.perf_counter() - start

            start = time.perf_counter()
            student.train()
            mse_total, ce_total, acc_total = 0.0, 0.0, 0.0
            for _ in range(dcfg.updates_per_iteration):
                front, wrist, state, target_continuous, target_gripper = buffer.sample(dcfg.batch_size, device)
                continuous, logits = student(front, wrist, state, augment=True)
                mse = F.mse_loss(continuous, target_continuous)
                ce = F.cross_entropy(logits, target_gripper)
                loss = mse + dcfg.gripper_loss_coef * ce
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), dcfg.max_grad_norm)
                optimizer.step()
                mse_total += mse.item()
                ce_total += ce.item()
                acc_total += (logits.argmax(-1) == target_gripper).float().mean().item()
            learn_time = time.perf_counter() - start

            n = dcfg.updates_per_iteration
            episode_stats = tracker.summary()
            log = {
                **episode_stats,
                "loss/continuous_mse": mse_total / n,
                "loss/gripper_ce": ce_total / n,
                "train/gripper_accuracy": acc_total / n,
                "train/teacher_mix": mix,
                "train/buffer_size": float(buffer.size),
                "perf/collect_s": collect_time,
                "perf/learn_s": learn_time,
            }
            metrics.log(log, iteration)
            logging.info(
                f"it {iteration:4d} | mix {mix:.2f} | buffer {buffer.size:6d} | mse {log['loss/continuous_mse']:.4f} | "
                f"grip acc {log['train/gripper_accuracy']:.3f} | "
                f"success {episode_stats.get('episode/success_rate', float('nan')):.2f} | "
                f"collect {collect_time:.1f}s learn {learn_time:.1f}s"
            )

            if iteration % cfg.save_interval == 0 or iteration == cfg.max_iterations - 1:
                save_checkpoint(run_dir / f"model_{iteration}.pt", student, model_kwargs, cfg, iteration)
            # Best = rolling success of pure-student episodes: past the mixing phase, over a full window
            # (a partial one, e.g. right after starting, can be a lucky 100%). Still a noisy estimate;
            # compare checkpoints with `eval` on a few hundred episodes before picking one.
            success = episode_stats.get("episode/success_rate", -1.0)
            pure_student = iteration >= dcfg.teacher_mix_decay_iterations + 10 or dcfg.teacher_mix_start == 0
            full_window = len(tracker.successes) == tracker.successes.maxlen
            if pure_student and full_window and success > best_success:
                best_success = success
                save_checkpoint(run_dir / "model_best.pt", student, model_kwargs, cfg, iteration)
    finally:
        envs.close()
        metrics.close()
    logging.info(f"Done. Best student success rate {best_success:.2f}; checkpoints in {run_dir}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
