# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a HIL-SERL checkpoint on the real robot or in simulation.

Runs the policy with the exact environment and processing pipeline used by the actor during
training (cropping, IK, bounds, reward classifier, ...), without a learner and without updating
the weights. Success is decided by the reward classifier / simulator (or the success key). Pressing
Space still lets you take over for safety; such episodes are reported separately.

Each episode is also classified by the furthest stage it reached: not grasped, grasped, lifted
(from the staged reward potential, when configured) or success. In simulation the true cube
height is checked too, so a "lifted" that did not really lift the cube shows up as "fake lift".

Examples:

```shell
# watch it (real robot, or sim with the MuJoCo window)
python -m lerobot.rl.eval_policy \
    --policy_path=outputs/train/<run>/checkpoints/last/pretrained_model --n_episodes=10

# fast statistics in simulation: no window, no leader arm, no real-time pacing
python -m lerobot.rl.eval_policy \
    --policy_path=outputs/train/<run>/checkpoints/0010000/pretrained_model --n_episodes=50 --headless=true
```
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.robots import make_robot_from_config, so101_sim, so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from .gym_manipulator import (
    RobotEnv,
    make_processors,
    make_robot_env,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from .train_rl import TrainRLServerPipelineConfig


@dataclass
class EvalRLConfig:
    # Checkpoint's pretrained_model directory (contains config.json, model.safetensors, train_config.json)
    policy_path: str
    n_episodes: int = 10
    # Use the mean action instead of sampling: evaluates the learned behaviour without exploration noise
    deterministic: bool = True
    # Simulation only: no viewer, no teleop device and no real-time pacing (fast statistics)
    headless: bool = False
    # Seed of the simulated scene (cube positions), so checkpoints can be compared on the same episodes
    seed: int | None = 0
    # A cube center higher than this (m) counts as really lifted (simulation only)
    lift_check_height: float = 0.03


# Furthest stage first
STAGES = ["SUCCESS", "lifted", "fake lift", "grasped", "not grasped"]


class _NoTeleop:
    """Teleop stand-in for headless evaluation: never intervenes (any action type is ignored)."""

    is_connected = True

    def __init__(self, motor_names: list[str]):
        self._motor_names = motor_names

    def get_action(self) -> dict[str, float]:
        return {f"{m}.pos": 0.0 for m in self._motor_names}

    def get_teleop_events(self) -> dict:
        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def send_feedback(self, *args, **kwargs) -> None:
        pass

    def enable_torque(self) -> None:
        pass

    def disable_torque(self) -> None:
        pass

    def disconnect(self) -> None:
        pass


@torch.no_grad()
def select_eval_action(policy, batch: dict[str, torch.Tensor], deterministic: bool) -> torch.Tensor:
    """Same as ``GaussianActorPolicy.select_action``, optionally using the distribution mean."""
    if not deterministic:
        return policy.select_action(batch)

    features = None
    if policy.shared_encoder and policy.actor.encoder.has_images:
        features = policy.actor.encoder.get_cached_image_features(batch)
    _, _, means = policy.actor(batch, features)
    actions = torch.tanh(means) * policy.actor.action_scale

    if policy.config.num_discrete_actions is not None:
        if policy.discrete_critic is not None:
            q_values = policy.discrete_critic(batch, features)
            discrete = torch.argmax(q_values, dim=-1, keepdim=True).to(actions.dtype)
        else:
            discrete = torch.ones((*actions.shape[:-1], 1), device=actions.device, dtype=actions.dtype)
        actions = torch.cat([actions, discrete], dim=-1)
    return actions


def run_episodes(env, env_processor, action_processor, policy, preprocessor, postprocessor, cfg, eval_cfg):
    results = []
    transition = reset_and_build_transition(env, env_processor, action_processor)
    robot = getattr(env, "robot", None)
    while len(results) < eval_cfg.n_episodes:
        episode_reward, steps, intervention_steps = 0.0, 0, 0
        max_potential, max_object_height = 0.0, None
        start_episode = time.perf_counter()
        while True:
            start_step = time.perf_counter()
            observation = {
                k: v
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }
            action = select_eval_action(policy, preprocessor.process_observation(observation), eval_cfg.deterministic)
            if cfg.policy.num_discrete_actions is not None:
                continuous = postprocessor.process_action(action[..., :-1])
                action = torch.cat([continuous, action[..., -1:].to(continuous)], dim=-1)
            else:
                action = postprocessor.process_action(action)

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            episode_reward += float(transition[TransitionKey.REWARD])
            steps += 1
            info = transition[TransitionKey.INFO]
            if info.get(TeleopEvents.IS_INTERVENTION, False):
                intervention_steps += 1
            max_potential = max(max_potential, float(info.get("staged_reward_potential", 0.0)))
            if hasattr(robot, "object_height"):
                height = robot.object_height()
                max_object_height = height if max_object_height is None else max(max_object_height, height)

            terminated = transition.get(TransitionKey.DONE, False)
            if terminated or transition.get(TransitionKey.TRUNCATED, False):
                break
            if not eval_cfg.headless:
                precise_sleep(max(1 / cfg.env.fps - (time.perf_counter() - start_step), 0.0))

        # Success = the episode ended on the task reward (shaping alone never terminates an episode)
        success = bool(terminated) and float(transition[TransitionKey.REWARD]) > 0.5
        staged = cfg.env.processor.staged_reward
        if success:
            stage = "SUCCESS"
        elif staged is not None and max_potential >= staged.lift_reward:
            really_lifted = max_object_height is None or max_object_height > eval_cfg.lift_check_height
            stage = "lifted" if really_lifted else "fake lift"
        elif staged is not None and max_potential >= staged.grasp_reward:
            stage = "grasped"
        else:
            stage = "not grasped"

        result = {
            "success": success,
            "stage": stage,
            "steps": steps,
            "seconds": time.perf_counter() - start_episode,
            "intervened": intervention_steps > 0,
        }
        results.append(result)
        height_note = f", max cube height {max_object_height:.3f} m" if max_object_height is not None else ""
        logging.info(
            f"Episode {len(results)}/{eval_cfg.n_episodes}: {stage} in {steps} steps ({result['seconds']:.1f}s)"
            + height_note
            + (f", intervened for {intervention_steps} steps" if result["intervened"] else "")
        )
        transition = reset_and_build_transition(env, env_processor, action_processor)
    return results


def report(results: list[dict]) -> None:
    autonomous = [r for r in results if not r["intervened"]]
    successes = [r for r in autonomous if r["success"]]
    logging.info("=" * 50)
    logging.info(f"Episodes: {len(results)} | with intervention (excluded): {len(results) - len(autonomous)}")
    if autonomous:
        logging.info(
            f"Autonomous success rate: {len(successes)}/{len(autonomous)} "
            f"= {100 * len(successes) / len(autonomous):.0f}%"
        )
    if successes:
        mean_steps = sum(r["steps"] for r in successes) / len(successes)
        logging.info(f"Mean steps of successful episodes: {mean_steps:.1f}")
    if autonomous:
        counts = Counter(r["stage"] for r in autonomous)
        logging.info("Furthest stage reached (autonomous episodes):")
        for stage in STAGES:
            if counts.get(stage):
                logging.info(f"  {stage:12s} {counts[stage]:3d}  ({100 * counts[stage] / len(autonomous):.0f}%)")
        if counts.get("fake lift"):
            logging.info("  'fake lift': the staged reward saw a lift but the cube stayed on the table")
    logging.info("=" * 50)


@draccus.wrap()
def main(eval_cfg: EvalRLConfig):
    init_logging()
    policy_dir = Path(eval_cfg.policy_path)
    cfg = TrainRLServerPipelineConfig.from_pretrained(str(policy_dir / "train_config.json"))
    cfg.policy.pretrained_path = policy_dir
    if hasattr(cfg.env.robot, "seed"):
        cfg.env.robot.seed = eval_cfg.seed

    if eval_cfg.headless:
        if not hasattr(cfg.env.robot, "show_viewer"):
            raise ValueError("headless evaluation is only available for simulated robots")
        cfg.env.robot.show_viewer = False
        robot = make_robot_from_config(cfg.env.robot)
        reset = cfg.env.processor.reset
        env = RobotEnv(
            robot=robot,
            use_gripper=cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper else True,
            reset_pose=reset.fixed_reset_joint_positions if reset else None,
            reset_time_s=0.0,
            terminate_on_success=reset.terminate_on_success if reset else True,
        )
        teleop_device = _NoTeleop(list(robot.bus.motors))
    else:
        env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.policy.device)
    try:
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy, dataset_stats=cfg.policy.dataset_stats
        )
        logging.info(
            f"Evaluating {policy_dir} for {eval_cfg.n_episodes} episodes "
            f"({'deterministic' if eval_cfg.deterministic else 'stochastic'} actions). Space = take over."
        )
        results = run_episodes(
            env, env_processor, action_processor, policy, preprocessor, postprocessor, cfg, eval_cfg
        )
        report(results)
    finally:
        env.close()
        if teleop_device is not None and teleop_device.is_connected:
            teleop_device.disconnect()


if __name__ == "__main__":
    main()
