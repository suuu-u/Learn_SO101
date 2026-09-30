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

"""Run a distilled vision student through the `gym_manipulator` pipeline (simulated or real SO101).

The environment and processors are the ones HIL-SERL uses (`make_robot_env` / `make_processors`):
camera crop / resize, joint velocity and current features, EE bounds, discrete gripper and IK. So
the only difference between the `so101_sim_follower` and the real arm is the env config file. The
student's training settings (fps, crops, step sizes, bounds, gripper range) are checked against the
env config before anything moves.

With a teleoperator configured (not `--headless`), taking over (e.g. the leader arm / Space)
overrides the policy like during HIL-SERL; such episodes are reported separately.

Examples:
```shell
# Simulation, headless statistics
python -m lerobot.rl.so101_mujoco.deploy --headless true --n_episodes 50
# Simulation with the viewer
python -m lerobot.rl.so101_mujoco.deploy --n_episodes 5
# Real robot, slowed down for the first runs
python -m lerobot.rl.so101_mujoco.deploy --env_config src/lerobot/configs/env_config_so101.json \
    --action_scale 0.5 --n_episodes 3
```
"""

import logging
import time
from dataclasses import dataclass

import draccus
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.cameras import opencv  # noqa: F401
from lerobot.processor import TransitionKey
from lerobot.robots import make_robot_from_config, so101_sim, so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

from ..eval_policy import _NoTeleop
from ..gym_manipulator import (
    GymManipulatorConfig,
    RobotEnv,
    log_transition_visualization,
    make_processors,
    make_robot_env,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from .configs import SO101PickPlaceEnvConfig, resolve_path
from .utils import STUDENT, load_checkpoint, resolve_device


@dataclass
class DeployConfig:
    # Distilled student checkpoint (from train_distill)
    checkpoint: str = "outputs/so101_mujoco/student_realcam.pt"
    # `gym_manipulator` env config: the simulated or the real SO101
    env_config: str = "src/lerobot/configs/env_config_so101_sim.json"
    n_episodes: int = 10
    # Simulation only: no viewer, no teleoperator and no real-time pacing (fast statistics)
    headless: bool = False
    # Connect the config's teleoperator to take over (HIL-SERL style). Turn off to watch the simulation
    # without the leader arm plugged in; on the real robot keep it on as the safety takeover.
    teleop: bool = True
    # Seed of the simulated scene
    seed: int | None = 0
    device: str = "cuda"
    # Multiplies the EE delta of the policy (< 1 moves slower, e.g. for the first real-robot runs)
    action_scale: float = 1.0


def check_compatible(train_env: SO101PickPlaceEnvConfig, env_cfg) -> None:
    """Refuse to run a student whose training settings differ from the deployment pipeline."""
    proc = env_cfg.processor
    crops = proc.image_preprocessing.crop_params_dict or {}
    ik = proc.inverse_kinematics
    # A student with per-camera image sizes needs the pipeline to crop only (`resize_size` null): the
    # images are then resized here, per camera, like in training
    pipeline_size = proc.image_preprocessing.resize_size
    pipeline_size = None if pipeline_size is None else list(pipeline_size)
    expected = {
        f"{camera} image size": (train_env.image_size(camera), pipeline_size) for camera in ("front", "wrist")
    }
    expected |= {
        "fps": (train_env.fps, env_cfg.fps),
        "front crop": (train_env.front_crop, crops.get("observation.images.front")),
        "max gripper position": (train_env.max_gripper_pos, proc.max_gripper_pos),
        "EE step sizes": (list(train_env.ee_step_sizes), [ik.end_effector_step_sizes[a] for a in "xyz"]),
        "EE bounds min": (list(train_env.ee_bounds_min), list(ik.end_effector_bounds["min"])),
        "EE bounds max": (list(train_env.ee_bounds_max), list(ik.end_effector_bounds["max"])),
    }
    # The wrist image is not cropped in training; a full-frame crop in the config is equivalent
    wrist_crop = crops.get("observation.images.wrist")
    full_frame = [0, 0, train_env.image_height, train_env.image_width]
    if train_env.wrist_crop is None and wrist_crop is not None and list(wrist_crop) != full_frame:
        expected["wrist crop"] = (full_frame, wrist_crop)
    mismatches = [
        f"{name}: trained with {a}, env config has {b}"
        for name, (a, b) in expected.items()
        if a is not None and b is not None and not np.allclose(np.asarray(a, float), np.asarray(b, float))
    ]
    if mismatches:
        raise ValueError("Student and env config disagree:\n  " + "\n  ".join(mismatches))
    if train_env.use_wrist_roll:
        raise ValueError("The student uses the wrist-roll action, which the HIL-SERL pipeline does not have")


def make_env(cfg: DeployConfig, env_cfg, train_env: SO101PickPlaceEnvConfig):
    """`make_robot_env`, or the same env with a never-intervening teleop stand-in."""
    simulated = hasattr(env_cfg.robot, "show_viewer")
    if simulated:
        env_cfg.robot.seed = cfg.seed
        # Simulate the task the student was trained on (object, success when back home)
        env_cfg.robot.object_size = None if train_env.object_size is None else list(train_env.object_size)
        env_cfg.robot.return_home = train_env.return_home
        env_cfg.robot.robot_base_z = train_env.robot_base_z
        for name in ("object_spawn_min", "object_spawn_max"):
            value = getattr(train_env, name)
            if value is not None:
                setattr(env_cfg.robot, name.replace("object", "cube"), list(value))
        if env_cfg.processor.reset is not None:
            env_cfg.processor.reset.terminate_on_success = train_env.terminate_on_success
    if cfg.teleop and not cfg.headless:
        return make_robot_env(cfg=env_cfg)
    if cfg.headless:
        if not simulated:
            raise ValueError("--headless is only available for simulated robots")
        env_cfg.robot.show_viewer = False
    elif not simulated:
        logging.warning("No teleoperator: the policy cannot be taken over on the real robot")
    robot = make_robot_from_config(env_cfg.robot)
    reset = env_cfg.processor.reset
    env = RobotEnv(
        robot=robot,
        use_gripper=env_cfg.processor.gripper.use_gripper if env_cfg.processor.gripper else True,
        reset_pose=reset.fixed_reset_joint_positions if reset else None,
        reset_time_s=0.0 if simulated else (reset.reset_time_s if reset else 5.0),
        terminate_on_success=reset.terminate_on_success if reset else True,
    )
    return env, _NoTeleop(list(robot.bus.motors))


@torch.no_grad()
def student_action(
    student, observation: dict, device: torch.device, action_scale: float, image_sizes: dict[str, list[int]]
) -> torch.Tensor:
    """Policy action in the pipeline's format: [dx, dy, dz, gripper command] (batched)."""
    images = {}
    for camera, size in image_sizes.items():
        image = observation[f"observation.images.{camera}"]
        if list(image.shape[-2:]) != list(size):
            image = F.resize(image, list(size), antialias=True).clamp(0.0, 1.0)
        # The pipeline gives float images in [0, 1]; the student takes the 0-255 range it was trained on
        images[camera] = image.to(device) * 255.0
    front, wrist = images["front"], images["wrist"]
    state = observation["observation.state"].to(device).float()
    action = student.act_inference(front, wrist, state).cpu()
    action[..., :-1] *= action_scale
    return action


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    init_logging()
    device = resolve_device(cfg.device)
    kind, student, train_env = load_checkpoint(cfg.checkpoint, device)
    if kind != STUDENT:
        raise ValueError(f"{cfg.checkpoint} is a {kind} checkpoint; only vision students can be deployed")

    manipulator_cfg = draccus.parse(GymManipulatorConfig, config_path=resolve_path(cfg.env_config), args=[])
    env_cfg = manipulator_cfg.env
    check_compatible(train_env, env_cfg)
    image_sizes = {camera: train_env.image_size(camera) for camera in ("front", "wrist")}

    # The config's `display_mode` (e.g. "rerun") streams the full-resolution camera frames and the
    # executed action, like `gym_manipulator`. Started before the cameras open: a spawned viewer would
    # inherit their file handles and keep them busy after this script ends
    display = manipulator_cfg.display_mode is not None and not cfg.headless
    if display:
        init_visualization(manipulator_cfg.display_mode, session_name="so101_mujoco_deploy")
    env, teleop_device = make_env(cfg, env_cfg, train_env)
    env_processor, action_processor = make_processors(env, teleop_device, env_cfg, "cpu")

    def show(transition) -> None:
        if display:
            raw_images = env.get_raw_images() if hasattr(env, "get_raw_images") else None
            log_transition_visualization(transition, manipulator_cfg, raw_images)
    results = []
    try:
        transition = reset_and_build_transition(env, env_processor, action_processor)
        show(transition)
        while len(results) < cfg.n_episodes:
            steps, intervention_steps, start_episode = 0, 0, time.perf_counter()
            while True:
                start_step = time.perf_counter()
                action = student_action(
                    student, transition[TransitionKey.OBSERVATION], device, cfg.action_scale, image_sizes
                )
                transition = step_env_and_process_transition(
                    env=env,
                    transition=transition,
                    action=action,
                    env_processor=env_processor,
                    action_processor=action_processor,
                )
                steps += 1
                show(transition)
                if transition[TransitionKey.INFO].get(TeleopEvents.IS_INTERVENTION, False):
                    intervention_steps += 1
                terminated = bool(transition.get(TransitionKey.DONE, False))
                if terminated or transition.get(TransitionKey.TRUNCATED, False):
                    break
                if not cfg.headless:
                    precise_sleep(max(1 / env_cfg.fps - (time.perf_counter() - start_step), 0.0))

            # Success = the task reward on the last step (the episode may run on after reaching it)
            success = float(transition[TransitionKey.REWARD]) > 0.5
            results.append({"success": success, "steps": steps, "intervened": intervention_steps > 0})
            logging.info(
                f"Episode {len(results)}/{cfg.n_episodes}: {'SUCCESS' if success else 'failure'} in {steps} "
                f"steps ({time.perf_counter() - start_episode:.1f}s)"
                + (f", intervened for {intervention_steps} steps" if intervention_steps else "")
            )
            transition = reset_and_build_transition(env, env_processor, action_processor)
            show(transition)
    finally:
        env.close()
        if display:
            shutdown_visualization(manipulator_cfg.display_mode)
        if teleop_device is not None and teleop_device.is_connected:
            teleop_device.disconnect()

    autonomous = [r for r in results if not r["intervened"]]
    successes = [r for r in autonomous if r["success"]]
    logging.info(f"Episodes: {len(results)} | with intervention (excluded): {len(results) - len(autonomous)}")
    if autonomous:
        rate = len(successes) / len(autonomous)
        mean_steps = np.mean([r["steps"] for r in successes]) if successes else float("nan")
        logging.info(
            f"Autonomous success rate: {len(successes)}/{len(autonomous)} = {rate:.0%} "
            f"(mean steps of successes {mean_steps:.1f})"
        )


def main() -> None:
    deploy()


if __name__ == "__main__":
    main()
