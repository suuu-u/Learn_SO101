#!/usr/bin/env python

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

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.utils import TeleopEvents

if TYPE_CHECKING:
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.teleoperators.teleoperator import Teleoperator

from lerobot.lerobot_types import EnvTransition, PolicyAction, TransitionKey
from lerobot.utils.constants import OBS_IMAGE

from .converters import batch_to_transition, transition_to_batch
from .pipeline import (
    ComplementaryDataProcessorStep,
    DataProcessorPipeline,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"
LEADER_JOINT_ACTION_KEY = "leader_joint_action"


@runtime_checkable
class HasTeleopEvents(Protocol):
    """
    Minimal protocol for objects that provide teleoperation events.

    This protocol defines the `get_teleop_events()` method, allowing processor
    steps to interact with teleoperators that support event-based controls
    (like episode termination or success flagging) without needing to know the
    teleoperator's specific class.
    """

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the teleoperator.

        Returns:
            A dictionary containing control events such as:
            - `is_intervention`: bool - Whether the human is currently intervening.
            - `terminate_episode`: bool - Whether to terminate the current episode.
            - `success`: bool - Whether the episode was successful.
            - `rerecord_episode`: bool - Whether to rerecord the episode.
        """
        ...


# Type variable constrained to Teleoperator subclasses that also implement events
TeleopWithEvents = TypeVar("TeleopWithEvents", bound="Teleoperator")


def _check_teleop_with_events(teleop: "Teleoperator") -> None:
    """
    Runtime check that a teleoperator implements the `HasTeleopEvents` protocol.

    Args:
        teleop: The teleoperator instance to check.

    Raises:
        TypeError: If the teleoperator does not have a `get_teleop_events` method.
    """
    if not isinstance(teleop, HasTeleopEvents):
        raise TypeError(
            f"Teleoperator {type(teleop).__name__} must implement get_teleop_events() method. "
            f"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop"
        )


@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryDataStep(ComplementaryDataProcessorStep):
    """
    Adds the raw action from a teleoperator to the transition's complementary data.

    This is useful for human-in-the-loop scenarios where the human's input needs to
    be available to downstream processors, for example, to override a policy's action
    during an intervention.

    Attributes:
        teleop_device: The teleoperator instance to get the action from.
    """

    teleop_device: "Teleoperator"

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Retrieves the teleoperator's action and adds it to the complementary data.

        Args:
            complementary_data: The incoming complementary data dictionary.

        Returns:
            A new dictionary with the teleoperator action added under the
            `teleop_action` key.
        """
        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = self.teleop_device.get_action()
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_teleop_action_as_info")
@dataclass
class AddTeleopEventsAsInfoStep(InfoProcessorStep):
    """
    Adds teleoperator control events (e.g., terminate, success) to the transition's info.

    This step extracts control events from teleoperators that support event-based
    interaction, making these signals available to other parts of the system.

    Attributes:
        teleop_device: An instance of a teleoperator that implements the
                       `HasTeleopEvents` protocol.
        latch_success: If True, a success event stays set for the rest of the episode (until
                       `reset()`), so every frame after the success key press is labeled as a
                       success. Useful for collecting reward classifier data when episodes do not
                       terminate on success.
    """

    teleop_device: TeleopWithEvents
    latch_success: bool = False

    def __post_init__(self):
        """Validates that the provided teleoperator supports events after initialization."""
        _check_teleop_with_events(self.teleop_device)
        self._success_latched = False

    def info(self, info: dict) -> dict:
        """
        Retrieves teleoperator events and updates the info dictionary.

        Args:
            info: The incoming info dictionary.

        Returns:
            A new dictionary including the teleoperator events.
        """
        new_info = dict(info)

        teleop_events = dict(self.teleop_device.get_teleop_events())
        if self.latch_success:
            self._success_latched = self._success_latched or bool(teleop_events.get(TeleopEvents.SUCCESS))
            teleop_events[TeleopEvents.SUCCESS] = self._success_latched
        new_info.update(teleop_events)
        return new_info

    def reset(self) -> None:
        self._success_latched = False

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessorStep(ObservationProcessorStep):
    """
    Crops and/or resizes image observations.

    This step iterates through all image keys in an observation dictionary and applies
    the specified transformations. It handles device placement, moving tensors to the
    CPU if necessary for operations not supported on certain accelerators like MPS.

    Attributes:
        crop_params_dict: A dictionary mapping image keys to cropping parameters
                          (top, left, height, width).
        resize_size: A tuple (height, width) to resize all images to.
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:
        """
        Applies cropping and resizing to all images in the observation dictionary.

        Args:
            observation: The observation dictionary, potentially containing image tensors.

        Returns:
            A new observation dictionary with transformed images.
        """
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)

        # Process all image keys in the observation
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            # NOTE (maractingi): No mps kernel for crop and resize, so we need to move to cpu
            if device.type == "mps":
                image = image.cpu()
            # Crop if crop params are provided for this key
            if self.crop_params_dict is not None and key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            if self.resize_size is not None:
                image = F.resize(image, self.resize_size)
                image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary with the crop parameters and resize dimensions.
        """
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the image feature shapes in the policy features dictionary if resizing is applied.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary with new image shapes.
        """
        if self.resize_size is None:
            return features
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessorStep(TruncatedProcessorStep):
    """
    Tracks episode steps and enforces a time limit by truncating the episode.

    Attributes:
        max_episode_steps: The maximum number of steps allowed per episode.
        current_step: The current step count for the active episode.
    """

    max_episode_steps: int
    current_step: int = 0

    def truncated(self, truncated: bool) -> bool:
        """
        Increments the step counter and sets the truncated flag if the time limit is reached.

        Args:
            truncated: The incoming truncated flag.

        Returns:
            True if the episode step limit is reached, otherwise the incoming value.
        """
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        # TODO (steven): missing an else truncated = False?
        return truncated

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the `max_episode_steps`.
        """
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:
        """Resets the step counter, typically called at the start of a new episode."""
        self.current_step = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("gym_hil_adapter_processor")
class GymHILAdapterProcessorStep(ProcessorStep):
    """
    Adapts the output of the `gym-hil` environment to the format expected by `lerobot` processors.

    This step normalizes the `transition` object by:
    1. Copying `teleop_action` from `info` to `complementary_data`.
    2. Copying `is_intervention` from `info` (using the string key) to `info` (using the enum key).
    3. Copying `discrete_penalty` from `info` to `complementary_data`.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        if TELEOP_ACTION_KEY in info:
            complementary_data[TELEOP_ACTION_KEY] = info[TELEOP_ACTION_KEY]

        if DISCRETE_PENALTY_KEY in info:
            complementary_data[DISCRETE_PENALTY_KEY] = info[DISCRETE_PENALTY_KEY]

        if "is_intervention" in info:
            info[TeleopEvents.IS_INTERVENTION] = info["is_intervention"]

        transition[TransitionKey.INFO] = info
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessorStep(ProcessorStep):
    """
    Applies a small per-transition cost on the discrete gripper action.

    Fires only when the commanded action would actually transition the gripper
    from one extreme to the other (close-while-open or open-while-closed).
    This discourages gripper oscillation while leaving "stay" and saturating-further
    commands unpenalized.

    Attributes:
        penalty: The negative reward value to apply.
        max_gripper_pos: The maximum position value for the gripper, used for normalization.
        open_threshold: Normalized state below which the gripper is considered "open".
        closed_threshold: Normalized state above which the gripper is considered "closed".
    """

    penalty: float = -0.02
    max_gripper_pos: float = 30.0
    open_threshold: float = 0.1
    closed_threshold: float = 0.9

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Calculates the gripper penalty and adds it to the complementary data.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition with the penalty added to complementary data.
        """
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        raw_joint_positions = complementary_data.get("raw_joint_positions")
        if raw_joint_positions is None:
            return new_transition

        current_gripper_pos = raw_joint_positions.get(f"{GRIPPER_KEY}.pos", None)
        if current_gripper_pos is None:
            return new_transition

        # During reset, the transition may not carry any action yet.
        if action is None:
            return new_transition

        # Gripper action is expected as the last action dimension.
        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos

        # Normalize gripper state and action
        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos

        # Calculate penalty boolean as in original
        #   - currently open  AND target is closed  -> close transition
        #   - currently closed AND target is open   -> open transition
        is_open = gripper_state_normalized < self.open_threshold
        is_closed = gripper_state_normalized > self.closed_threshold
        cmd_close = gripper_action_normalized > self.closed_threshold
        cmd_open = gripper_action_normalized < self.open_threshold
        gripper_penalty_bool = (is_open and cmd_close) or (is_closed and cmd_open)

        gripper_penalty = self.penalty * int(gripper_penalty_bool)

        # Update complementary data with penalty info
        new_complementary_data = dict(complementary_data)
        new_complementary_data[DISCRETE_PENALTY_KEY] = gripper_penalty
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the penalty value, max gripper position,
            and the open/closed thresholds.
        """
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
            "open_threshold": self.open_threshold,
            "closed_threshold": self.closed_threshold,
        }

    def reset(self) -> None:
        """Resets the processor's internal state."""
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
class LeaderArmTeleopStep(ProcessorStep):
    """
    Lets a leader arm (e.g. SO101 leader) drive HIL-SERL interventions.

    The follower is driven with the leader's joint positions (see `LeaderJointPassthroughStep`), the
    way `lerobot-teleoperate` does. What gets recorded is the same `[delta_x, delta_y, delta_z,
    gripper]` action the SAC policy outputs: the leader's end-effector displacement since the last
    command, computed with forward kinematics, plus a discrete gripper command.

    While not intervening, the leader's torque is enabled and it tracks the follower, so taking
    over starts from the follower's current pose. When intervention starts, torque is released
    so the human can move the leader freely.

    With `clutch=True` the leader is instead used like a 3D mouse and never tracks the follower
    (its torque stays off). Taking over anchors the leader's current end-effector position to the
    follower's; from then on the follower target is its anchor plus the leader's displacement since
    the anchor, reached through the same rate-limited EE deltas the policy uses. There is no jump
    at takeover and the recorded action is the motion actually executed. The gripper keeps its
    state at takeover (holding keeps squeezing) until the leader gripper crosses the threshold.

    Must be placed after `AddTeleopActionAsComplimentaryDataStep` and `AddTeleopEventsAsInfoStep`
    and before `InterventionActionProcessorStep`. The transition observation must hold the
    follower's raw joint positions (`"<motor>.pos"`).

    Attributes:
        teleop_device: The leader arm teleoperator.
        kinematics: Kinematics solver of the follower robot.
        motor_names: Ordered motor names shared by leader and follower.
        end_effector_step_sizes: Max EE displacement per step for x, y, z (meters); a delta of
            1.0 corresponds to one step size.
        max_gripper_pos: Gripper position the follower closes to; half of it is the threshold
            used to decide the discrete gripper command.
        use_gripper: Whether to append a discrete gripper command (0 opens, 1 stays, 2 closes).
        clutch: Relative ("clutch") control instead of copying the leader's absolute pose.
    """

    teleop_device: "Teleoperator"
    kinematics: "RobotKinematics"
    motor_names: list[str]
    end_effector_step_sizes: dict[str, float]
    max_gripper_pos: float = 100.0
    use_gripper: bool = True
    clutch: bool = False

    def __post_init__(self):
        self._step_sizes = np.array([self.end_effector_step_sizes[axis] for axis in ("x", "y", "z")])
        self._leader_torque_enabled: bool | None = None
        self._was_intervening = False
        self._leader_anchor: np.ndarray | None = None
        self._follower_anchor: np.ndarray | None = None
        self._leader_gripper_closed: bool | None = None
        self._gripper_command: float | None = None

    def _set_leader_tracking(self, tracking: bool, follower_joints: dict[str, float]) -> None:
        if tracking != self._leader_torque_enabled:
            if tracking:
                self.teleop_device.enable_torque()
            else:
                self.teleop_device.disable_torque()
            self._leader_torque_enabled = tracking
        if tracking:
            self.teleop_device.send_feedback(follower_joints)

    def _gripper_action(self, leader_pos: float, follower_pos: float) -> float:
        # GripperVelocityToJoint (discrete) moves the gripper to a limit:
        # 0 drives the position up to max_gripper_pos, 2 drives it down to 0, 1 holds.
        threshold = self.max_gripper_pos / 2
        if leader_pos > threshold and follower_pos <= threshold:
            return 0.0
        if leader_pos <= threshold and follower_pos > threshold:
            return 2.0
        return 1.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        leader_action = complementary_data.get(TELEOP_ACTION_KEY)
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        if not isinstance(leader_action, dict) or not observation:
            return transition

        follower_joints = {f"{m}.pos": float(observation[f"{m}.pos"]) for m in self.motor_names}
        is_intervention = transition.get(TransitionKey.INFO, {}).get(TeleopEvents.IS_INTERVENTION, False)
        # In clutch mode the leader never tracks the follower: its torque stays off
        self._set_leader_tracking(not is_intervention and not self.clutch, follower_joints)

        # The delta is measured from the last joint command (the previous IK solution, or the previous
        # leader pose while intervening), the same reference EEReferenceAndDelta adds deltas to.
        # Measuring it from the lagging follower instead would integrate the tracking error.
        reference_q = complementary_data.get("IK_solution")
        if reference_q is None:
            reference_q = [follower_joints[f"{m}.pos"] for m in self.motor_names]
        leader_q = np.array([float(leader_action[f"{m}.pos"]) for m in self.motor_names])
        leader_ee = self.kinematics.forward_kinematics(leader_q)[:3, 3]
        reference_ee = self.kinematics.forward_kinematics(np.asarray(reference_q, dtype=float))[:3, 3]
        leader_gripper = float(leader_action[f"{GRIPPER_KEY}.pos"])
        follower_gripper = follower_joints[f"{GRIPPER_KEY}.pos"]

        if self.clutch:
            delta, gripper = self._clutch_action(
                is_intervention, leader_ee, reference_ee, leader_gripper, follower_gripper
            )
        else:
            delta = np.clip((leader_ee - reference_ee) / self._step_sizes, -1.0, 1.0)
            gripper = self._gripper_action(leader_gripper, follower_gripper)

        ee_action = {"delta_x": float(delta[0]), "delta_y": float(delta[1]), "delta_z": float(delta[2])}
        if self.use_gripper:
            ee_action[GRIPPER_KEY] = gripper

        new_transition = transition.copy()
        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = ee_action
        new_complementary_data[LEADER_JOINT_ACTION_KEY] = dict(leader_action)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return new_transition

    def _clutch_action(
        self,
        is_intervention: bool,
        leader_ee: np.ndarray,
        reference_ee: np.ndarray,
        leader_gripper: float,
        follower_gripper: float,
    ) -> tuple[np.ndarray, float]:
        threshold = self.max_gripper_pos / 2
        leader_closed = leader_gripper <= threshold
        if is_intervention and not self._was_intervening:
            # Engage the clutch: from here on only the leader's motion counts, not its pose
            self._leader_anchor = leader_ee
            self._follower_anchor = reference_ee
            self._leader_gripper_closed = leader_closed
            self._gripper_command = None
        self._was_intervening = is_intervention
        if not is_intervention:
            return np.zeros(3), 1.0

        target = self._follower_anchor + (leader_ee - self._leader_anchor)
        delta = np.clip((target - reference_ee) / self._step_sizes, -1.0, 1.0)

        # The gripper follows the leader only once the leader gripper crosses the threshold
        if leader_closed != self._leader_gripper_closed:
            self._gripper_command = 2.0 if leader_closed else 0.0
            self._leader_gripper_closed = leader_closed
        if self._gripper_command is not None:
            gripper = self._gripper_command
        else:
            # Keep the state at takeover; a closed (e.g. holding) gripper keeps squeezing
            gripper = 2.0 if follower_gripper <= threshold else 1.0
        return delta, gripper

    def reset(self) -> None:
        # Re-sync the torque state on the next step (the env reset may have moved the follower).
        self._leader_torque_enabled = None
        # The follower was reset: re-anchor the clutch on the next intervening step
        self._was_intervening = False

    def get_config(self) -> dict[str, Any]:
        return {
            "motor_names": self.motor_names,
            "end_effector_step_sizes": self.end_effector_step_sizes,
            "max_gripper_pos": self.max_gripper_pos,
            "use_gripper": self.use_gripper,
            "clutch": self.clutch,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
class LeaderJointPassthroughStep(ProcessorStep):
    """
    During a leader-arm intervention, sends the leader's joint positions straight to the follower,
    the same way `lerobot-teleoperate` does (`robot.send_action(teleop.get_action())`).

    The follower copies the leader exactly (wrist orientation included) instead of following the IK
    solution of the recorded EE delta. The gripper is clipped to `[0, max_gripper_pos]`, the range
    the policy's discrete gripper command can reach. When the leader leaves the end-effector
    bounds, the follower falls back to the clipped IK command and waits at the edge of the
    workspace, so it is always inside the bounds when control is handed back.

    While the leader drives the follower, the kinematic steps are reset every step so they restart
    from the leader's pose (published as `IK_solution`) instead of their stale pre-intervention
    command.

    Must be placed after `InverseKinematicsRLStep` and before
    `RobotActionToPolicyActionProcessorStep`.

    Attributes:
        kinematics: Kinematics solver of the follower robot.
        motor_names: Ordered motor names shared by leader and follower.
        end_effector_bounds: The `min`/`max` EE position bounds of the workspace.
        max_gripper_pos: Upper gripper position limit.
        kinematic_steps: The pipeline's EEReferenceAndDelta, EEBoundsAndSafety and
            InverseKinematicsRLStep steps.
    """

    kinematics: "RobotKinematics"
    motor_names: list[str]
    end_effector_bounds: dict[str, list[float]]
    max_gripper_pos: float
    kinematic_steps: list[ProcessorStep]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        is_intervention = transition.get(TransitionKey.INFO, {}).get(TeleopEvents.IS_INTERVENTION, False)
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        leader_action = complementary_data.get(LEADER_JOINT_ACTION_KEY)
        if not is_intervention or not isinstance(leader_action, dict):
            return transition

        leader_q = np.array([float(leader_action[f"{m}.pos"]) for m in self.motor_names])
        leader_pos = self.kinematics.forward_kinematics(leader_q)[:3, 3]
        if np.any(leader_pos < self.end_effector_bounds["min"]) or np.any(
            leader_pos > self.end_effector_bounds["max"]
        ):
            # Leader is outside the workspace: keep the IK command, which the bounds step clipped
            return transition

        if GRIPPER_KEY in self.motor_names:
            gripper_idx = self.motor_names.index(GRIPPER_KEY)
            leader_q[gripper_idx] = np.clip(leader_q[gripper_idx], 0.0, self.max_gripper_pos)

        new_transition = transition.copy()
        action = dict(transition.get(TransitionKey.ACTION) or {})
        for name, q in zip(self.motor_names, leader_q, strict=True):
            action[f"{name}.pos"] = float(q)
        new_transition[TransitionKey.ACTION] = action

        new_complementary_data = dict(complementary_data)
        new_complementary_data["IK_solution"] = leader_q
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data

        for step in self.kinematic_steps:
            step.reset()
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "motor_names": self.motor_names,
            "end_effector_bounds": self.end_effector_bounds,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
    """
    Handles human intervention, overriding policy actions and managing episode termination.

    When an intervention is detected (via teleoperator events in the `info` dict),
    this step replaces the policy's action with the human's teleoperated action.
    It also processes signals to terminate the episode or flag success.

    Attributes:
        use_gripper: Whether to include the gripper in the teleoperated action.
        terminate_on_success: If True, automatically sets the `done` flag when a
                              `success` event is received.
    """

    use_gripper: bool = False
    terminate_on_success: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes the transition to handle interventions.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition, potentially with an overridden action, updated
            reward, and termination status.
        """
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # Get intervention signals from complementary data
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # Override action if intervention is active
        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict):
                # Convert teleop_action dict to tensor format
                action_list = [
                    teleop_action.get("delta_x", 0.0),
                    teleop_action.get("delta_y", 0.0),
                    teleop_action.get("delta_z", 0.0),
                ]
                if self.use_gripper:
                    action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
            elif isinstance(teleop_action, np.ndarray):
                action_list = teleop_action.tolist()
            else:
                action_list = teleop_action

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # Handle episode termination
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info with intervention metadata
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data with teleop action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessorStep(ProcessorStep):
    """
    Applies a pretrained reward classifier to image observations to predict success.

    This step uses a model to determine if the current state is successful, updating
    the reward and potentially terminating the episode.

    Attributes:
        pretrained_path: Path to the pretrained reward classifier model.
        device: The device to run the classifier on.
        success_threshold: The probability threshold to consider a prediction as successful.
        success_reward: The reward value to assign on success.
        terminate_on_success: If True, terminates the episode upon successful classification.
        reward_classifier: The loaded classifier model instance.
    """

    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None
    classifier_preprocessor: Any = None

    def __post_init__(self):
        """Initializes the reward classifier model after the dataclass is created."""
        self._last_success = False
        if self.pretrained_path is not None:
            from lerobot.rewards.classifier.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

            # Apply the same normalization the classifier was trained with, and move images to its device
            self.classifier_preprocessor = DataProcessorPipeline.from_pretrained(
                self.pretrained_path,
                config_filename="classifier_preprocessor.json",
                overrides={"device_processor": {"device": self.device}},
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes a transition, applying the reward classifier to its image observations.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition with an updated reward and done flag based on the
            classifier's prediction.
        """
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return new_transition

        # Extract images from observation
        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return new_transition

        # Run reward classifier
        start_time = time.perf_counter()
        with torch.inference_mode():
            images = self.classifier_preprocessor(images)
            image_inputs = [
                images[key] for key in self.reward_classifier.config.input_features if key.startswith(OBS_IMAGE)
            ]
            probabilities = self.reward_classifier.predict(image_inputs).probabilities
            if self.reward_classifier.config.num_classes == 2:
                success_probability = float(probabilities.flatten()[0])
                success = float(success_probability > self.success_threshold)
            else:
                success_probability = float(probabilities.flatten()[-1])
                success = float(torch.argmax(probabilities, dim=-1).flatten()[0])

        classifier_frequency = 1 / (time.perf_counter() - start_time)

        is_success = math.isclose(success, 1, abs_tol=1e-2)
        if is_success != self._last_success:
            logging.info(
                f"Reward classifier: {'SUCCESS' if is_success else 'no success'} (p={success_probability:.2f})"
            )
            self._last_success = is_success

        # Calculate reward and termination
        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if is_success:
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True

        # Update transition
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        # Update info with classifier frequency
        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        info["reward_classifier_probability"] = success_probability
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


# Gripper motion (robot units per step) below which a closing gripper counts as stopped
GRASP_STALL_TOLERANCE = 1.0


def grasp_progress(
    joint_positions: np.ndarray,
    closing: bool,
    kinematics: Any,
    grasp_min_gripper_pos: float,
    lift_height: float,
    grasp_reward: float,
    lift_reward: float,
    previous_gripper_pos: float | None = None,
) -> float:
    """Task-progress potential Φ from proprioception only (joints in robot units, gripper last).

    Grasped = the last gripper command was "close" and the gripper stopped (moved less than
    `GRASP_STALL_TOLERANCE` since `previous_gripper_pos`) above `grasp_min_gripper_pos`: the jaws are
    blocked by an object. Lifted = grasped and the end-effector is higher than `lift_height`.
    """
    if not closing or joint_positions[-1] <= grasp_min_gripper_pos:
        return 0.0
    if previous_gripper_pos is not None and abs(joint_positions[-1] - previous_gripper_pos) > GRASP_STALL_TOLERANCE:
        # Still closing: not blocked by anything yet
        return 0.0
    ee_z = kinematics.forward_kinematics(np.asarray(joint_positions, dtype=float))[2, 3]
    return lift_reward if ee_z > lift_height else grasp_reward


def update_closing_intent(closing: bool, gripper_command: float) -> bool:
    """Latch the last non-"stay" discrete gripper command (on SO101: 2 closes, 0 opens, 1 stays)."""
    if gripper_command > 1.5:
        return True
    if gripper_command < 0.5:
        return False
    return closing


def staged_shaping_for_trajectories(
    joint_positions: np.ndarray,
    gripper_commands: np.ndarray,
    dones: np.ndarray,
    kinematics: Any,
    grasp_min_gripper_pos: float,
    lift_height: float,
    grasp_reward: float,
    lift_reward: float,
    discount: float,
) -> np.ndarray:
    """Shaping terms discount·Φ(s') − Φ(s) for consecutive transitions (e.g. an offline buffer).

    Row i holds the state s_i and the command applied from it; s' is row i + 1 unless the
    transition ends an episode (terminal: Φ(s') = 0). Matches `StagedRewardProcessorStep`.
    """
    shaping = np.zeros(len(joint_positions), dtype=np.float32)
    closing, phi = False, 0.0
    for i in range(len(joint_positions)):
        closing = update_closing_intent(closing, float(gripper_commands[i]))
        if dones[i] or i + 1 >= len(joint_positions):
            shaping[i] = -phi
            closing, phi = False, 0.0
            continue
        phi_next = grasp_progress(
            joint_positions[i + 1],
            closing,
            kinematics,
            grasp_min_gripper_pos,
            lift_height,
            grasp_reward,
            lift_reward,
            previous_gripper_pos=float(joint_positions[i][-1]),
        )
        shaping[i] = discount * phi_next - phi
        phi = phi_next
    return shaping


@ProcessorStepRegistry.register("staged_reward_processor")
@dataclass
class StagedRewardProcessorStep(ProcessorStep):
    """Adds potential-based grasp/lift shaping to the environment reward.

    Φ comes from `grasp_progress` (raw joint positions after the step and the latched gripper
    command of the executed action), and each step adds discount·Φ(s') − Φ(s), with Φ(s') = 0 on
    terminal steps. The shaping cannot be farmed (dropping gives it back) and leaves the optimal
    policy unchanged, while rewarding the grasp and the lift as soon as they happen.

    Attributes:
        kinematics: Robot kinematics used to get the end-effector height.
        motor_names: Joint order of the raw joint positions (gripper last).
        grasp_reward: Φ when grasped.
        lift_reward: Φ when grasped and lifted.
        grasp_min_gripper_pos: Gripper position above which a closed gripper holds something.
        lift_height: End-effector height above which a grasped object is lifted.
        discount: RL discount used in the shaping term.
    """

    kinematics: Any
    motor_names: list[str]
    grasp_reward: float = 0.1
    lift_reward: float = 0.3
    grasp_min_gripper_pos: float = 2.5
    lift_height: float = 0.05
    discount: float = 0.97

    def __post_init__(self):
        self._closing = False
        self._phi = 0.0
        self._previous_gripper_pos: float | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        raw_joints = complementary_data.get("raw_joint_positions")
        executed_action = complementary_data.get(TELEOP_ACTION_KEY)
        if raw_joints is None or not isinstance(executed_action, torch.Tensor):
            return transition

        self._closing = update_closing_intent(self._closing, float(executed_action.flatten()[-1]))
        joints = np.array([raw_joints[f"{m}.pos"] for m in self.motor_names], dtype=float)
        phi_next = grasp_progress(
            joints,
            self._closing,
            self.kinematics,
            self.grasp_min_gripper_pos,
            self.lift_height,
            self.grasp_reward,
            self.lift_reward,
            previous_gripper_pos=self._previous_gripper_pos,
        )
        self._previous_gripper_pos = float(joints[-1])
        terminal = bool(transition.get(TransitionKey.DONE, False))
        shaping = (0.0 if terminal else self.discount * phi_next) - self._phi
        self._phi = phi_next

        new_transition = transition.copy()
        new_transition[TransitionKey.REWARD] = float(transition.get(TransitionKey.REWARD) or 0.0) + shaping
        info = dict(new_transition.get(TransitionKey.INFO) or {})
        info["staged_reward_potential"] = phi_next
        new_transition[TransitionKey.INFO] = info
        return new_transition

    def reset(self) -> None:
        self._closing = False
        self._phi = 0.0
        self._previous_gripper_pos = None

    def get_config(self) -> dict[str, Any]:
        return {
            "motor_names": self.motor_names,
            "grasp_reward": self.grasp_reward,
            "lift_reward": self.lift_reward,
            "grasp_min_gripper_pos": self.grasp_min_gripper_pos,
            "lift_height": self.lift_height,
            "discount": self.discount,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
