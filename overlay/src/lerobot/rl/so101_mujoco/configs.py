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

"""Configs for the MuJoCo SO101 pick-and-place RL pipeline (PPO teacher -> vision student)."""

from dataclasses import dataclass, field
from pathlib import Path

# Repository root, so the default asset paths work from any working directory
REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_path(path: str) -> str:
    """Absolute path; relative paths are tried against the working directory, then the repo root."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return str(p.resolve())
    return str(REPO_ROOT / p)


@dataclass
class RewardConfig:
    """Staged dense reward, normalized to [-1, 0] per step (before penalties) so finishing early is
    always better.

    Stage value S: reach (<=1) + grasped (1) + lift (<=1) [+ transport_weight * transport], and S = 6
    once the object is placed (still, its whole outline inside the ring). With `return_home`, a placed
    object is worth 6 + home_weight * (1 - d / home_radius) for the gripper at distance d from its
    reset pose, and success needs it back home. The step reward is S / S_max - 1 (0 at success).
    """

    # Linear reach term 1 - |finger-pad midpoint - cube| / reach_radius, so the cube pulls from the start
    # pose (a tanh(10 d) term is flat beyond ~10 cm and gives potential shaping nothing to follow)
    reach_radius: float = 0.3
    # Cube lift above its resting height (m) that earns the full lift term
    lift_height: float = 0.04
    # Linear transport term 1 - d / transport_radius, weighted by `transport_weight` while grasped (0
    # turns it off: then only a placed object pays for carrying it to the ring), d being the distance
    # of the cube to the carry region: the column above the ring interior, above the carry height
    # outside the ring and at any height inside it. Outside the ring only a lifted cube can reduce d,
    # so dragging it against the ring wall pays less than carrying it over; inside, lowering is free.
    transport_radius: float = 0.4
    transport_weight: float = 0.0
    # Gap between the bottom of the carried object and the top of the ring wall (sets the carry height)
    carry_clearance: float = 0.01
    # The region's radius is `ring inner radius - object half diagonal - carry_margin`: released anywhere
    # inside it, the object lands in the ring whatever its yaw (its corners reach its half diagonal)
    carry_margin: float = 0.003
    # Per meter of distance between the IK reference and the measured EE beyond the tolerance. Without
    # it the policy parks the reference in an unreachable workspace corner, where the arm stops moving
    # and exploration noise has no effect (a safe way to hold the cube instead of carrying it). The
    # tolerance is above the ~1.6 cm tracking lag of full-speed moves, which must stay free: otherwise
    # every move is penalized and the policy hides in a workspace corner, where bounds clip the noise.
    reference_penalty: float = 1.0
    reference_tolerance: float = 0.02
    # Per step and axis on which the EE command is clipped by the x / y bounds or the top of the
    # workspace. Pushing into a bound corner freezes the arm and clips the exploration noise away, a
    # safe haven for holding the cube that the policy otherwise prefers to carrying it. The task never
    # needs those bounds (spawn area and ring are inside); the floor bound is exempt since grasping a
    # cube on the table presses down onto it.
    bound_penalty: float = 0.1
    # Growing penalty for keeping the EE close to the table: after `low_ee_grace_steps` consecutive
    # steps below `low_ee_height`, each further step adds `low_ee_penalty` (capped per step at
    # `low_ee_penalty_max`). Reaching / grasping near the table is short and free, dragging the cube
    # along the table to the ring before lifting it gets expensive. Not counted while the cube is
    # inside the ring (placing it there brings the EE down again).
    low_ee_height: float = 0.025
    low_ee_grace_steps: int = 5
    low_ee_penalty: float = 0.02
    low_ee_penalty_max: float = 0.5
    # With `return_home`: weight of the home term of a placed object. The term averages a linear decay
    # over `home_radius` (pulls the gripper back from anywhere) and 1 - tanh(d / home_precision), steep
    # within a few cm (pays for really reaching and staying at the reset pose instead of drifting
    # around it)
    home_weight: float = 4.0
    home_radius: float = 0.25
    home_precision: float = 0.03
    # Action smoothness: penalty on the squared change of the EE delta action between steps (0 to
    # 12 for actions in [-1, 1]); discourages jittery back-and-forth commands
    action_rate_penalty: float = 0.02
    # Penalty per gripper open/close command (the HIL-SERL `gripper_penalty`); discourages chattering
    gripper_penalty: float = 0.0
    # Potential-based shaping (Ng et al. 1999) instead of the absolute stage reward: the step reward is
    # gamma * S(s') - S(s) plus `success_bonus` on success, so holding still pays ~0 and only progress
    # pays (the delta-progress idea of OpenSO-101's PickPlace rewards, without changing the optimum).
    # On this task it learned nothing in 600k steps: the per-step signal (~1e-3) drowned in the value
    # error, so the stage reward stays the default.
    potential_shaping: bool = False
    # Should match the PPO discount
    potential_gamma: float = 0.99
    success_bonus: float = 10.0


@dataclass
class VisualRandomizationConfig:
    """Per-episode appearance randomization of the rendered camera images (student only).

    Only rendering changes (colors, lights, camera poses); the physics is untouched. Colors are
    sampled in HSV, `*_value` being the brightness range.
    """

    enabled: bool = False
    # Ring: random hue; with probability `ring_single_color_prob` both segment colors match (a plain
    # roll of tape), otherwise they alternate like the default red / yellow ring
    ring_saturation: list[float] = field(default_factory=lambda: [0.3, 1.0])
    ring_value: list[float] = field(default_factory=lambda: [0.3, 1.0])
    ring_single_color_prob: float = 0.5
    cube_saturation: list[float] = field(default_factory=lambda: [0.0, 1.0])
    cube_value: list[float] = field(default_factory=lambda: [0.03, 0.8])
    table_saturation: list[float] = field(default_factory=lambda: [0.0, 0.2])
    table_value: list[float] = field(default_factory=lambda: [0.5, 1.0])
    # Brightness factor of the white printed robot parts
    robot_brightness: list[float] = field(default_factory=lambda: [0.8, 1.05])
    # Factors on the headlight / directional light intensity, and std of the light direction tilt
    light_scale: list[float] = field(default_factory=lambda: [0.5, 1.5])
    light_direction_std: float = 0.2
    # Camera pose jitter: uniform position offset (m) per axis and rotation (deg) about a random axis
    front_camera_position: float = 0.01
    wrist_camera_position: float = 0.003
    camera_rotation_deg: float = 2.0
    # Stronger variations (off by default so older checkpoints evaluate as trained). Table: probability
    # of a procedural texture (checker / gradient / speckled) instead of a plain color
    table_texture_prob: float = 0.0
    # Up to this many visual-only distractor objects (no collisions) placed on the table per episode
    max_distractors: int = 0
    # Image corruptions of the rendered (cropped, resized) images: Gaussian blur, per-pixel noise (std in
    # 0-255 units) and JPEG compression (quality range)
    blur_prob: float = 0.0
    blur_sigma: list[float] = field(default_factory=lambda: [0.3, 1.2])
    noise_std: list[float] = field(default_factory=lambda: [0.0, 0.0])
    jpeg_prob: float = 0.0
    jpeg_quality: list[int] = field(default_factory=lambda: [30, 90])
    # Photometric changes of the rendered images, sampled per episode and camera (each camera has its
    # own auto exposure / white balance): gain on the pixel values, contrast and saturation factors
    # around the image mean / the pixel gray value, and gamma. [1, 1] leaves the image unchanged
    exposure: list[float] = field(default_factory=lambda: [1.0, 1.0])
    contrast: list[float] = field(default_factory=lambda: [1.0, 1.0])
    saturation: list[float] = field(default_factory=lambda: [1.0, 1.0])
    gamma: list[float] = field(default_factory=lambda: [1.0, 1.0])
    # Shadows and reflections (off by default). The overhead light always casts short shadows; with
    # probability `lamp_prob` a shadow-casting spot light (a desk lamp / window) shines from a random
    # side, `lamp_distance` m away from the workspace center and `lamp_elevation_deg` above the table,
    # with a diffuse intensity in `lamp_intensity`: long shadows of the arm and object
    lamp_prob: float = 0.0
    lamp_distance: list[float] = field(default_factory=lambda: [0.5, 1.2])
    lamp_elevation_deg: list[float] = field(default_factory=lambda: [25.0, 70.0])
    lamp_intensity: list[float] = field(default_factory=lambda: [0.3, 0.8])
    # Specular strength of all lights (highlights on the arm, ring and object; 0.3 by default)
    light_specular: list[float] = field(default_factory=lambda: [0.3, 0.3])
    # Glossy table: mirror reflectance, specular strength and shininess of its material
    table_reflectance: list[float] = field(default_factory=lambda: [0.0, 0.0])
    table_specular: list[float] = field(default_factory=lambda: [0.5, 0.5])
    table_shininess: list[float] = field(default_factory=lambda: [0.5, 0.5])

    @property
    def lighting_variety(self) -> bool:
        """Whether the scene needs the extra lamp and the glossy table material."""
        return (
            self.lamp_prob > 0
            or self.table_reflectance[1] > 0
            or list(self.table_specular) != [0.5, 0.5]
            or list(self.table_shininess) != [0.5, 0.5]
        )


@dataclass
class CameraTimingConfig:
    """When the student's camera frames are captured (student only; off by default).

    Latency: the frame observed after a control step was captured `latency` s before its end (USB
    webcams deliver frames ~50-150 ms late), with a per-episode base in `latency_s` plus a per-step
    uniform jitter. Up to two control periods; each camera has its own values. Motion blur: the frame
    averages `blur_samples` renders spread over an exposure time drawn per episode from `exposure_s`.
    """

    latency_s: list[float] = field(default_factory=lambda: [0.0, 0.0])
    latency_jitter_s: float = 0.0
    exposure_s: list[float] = field(default_factory=lambda: [0.0, 0.0])
    blur_samples: int = 3

    @property
    def enabled(self) -> bool:
        return self.latency_s[1] > 0 or self.latency_jitter_s > 0 or self.exposure_s[1] > 0


@dataclass
class PhysicsRandomizationConfig:
    """Per-episode physics randomization (factors on the scene's nominal values)."""

    enabled: bool = False
    # Servo position gain (kp) and damping (kv) of every actuator
    actuator_kp_scale: list[float] = field(default_factory=lambda: [0.75, 1.25])
    actuator_kv_scale: list[float] = field(default_factory=lambda: [0.75, 1.25])
    # Joint dry friction, viscous damping and rotor inertia (armature)
    joint_friction_scale: list[float] = field(default_factory=lambda: [0.5, 2.0])
    joint_damping_scale: list[float] = field(default_factory=lambda: [0.75, 1.25])
    joint_armature_scale: list[float] = field(default_factory=lambda: [0.8, 1.2])
    # Masses (inertias scaled alike) of the arm links and of the object
    link_mass_scale: list[float] = field(default_factory=lambda: [0.8, 1.2])
    object_mass_scale: list[float] = field(default_factory=lambda: [0.7, 1.3])
    # Sliding friction of the object and of the finger pads
    object_friction_scale: list[float] = field(default_factory=lambda: [0.6, 1.4])
    pad_friction_scale: list[float] = field(default_factory=lambda: [0.8, 1.2])


@dataclass
class ObservationNoiseConfig:
    """Encoder errors on the measured joint positions (robot units: degrees, gripper 0-100): a bias per
    episode (calibration offset) plus noise per reading. They reach the student's state (velocities
    are finite differences of the noisy positions, as on the real pipeline) and the action pipeline's
    measured joints (gripper command, first IK reference), not the teacher's privileged state."""

    enabled: bool = False
    joint_bias_deg: float = 1.0
    joint_noise_deg: float = 0.3
    gripper_bias: float = 1.0
    gripper_noise: float = 0.5


@dataclass
class SO101PickPlaceEnvConfig:
    """The `so101_sim_follower` pick-and-place scene driven through the HIL-SERL action pipeline.

    Defaults mirror `src/lerobot/configs/env_config_so101_sim.json`: EE-delta actions with the same
    step sizes / bounds, a discrete gripper clipped at `max_gripper_pos`, the same reset pose, and
    student images cropped/resized exactly like `image_preprocessing` there.
    """

    xml_path: str = "Simulation/SO101/so101_new_calib.xml"
    urdf_path: str = "Simulation/SO101/so101_new_calib_camera.urdf"
    target_frame_name: str = "gripper_frame_link"
    # Control rate; one action advances the physics by 1 / fps
    fps: int = 10
    episode_length_s: float = 10.0
    reset_joint_positions: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 90.0, 0.0, 5.0])
    cube_yaw_range_deg: float = 15.0
    # Ring center (x, y) in the robot base frame; None keeps the `so101_sim_follower` default
    target_center: list[float] | None = None
    # Object to pick: full (x, y, z) size in meters, x along the gripper's closing direction (e.g.
    # [0.01, 0.02, 0.005] to grasp a 2 x 1 x 0.5 cm bar across its width). None keeps the 2 cm cube
    object_size: list[float] | None = None
    # After placing the object, the gripper must return to its reset pose to succeed (and scores more
    # the closer it gets, see `RewardConfig.home_weight`)
    return_home: bool = False
    # Height of the robot base above the table (m), see `SO101SimFollowerConfig.robot_base_z`. Actions,
    # EE bounds and the IK work in the base frame like on the real robot
    robot_base_z: float = 0.0
    # Object spawn area (x, y) in the robot base frame; None keeps the `so101_sim_follower` default
    # ([0.12, 0.10] to [0.24, 0.22])
    object_spawn_min: list[float] | None = None
    object_spawn_max: list[float] | None = None
    # Per-episode uniform offset (m, on x and y) of the ring around `target_center`, for robustness
    # to where the ring sits on the real table
    ring_position_range: float = 0.0
    ee_step_sizes: list[float] = field(default_factory=lambda: [0.02, 0.02, 0.02])
    ee_bounds_min: list[float] = field(default_factory=lambda: [0.0673, -0.1982, 0.005])
    ee_bounds_max: list[float] = field(default_factory=lambda: [0.3155, 0.2947, 0.1477])
    max_ee_step_m: float = 0.05
    max_gripper_pos: float = 30.0
    # Adds a wrist-roll delta (joint space, `wrist_roll_step_deg` per unit) as a 4th continuous action.
    # Off by default to keep the 3-D EE action of the HIL-SERL pipeline; when enabling it, widen
    # `cube_yaw_range_deg` so rotating the gripper matters.
    use_wrist_roll: bool = False
    wrist_roll_step_deg: float = 10.0
    terminate_on_success: bool = True

    # Camera images for the student (rendered only when a vec env is created with render=True).
    # Rendered at the real camera resolution, then cropped (top, left, height, width) and resized.
    image_width: int = 640
    image_height: int = 480
    image_resize: list[int] = field(default_factory=lambda: [128, 128])
    # Per-camera (height, width) overriding `image_resize`, e.g. to keep each crop's aspect ratio. The
    # HIL-SERL `image_preprocessing` has a single size; `deploy` resizes such students' images itself
    front_resize: list[int] | None = None
    wrist_resize: list[int] | None = None
    front_crop: list[int] | None = field(default_factory=lambda: [0, 0, 379, 575])
    wrist_crop: list[int] | None = None
    visual_randomization: VisualRandomizationConfig = field(default_factory=VisualRandomizationConfig)
    camera_timing: CameraTimingConfig = field(default_factory=CameraTimingConfig)
    physics_randomization: PhysicsRandomizationConfig = field(default_factory=PhysicsRandomizationConfig)
    observation_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)
    # Actuation latency: each episode executes the commanded actions a random 0..action_delay_max
    # control steps late (0 disables it)
    action_delay_max: int = 0

    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def max_episode_steps(self) -> int:
        return round(self.episode_length_s * self.fps)

    def image_size(self, camera: str) -> list[int]:
        """(height, width) of a camera's student image."""
        size = {"front": self.front_resize, "wrist": self.wrist_resize}[camera]
        return list(self.image_resize if size is None else size)

    @property
    def num_continuous_actions(self) -> int:
        """EE delta (x, y, z) and optionally the wrist roll; the discrete gripper comes last."""
        return 4 if self.use_wrist_roll else 3


@dataclass
class PPOConfig:
    """PPO hyperparameters (defaults follow OpenSO-101's rsl_rl PickPlace config where they transfer)."""

    num_steps_per_env: int = 64
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    # Adaptive learning rate (rsl_rl "adaptive" schedule); None keeps it fixed
    desired_kl: float | None = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    init_noise_std: float = 0.8
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])


@dataclass
class TrainPPOConfig:
    env: SO101PickPlaceEnvConfig = field(default_factory=SO101PickPlaceEnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    num_envs: int = 32
    # Worker processes stepping the envs in parallel (each owns num_envs / num_workers envs). Use
    # about the number of physical cores that stay free; on a laptop more workers can be slower.
    num_workers: int = 4
    max_iterations: int = 3000
    save_interval: int = 100
    seed: int = 0
    device: str = "cuda"
    output_dir: str = "outputs/so101_mujoco/ppo"
    # Resume from a checkpoint written by this script
    resume: str | None = None
    wandb: bool = False
    wandb_project: str = "so101_mujoco"


@dataclass
class DistillConfig:
    """DAgger distillation: the student drives the envs, the teacher labels every visited state."""

    num_steps_per_env: int = 32
    # Probability of executing the teacher action instead of the student's, decayed linearly to 0
    teacher_mix_start: float = 1.0
    teacher_mix_decay_iterations: int = 50
    # Aggregated DAgger dataset kept on CPU (two uint8 128x128 images per sample ~ 98 kB)
    buffer_capacity: int = 20000
    batch_size: int = 256
    updates_per_iteration: int = 32
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    # Weight of the gripper cross-entropy against the EE-delta MSE
    gripper_loss_coef: float = 0.5
    # Random-shift augmentation (pixels of padding) applied to both cameras during training
    image_shift_pad: int = 4
    # Hide the 6 emulated motor currents of `observation.state` from the student (their real
    # calibration is unknown); applied inside the network, so deployment does it too
    drop_current: bool = True
    image_feature_dim: int = 128
    head_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])


@dataclass
class TrainDistillConfig:
    # PPO checkpoint (model_best.pt / model_*.pt) of the teacher
    teacher_checkpoint: str = ""
    # Student checkpoint to start from (e.g. continue a distillation with visual randomization on)
    init_checkpoint: str | None = None
    env: SO101PickPlaceEnvConfig = field(default_factory=SO101PickPlaceEnvConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    num_envs: int = 16
    num_workers: int = 4
    max_iterations: int = 300
    save_interval: int = 25
    seed: int = 1
    device: str = "cuda"
    output_dir: str = "outputs/so101_mujoco/distill"
    wandb: bool = False
    wandb_project: str = "so101_mujoco"


@dataclass
class EvalConfig:
    # A PPO (teacher) or distillation (student) checkpoint; the type is read from the file
    checkpoint: str = ""
    env: SO101PickPlaceEnvConfig = field(default_factory=SO101PickPlaceEnvConfig)
    num_episodes: int = 50
    num_envs: int = 10
    num_workers: int = 4
    seed: int = 1000
    device: str = "cuda"
    # Run a single env in this process with the interactive MuJoCo viewer instead
    viewer: bool = False
    # Save the front + wrist camera stream of the first episodes to this .mp4
    video_path: str | None = None
    video_episodes: int = 3
