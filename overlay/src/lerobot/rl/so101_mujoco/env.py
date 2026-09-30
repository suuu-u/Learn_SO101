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

"""Single MuJoCo SO101 pick-and-place env for RL.

The scene, units and success test come from `SO101SimFollower` (the `so101_sim_follower` robot), and
actions go through a replica of the HIL-SERL action pipeline of `gym_manipulator`
(`EEReferenceAndDelta` -> `EEBoundsAndSafety` -> `GripperVelocityToJoint` -> `InverseKinematicsRLStep`),
so a policy trained here speaks the same action language as one trained with `lerobot.rl`.

Action: `[dx, dy, dz, gripper]`, the EE delta in [-1, 1] (scaled by `ee_step_sizes`) and the discrete
gripper command {0, 1 = stay, 2} of `GripperVelocityToJoint(discrete_gripper=True)`. In this scene
0 drives the gripper to `max_gripper_pos` (open) and 2 to 0 (closed). With `use_wrist_roll` the
action is `[dx, dy, dz, droll, gripper]`, `droll` in [-1, 1] turning the wrist-roll joint target by
`wrist_roll_step_deg` (not part of the HIL-SERL pipeline).

Observation (dict):
    "teacher": privileged state (cube pose, contacts, IK reference, ...) for the PPO teacher.
    "state":   the 18-dim `observation.state` of the HIL-SERL pipeline: joint positions (degrees,
               gripper 0-100), their finite-difference velocities and the emulated motor currents.
    "front", "wrist": uint8 (3, H, W) camera images, cropped and resized like `image_preprocessing`
               (only when the env renders).
"""

import contextlib
import ctypes
import re
from collections import deque
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_sim import SO101SimFollower, SO101SimFollowerConfig
from lerobot.robots.so101_sim.so101_sim import CAMERAS, MOTORS, footprint_overshoot

from .configs import SO101PickPlaceEnvConfig, resolve_path
from .randomization import (
    ImageCorruptor,
    JointReadingNoise,
    PhysicsRandomizer,
    VisualRandomizer,
    add_lighting_variety,
    add_visual_variety,
    restore_appearance,
    restore_physics,
)

# Discrete gripper commands of `GripperVelocityToJoint`
GRIPPER_STAY = 1
NUM_GRIPPER_ACTIONS = 3
# Stage value of a placed object (see `RewardConfig`)
S_PLACED = 6.0


# Per-process caches: the compiled scene (~200 MB with its meshes), its renderer and the placo solver
# are shared by all envs of a worker process. Sharing is safe because envs run sequentially and every
# FK / IK call sets the solver's joints first.
_MODELS: dict = {}
_RENDERERS: dict = {}
_KINEMATICS: dict = {}


class _MeshFreeKinematics(RobotKinematics):
    """`RobotKinematics` on the URDF stripped of its visual / collision meshes.

    FK / IK are identical (they only use the kinematic tree), but placo no longer loads every mesh,
    which saves ~440 MB per process.
    """

    def __init__(self, urdf_path: str, target_frame_name: str, joint_names: list[str]):
        import placo

        urdf = re.sub(r"<(visual|collision)\b.*?</\1>", "", Path(urdf_path).read_text(), flags=re.S)
        self.robot = placo.RobotWrapper(urdf_path, 0, urdf)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.target_frame_name = target_frame_name
        self.joint_names = joint_names
        self.tip_frame = self.solver.add_frame_task(target_frame_name, np.eye(4))


def _shared_kinematics(urdf_path: str, target_frame_name: str) -> RobotKinematics:
    key = (urdf_path, target_frame_name)
    if key not in _KINEMATICS:
        _KINEMATICS[key] = _MeshFreeKinematics(urdf_path, target_frame_name, list(MOTORS))
    return _KINEMATICS[key]


def _extended_model(robot: SO101SimFollower, visual_variety: bool, lighting_variety: bool):
    """The robot's scene plus the extra assets of the visual randomization: table textures and
    distractor objects, a side lamp and a glossy table material."""
    spec = robot._build_spec()
    if visual_variety:
        add_visual_variety(spec)
    if lighting_variety:
        add_lighting_variety(spec)
    return spec.compile()


class _ViewerSO101Sim(SO101SimFollower):
    """The stock robot (interactive viewer) with the randomization assets in its scene. The env steps
    a control period in several pieces (camera timing), so the viewer is locked and synced once per
    period (`stepping`) rather than per piece, which waits for the viewer's frame every time."""

    def __init__(self, config: SO101SimFollowerConfig, visual_variety: bool, lighting_variety: bool):
        super().__init__(config)
        self._variety = (visual_variety, lighting_variety)

    def _build_model(self):
        return _extended_model(self, *self._variety)

    def _step(self, duration: float) -> None:
        import mujoco

        dt = self.model.opt.timestep
        for _ in range(max(1, round(duration / dt))):
            mujoco.mj_step(self.model, self.data)
            self._placed_time = self._placed_time + dt if self._is_placed() else 0.0

    @contextlib.contextmanager
    def stepping(self):
        with self._viewer.lock() if self._viewer is not None else contextlib.nullcontext():
            yield
        if self._viewer is not None:
            self._viewer.sync()


class _FastSO101Sim(SO101SimFollower):
    """`SO101SimFollower` without viewer handling, sharing its model / renderer across the process,
    with a leaner physics loop."""

    def __init__(
        self,
        config: SO101SimFollowerConfig,
        render: bool,
        visual_variety: bool = False,
        lighting_variety: bool = False,
    ):
        super().__init__(config)
        self._render_enabled = render
        self._variety = (visual_variety, lighting_variety)

    def _build_model(self):
        return _extended_model(self, *self._variety)

    def connect(self, calibrate: bool = True) -> None:
        import mujoco

        # The scene depends on everything but the seed
        key = repr({k: v for k, v in vars(self.config).items() if k != "seed"}) + f"{self._variety}"
        if key not in _MODELS:
            _MODELS[key] = self._build_model()
            # Hand the scene compiler's scratch memory (~100 MB) back to the OS
            with contextlib.suppress(OSError, AttributeError):
                ctypes.CDLL("libc.so.6").malloc_trim(0)
        self.model = _MODELS[key]
        self.data = mujoco.MjData(self.model)
        self._joint_qpos = np.array([self.model.joint(m).qposadr[0] for m in MOTORS])
        self._joint_dof = np.array([self.model.joint(m).dofadr[0] for m in MOTORS])
        self._actuators = np.array([self.model.actuator(m).id for m in MOTORS])
        self._gripper_range = self.model.jnt_range[self.model.joint("gripper").id].copy()
        self._cube_qpos = self.model.joint("cube").qposadr[0]
        self._cube_dof = self.model.joint("cube").dofadr[0]
        if self._render_enabled:
            if key not in _RENDERERS:
                _RENDERERS[key] = mujoco.Renderer(
                    self.model, height=self.config.image_height, width=self.config.image_width
                )
            self._renderer = _RENDERERS[key]
        # Live view of the ring's mocap position, so moving the ring updates the placement test
        self._ring_center = self.data.mocap_pos[self.model.body("ring").mocapid[0], :2]
        self._half_extents = self.config.object_half_extents

    def disconnect(self) -> None:
        # The shared model / renderer stay cached for the other envs of the process
        self._renderer = None
        self.model = None
        self.data = None

    def _is_placed(self) -> bool:
        # Same test as the parent, cheapest checks first (it runs every physics step)
        qpos = self.data.qpos[self._cube_qpos : self._cube_qpos + 7]
        v = self.data.qvel[self._cube_dof : self._cube_dof + 3]
        if float(v @ v) >= 0.02**2:
            return False
        inner = self.config.target_inner_radius
        return footprint_overshoot(qpos[:3], qpos[3:], self._half_extents, self._ring_center, inner) == 0.0

    def _step(self, duration: float) -> None:
        import mujoco

        dt = self.model.opt.timestep
        for _ in range(max(1, round(duration / dt))):
            mujoco.mj_step(self.model, self.data)
            self._placed_time = self._placed_time + dt if self._is_placed() else 0.0


class SO101PickPlaceEnv:
    """Gymnasium-style (reset/step) env; `render` adds the student camera images to observations."""

    def __init__(
        self,
        cfg: SO101PickPlaceEnvConfig,
        seed: int | None = None,
        render: bool = False,
        show_viewer: bool = False,
    ):
        self.cfg = cfg
        self.render = render or show_viewer
        sim_cfg = SO101SimFollowerConfig(
            xml_path=resolve_path(cfg.xml_path),
            fps=cfg.fps,
            show_viewer=show_viewer,
            seed=seed,
            image_width=cfg.image_width,
            image_height=cfg.image_height,
            cube_yaw_range_deg=cfg.cube_yaw_range_deg,
            object_size=None if cfg.object_size is None else list(cfg.object_size),
            return_home=cfg.return_home,
            robot_base_z=cfg.robot_base_z,
        )
        if cfg.target_center is not None:
            sim_cfg.target_center = list(cfg.target_center)
        if cfg.object_spawn_min is not None:
            sim_cfg.cube_spawn_min = list(cfg.object_spawn_min)
        if cfg.object_spawn_max is not None:
            sim_cfg.cube_spawn_max = list(cfg.object_spawn_max)
        vis_cfg = cfg.visual_randomization
        strong_visuals = vis_cfg.table_texture_prob > 0 or vis_cfg.max_distractors > 0
        visual_variety = vis_cfg.enabled and self.render and strong_visuals
        lighting_variety = vis_cfg.enabled and self.render and vis_cfg.lighting_variety
        self._spawn_center = (np.array(sim_cfg.cube_spawn_min) + np.array(sim_cfg.cube_spawn_max)) / 2
        self._spawn_radius = float(np.linalg.norm(np.array(sim_cfg.cube_spawn_max) - self._spawn_center))
        # The interactive viewer needs the stock robot; training uses the faster headless variant
        if show_viewer:
            self.robot = _ViewerSO101Sim(sim_cfg, visual_variety, lighting_variety)
        else:
            self.robot = _FastSO101Sim(sim_cfg, render, visual_variety, lighting_variety)
        self.robot.connect()
        self.kinematics = _shared_kinematics(resolve_path(cfg.urdf_path), cfg.target_frame_name)

        model = self.robot.model
        self._ee_site = model.site("gripperframe").id
        self._cube_geom = model.geom("cube").id
        self._fixed_pad = model.geom("fixed_pad").id
        self._moving_pad = model.geom("moving_pad").id
        self._ring_mocap = model.body("ring").mocapid[0]
        self._ring_base = np.array(sim_cfg.target_center)
        # Live view of the ring position (moved per episode with `ring_position_range`)
        self._ring_center = self.robot.data.mocap_pos[self._ring_mocap, :2]
        self._ring_rng = np.random.default_rng(None if seed is None else seed + 1)
        def rng(offset: int) -> np.random.Generator:
            return np.random.default_rng(None if seed is None else seed + offset)

        self._visuals = self._image_corruptor = None
        if vis_cfg.enabled and self.render:
            self._visuals = VisualRandomizer(model, vis_cfg, rng(2))
            self._image_corruptor = ImageCorruptor(vis_cfg, rng(3))
        self._physics = None
        if cfg.physics_randomization.enabled:
            self._physics = PhysicsRandomizer(model, cfg.physics_randomization, rng(4))
        self._reading_noise = None
        if cfg.observation_noise.enabled:
            self._reading_noise = JointReadingNoise(cfg.observation_noise, rng(5), len(MOTORS))
        self._delay_rng = rng(6)
        # Camera latency / motion blur: frames are captured while the physics steps (see `_advance`)
        self._timing = cfg.camera_timing if self.render and cfg.camera_timing.enabled else None
        self._timing_rng = rng(7)
        self._latency: dict[str, float] = {}
        self._exposure: dict[str, float] = {}
        self._next_latency: dict[str, float] = {}
        # Raw (H, W, 3) frame each camera shows after the current step, and the frame of the next step
        # when it was captured during this one (latency above a control period)
        self._frames: dict[str, np.ndarray] = {}
        self._early_frames: dict[str, np.ndarray | None] = {}
        self._pending_actions: deque = deque()
        half = sim_cfg.object_half_extents
        self._ring_inside = sim_cfg.target_inner_radius - max(half[0], half[1])
        # Release / carry region: any yaw of an object centered inside stays clear of the ring wall
        self._cube_half = half[2]
        self._carry_height = sim_cfg.target_height + half[2] + cfg.reward.carry_clearance
        # Yaw symmetry of the object: 90 deg for a square footprint, 180 deg otherwise
        self._yaw_symmetry = 4.0 if np.isclose(half[0], half[1]) else 2.0
        self._s_max = S_PLACED + (cfg.reward.home_weight if cfg.return_home else 0.0)

        self._num_continuous = cfg.num_continuous_actions
        self._roll_index = MOTORS.index("wrist_roll")
        self._roll_range_deg = np.rad2deg(model.jnt_range[model.joint("wrist_roll").id])
        self._step_sizes = np.array(cfg.ee_step_sizes, dtype=float)
        self._bounds_min = np.array(cfg.ee_bounds_min, dtype=float)
        self._bounds_max = np.array(cfg.ee_bounds_max, dtype=float)
        self._dt = 1.0 / cfg.fps
        # The kinematics work in the robot base frame, the scene in the world (table) frame
        self._base_offset = np.array([0.0, 0.0, cfg.robot_base_z])
        self.max_episode_steps = cfg.max_episode_steps

        self._q_ik: np.ndarray | None = None
        self._last_cmd_pos: np.ndarray | None = None
        self._gripper_target: float | None = None
        self._last_positions: np.ndarray | None = None
        self._prev_delta = np.zeros(self._num_continuous)
        self._prev_gripper = GRIPPER_STAY
        self._steps = 0
        self._prev_stage = 0.0
        self._bound_hits = 0
        self._low_ee_steps = 0

    def close(self) -> None:
        self.robot.disconnect()

    # ------------------------------------------------------------------ gym API

    def reset(self) -> dict[str, np.ndarray]:
        import mujoco

        self._use_physics(sample=True)
        self.robot.reset_scene(list(self.cfg.reset_joint_positions))
        if self.cfg.ring_position_range > 0:
            # The ring is kinematic (mocap) and away from the cube spawn area, so it can be moved
            # after the cube has settled
            r = self.cfg.ring_position_range
            offset = self._ring_rng.uniform(-r, r, 2)
            self.robot.data.mocap_pos[self._ring_mocap, :2] = self._ring_base + offset
            mujoco.mj_forward(self.robot.model, self.robot.data)
        if self._visuals is not None:
            self._visuals.sample()
            if self._visuals.distractor_mocap:
                keep_clear = [
                    (self._ring_center.copy(), 0.09),
                    (self._spawn_center, self._spawn_radius + 0.04),
                    (np.zeros(2), 0.1),
                ]
                self._visuals.place_distractors(self.robot.data, keep_clear)
                mujoco.mj_forward(self.robot.model, self.robot.data)
        if self._image_corruptor is not None:
            self._image_corruptor.sample()
        if self._reading_noise is not None:
            self._reading_noise.sample()
        # Actuation latency of this episode: commands wait `delay` steps (the first ones hold still)
        delay = int(self._delay_rng.integers(0, self.cfg.action_delay_max + 1))
        still = np.r_[np.zeros(self._num_continuous), GRIPPER_STAY]
        self._pending_actions = deque([still] * delay)
        # The pipeline processors reset with the episode: the first reference is the measured pose
        self._q_ik = self._positions()
        self._last_cmd_pos = None
        self._gripper_target = None
        self._last_positions = None
        self._prev_delta = np.zeros(self._num_continuous)
        self._prev_gripper = GRIPPER_STAY
        self._steps = 0
        self._low_ee_steps = 0
        if self._timing is not None:
            self._reset_camera_timing()
        contacts = self._contacts()
        self._prev_stage = self._stage(contacts, placed=False)
        return self._observe(contacts)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        action = np.asarray(action, dtype=float)
        delta = np.clip(action[: self._num_continuous], -1.0, 1.0)
        gripper = int(np.clip(np.rint(action[-1]), 0, NUM_GRIPPER_ACTIONS - 1))
        action_rate = float(np.sum((delta - self._prev_delta) ** 2))
        self._use_physics()
        # Actuation latency: execute the command issued `delay` steps ago
        self._pending_actions.append(np.r_[delta, gripper])
        executed = self._pending_actions.popleft()
        self._apply_action(executed[:-1], int(executed[-1]))
        self._steps += 1
        self._prev_delta = delta
        self._prev_gripper = gripper

        contacts = self._contacts()
        success = self.robot.is_success()
        placed = self.robot._is_placed()
        stage = self._stage(contacts, placed)
        reward = self._reward(stage, success)
        self._prev_stage = stage
        rcfg = self.cfg.reward
        reward -= rcfg.reference_penalty * max(0.0, self._reference_offset() - rcfg.reference_tolerance)
        reward -= rcfg.bound_penalty * self._bound_hits
        reward -= rcfg.action_rate_penalty * action_rate
        reward -= self._low_ee_penalty()
        if gripper != GRIPPER_STAY:
            reward -= rcfg.gripper_penalty

        terminated = success and self.cfg.terminate_on_success
        truncated = not terminated and self._steps >= self.max_episode_steps
        info = {"success": success, "placed": placed, "grasped": bool(contacts[0] and contacts[1])}
        return self._observe(contacts), reward, terminated, truncated, info

    # ------------------------------------------------------------------ action pipeline

    def _apply_action(self, delta: np.ndarray, gripper: int) -> None:
        cfg = self.cfg
        measured = self._positions()

        # EEReferenceAndDelta(use_latched_reference=False, use_ik_solution=True): the reference is the
        # FK of the previous IK solution and the orientation is kept (zero rotation delta)
        t_des = self.kinematics.forward_kinematics(self._q_ik)
        pos = t_des[:3, 3] + delta[:3] * self._step_sizes

        # EEBoundsAndSafety(raise_on_jump=False): clip to the workspace, rate-limit the step
        clipped = np.clip(pos, self._bounds_min, self._bounds_max)
        # Axes pushed into a bound, except the floor (see `RewardConfig.bound_penalty`)
        self._bound_hits = int(np.sum(clipped[:2] != pos[:2]) + (pos[2] > self._bounds_max[2]))
        pos = clipped
        if self._last_cmd_pos is not None:
            step = pos - self._last_cmd_pos
            norm = float(np.linalg.norm(step))
            if norm > cfg.max_ee_step_m:
                pos = self._last_cmd_pos + step * (cfg.max_ee_step_m / norm)
        self._last_cmd_pos = pos
        t_des[:3, 3] = pos

        # GripperVelocityToJoint(discrete_gripper=True, hold_last_target=True, speed_factor=1)
        gripper_vel = -(gripper - 1) * cfg.max_gripper_pos
        gripper_pos = float(np.clip(measured[-1] + gripper_vel, 0.0, cfg.max_gripper_pos))
        if gripper_vel != 0 or self._gripper_target is None:
            self._gripper_target = gripper_pos

        # InverseKinematicsRLStep(initial_guess_current_joints=False)
        q_target = self.kinematics.inverse_kinematics(self._q_ik, t_des)
        if self._num_continuous == 4:
            # Joint-space roll on top of the IK solution; it enters the next FK reference, so the
            # (soft) orientation task holds the new roll instead of undoing it
            roll = self._q_ik[self._roll_index] + delta[3] * cfg.wrist_roll_step_deg
            q_target[self._roll_index] = np.clip(roll, *self._roll_range_deg)
        self._q_ik = q_target
        goal = q_target.copy()
        goal[-1] = self._gripper_target
        self.robot._set_targets(dict(zip(MOTORS, goal, strict=True)))
        self._advance()

    # ------------------------------------------------------------------ camera timing

    def _reset_camera_timing(self) -> None:
        """Per-episode latency / exposure of each camera; the scene is still, so every frame until the
        first step shows the reset pose."""
        timing, rng = self._timing, self._timing_rng
        self._apply_appearance()
        for camera in CAMERAS:
            self._latency[camera] = float(rng.uniform(*timing.latency_s))
            self._exposure[camera] = float(rng.uniform(*timing.exposure_s))
            self._next_latency[camera] = self._step_latency(camera)
            self._frames[camera] = self.robot._render(camera)
            self._early_frames[camera] = self._frames[camera]

    def _step_latency(self, camera: str) -> float:
        jitter = self._timing.latency_jitter_s
        latency = self._latency[camera] + float(self._timing_rng.uniform(-jitter, jitter))
        # Captured during this control period or the previous one
        return float(np.clip(latency, 0.0, 2 * self._dt - self.robot.model.opt.timestep))

    def _exposure_substeps(self, camera: str, capture: float, n: int, timestep: float) -> list[int]:
        """Physics substeps (0..n, from the start of the step) to render at for a frame captured at
        `capture` s into the step: spread over the exposure before it (clipped to the step)."""
        exposure = self._exposure[camera]
        samples = self._timing.blur_samples if exposure > 0 else 1
        times = [capture - exposure * i / max(1, samples - 1) for i in range(samples)]
        return [int(np.clip(round(t / timestep), 0, n)) for t in times]

    def _advance(self) -> None:
        """Step the physics for one control period. With camera timing, render each camera at the
        moments its frames are exposed: this step's frame `latency` before the end of the step (or,
        above one period, during the previous step), and the next step's frame when it falls in this
        one; the renders of a frame are averaged (motion blur)."""
        stepping = getattr(self.robot, "stepping", contextlib.nullcontext)
        with stepping():
            self._advance_period()

    def _advance_period(self) -> None:
        if self._timing is None:
            self.robot._step(self._dt)
            return
        timestep = self.robot.model.opt.timestep
        n = max(1, round(self._dt / timestep))
        renders: dict[int, list[tuple[str, str]]] = {}
        for camera in CAMERAS:
            latency = self._next_latency[camera]
            self._next_latency[camera] = self._step_latency(camera)
            if latency < self._dt:
                for substep in self._exposure_substeps(camera, self._dt - latency, n, timestep):
                    renders.setdefault(substep, []).append((camera, "now"))
            if self._next_latency[camera] >= self._dt:
                capture = 2 * self._dt - self._next_latency[camera]
                for substep in self._exposure_substeps(camera, capture, n, timestep):
                    renders.setdefault(substep, []).append((camera, "next"))

        sums: dict[tuple[str, str], list] = {}
        done = 0
        for substep in sorted(renders):
            if substep > done:
                self.robot._step((substep - done) * timestep)
                done = substep
            self._apply_appearance()
            for camera, slot in renders[substep]:
                frame = self.robot._render(camera).astype(np.float32)
                total = sums.setdefault((camera, slot), [0.0, 0])
                total[0] = total[0] + frame
                total[1] += 1
        if done < n:
            self.robot._step((n - done) * timestep)

        def average(key: tuple[str, str]) -> np.ndarray | None:
            if key not in sums:
                return None
            total, count = sums[key]
            return np.clip(total / count + 0.5, 0, 255).astype(np.uint8)

        for camera in CAMERAS:
            now = average((camera, "now"))
            early = self._early_frames[camera]
            if now is not None:
                self._frames[camera] = now
            elif early is not None:
                self._frames[camera] = early
            self._early_frames[camera] = average((camera, "next"))

    # ------------------------------------------------------------------ observations and reward

    def _positions(self) -> np.ndarray:
        """Measured joint positions (robot units), with encoder errors when `observation_noise` is on."""
        positions = np.array(list(self.robot._read_positions().values()))
        return positions if self._reading_noise is None else self._reading_noise(positions)

    def _pending_features(self) -> np.ndarray:
        """The commands still waiting because of the actuation delay, in execution order (privileged:
        with them the teacher can compensate the latency). Per slot: the EE delta, the gripper command
        / 2 and whether the slot is used; empty without `action_delay_max`."""
        slot = self._num_continuous + 2
        features = np.zeros(self.cfg.action_delay_max * slot)
        for i, action in enumerate(self._pending_actions):
            features[i * slot : (i + 1) * slot] = np.r_[action[:-1], action[-1] / 2.0, 1.0]
        return features

    def _use_physics(self, sample: bool = False) -> None:
        """Put this env's physics into the (shared) model before stepping it."""
        if self._physics is None:
            restore_physics(self.robot.model)
            return
        if sample:
            self._physics.sample()
        self._physics.apply()

    def _grasp_center(self) -> np.ndarray:
        """Midpoint of the finger pads, where the cube sits in a grasp.

        The `gripperframe` site is on the fixed jaw, ~2 cm off this point, so reaching the site to
        the cube would press the fixed pad onto it.
        """
        xpos = self.robot.data.geom_xpos
        return 0.5 * (xpos[self._fixed_pad] + xpos[self._moving_pad])

    def _contacts(self) -> tuple[bool, bool]:
        """Whether the fixed and the moving finger pad touch the cube."""
        data = self.robot.data
        if data.ncon == 0:
            return False, False
        pairs = data.contact.geom[: data.ncon]
        on_cube = (pairs[:, 0] == self._cube_geom) | (pairs[:, 1] == self._cube_geom)
        others = pairs[on_cube].ravel()
        return bool(np.any(others == self._fixed_pad)), bool(np.any(others == self._moving_pad))

    def _observe(self, contacts: tuple[bool, bool]) -> dict[str, np.ndarray]:
        robot = self.robot
        data = robot.data

        positions = self._positions()
        velocities = (
            np.zeros_like(positions)
            if self._last_positions is None
            else (positions - self._last_positions) / self._dt
        )
        self._last_positions = positions
        currents = np.array(list(robot._read_currents().values()))
        obs = {"state": np.concatenate([positions, velocities, currents]).astype(np.float32)}

        ee = data.site_xpos[self._ee_site]
        cube = data.qpos[robot._cube_qpos : robot._cube_qpos + 3]
        qw, qz = data.qpos[robot._cube_qpos + 3], data.qpos[robot._cube_qpos + 6]
        # The cube looks the same every 90 degrees of yaw
        yaw_feature = self._yaw_symmetry * 2.0 * np.arctan2(qz, qw)
        gripper_onehot = np.zeros(NUM_GRIPPER_ACTIONS)
        gripper_onehot[self._prev_gripper] = 1.0
        # Reference of the next EE delta: the FK of the last IK solution, which drifts from the
        # measured pose while the arm is blocked or lags (hidden state of the HIL-SERL pipeline)
        reference = self.kinematics.forward_kinematics(self._q_ik)[:3, 3] + self._base_offset
        obs["teacher"] = np.concatenate(
            [
                ee,
                reference,
                reference - ee,
                cube,
                cube - self._grasp_center(),
                self._ring_center - cube[:2],
                [np.sin(yaw_feature), np.cos(yaw_feature)],
                data.qvel[robot._cube_dof : robot._cube_dof + 3],
                data.qpos[robot._joint_qpos],
                data.qvel[robot._joint_dof],
                [float(contacts[0]), float(contacts[1])],
                self._prev_delta,
                gripper_onehot,
                [(self._gripper_target or 0.0) / self.cfg.max_gripper_pos],
                self._pending_features(),
            ]
        ).astype(np.float32)

        if self.render:
            if self._timing is None:
                self._apply_appearance()
            for camera in CAMERAS:
                frame = self._frames[camera] if self._timing is not None else self.robot._render(camera)
                obs[camera] = self._image(camera, frame)
        return obs

    def _apply_appearance(self) -> None:
        """Put this env's appearance into the (shared) model before rendering."""
        import mujoco

        if self._visuals is not None:
            self._visuals.apply()
        else:
            restore_appearance(self.robot.model)
        # Rendering uses the camera / light poses of `data`, computed from the model at the last physics
        # step (possibly with another env's appearance in the shared model): recompute them
        mujoco.mj_camlight(self.robot.model, self.robot.data)

    def _image(self, camera: str, frame: np.ndarray) -> np.ndarray:
        """Crop / resize a rendered frame like `ImageCropResizeProcessorStep`, then corrupt it."""
        import torch
        import torchvision.transforms.functional as F  # noqa: N812

        crop = {"front": self.cfg.front_crop, "wrist": self.cfg.wrist_crop}[camera]
        image = torch.from_numpy(frame).permute(2, 0, 1)
        if crop is not None:
            image = F.crop(image, *crop)
        image = F.resize(image, self.cfg.image_size(camera), antialias=True).numpy()
        return image if self._image_corruptor is None else self._image_corruptor(image, camera)

    def _stage(self, contacts: tuple[bool, bool], placed: bool) -> float:
        """Task progress S in [0, S_max] (see `RewardConfig`)."""
        rcfg = self.cfg.reward
        # An object in the ring and not held counts as placed for the reward, even before it has
        # settled: otherwise, until it is still, only the reach term is left, which pulls the gripper
        # back to the object and makes re-grasping it pay (success still needs it to be still)
        held = contacts[0] and contacts[1]
        if placed or (not held and self.robot.footprint_overshoot() == 0.0):
            if not self.cfg.return_home:
                return S_PLACED
            distance = float(np.linalg.norm(self.robot.data.site_xpos[self._ee_site] - self.robot._home_ee))
            coarse = max(0.0, 1.0 - distance / rcfg.home_radius)
            fine = 1.0 - float(np.tanh(distance / rcfg.home_precision))
            return S_PLACED + rcfg.home_weight * 0.5 * (coarse + fine)
        cube = self.robot.data.qpos[self.robot._cube_qpos : self.robot._cube_qpos + 3]

        stage = max(0.0, 1.0 - float(np.linalg.norm(self._grasp_center() - cube)) / rcfg.reach_radius)
        if contacts[0] and contacts[1]:
            ring_dist = float(np.linalg.norm(cube[:2] - self._ring_center))
            # Over the ring, lowering the cube must not cost the lift term
            if ring_dist < self._ring_inside:
                lift = 1.0
            else:
                lift = float(np.clip((cube[2] - self._cube_half) / rcfg.lift_height, 0.0, 1.0))
            transport = max(0.0, 1.0 - self._carry_distance(cube) / rcfg.transport_radius)
            stage += 1.0 + lift + rcfg.transport_weight * transport
        return float(stage)

    def _carry_distance(self, cube: np.ndarray) -> float:
        """Distance of the object at `cube` (current orientation) to the carry region: horizontally,
        how far its farthest footprint corner is outside the ring (`carry_margin` inside the wall);
        vertically, how far it is below the carry height while not over the ring."""
        rcfg = self.cfg.reward
        quat = self.robot.data.qpos[self.robot._cube_qpos + 3 : self.robot._cube_qpos + 7]
        inner = self.robot.config.target_inner_radius - rcfg.carry_margin
        half = self.robot.config.object_half_extents
        horizontal = footprint_overshoot(cube, quat, half, self._ring_center, inner)
        vertical = max(0.0, self._carry_height - float(cube[2])) if horizontal > 0.0 else 0.0
        return float(np.hypot(horizontal, vertical))

    def _reward(self, stage: float, success: bool) -> float:
        rcfg = self.cfg.reward
        if rcfg.potential_shaping:
            # The success step ends the episode; its bonus replaces the shaping term
            if success:
                return rcfg.success_bonus
            return rcfg.potential_gamma * stage - self._prev_stage
        return 0.0 if success else stage / self._s_max - 1.0

    def _low_ee_penalty(self) -> float:
        """Penalty growing with the time the EE has stayed near the table (see `RewardConfig`)."""
        rcfg = self.cfg.reward
        cube = self.robot.data.qpos[self.robot._cube_qpos : self.robot._cube_qpos + 2]
        in_ring = float(np.linalg.norm(cube - self._ring_center)) < self._ring_inside
        low = self.robot.data.site_xpos[self._ee_site][2] < rcfg.low_ee_height
        self._low_ee_steps = self._low_ee_steps + 1 if low and not in_ring else 0
        excess = self._low_ee_steps - rcfg.low_ee_grace_steps
        return min(rcfg.low_ee_penalty_max, rcfg.low_ee_penalty * excess) if excess > 0 else 0.0

    def _reference_offset(self) -> float:
        """Distance between the IK reference (FK of the last IK solution) and the measured EE."""
        reference = self.kinematics.forward_kinematics(self._q_ik)[:3, 3] + self._base_offset
        return float(np.linalg.norm(reference - self.robot.data.site_xpos[self._ee_site]))
