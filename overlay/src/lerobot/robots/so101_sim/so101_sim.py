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

import contextlib
import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_so101_sim import SO101SimFollowerConfig

logger = logging.getLogger(__name__)

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERAS = ["front", "wrist"]
# Seconds of physics per Goal_Position write (matches the pacing of `reset_follower_position`)
GOAL_WRITE_DT = 0.015


def footprint_overshoot(
    pos: np.ndarray, quat: np.ndarray, half_extents: list[float], center: np.ndarray, radius: float
) -> float:
    """Distance beyond `radius` (around `center`, in the table plane) of the farthest corner of a box
    at `pos` / `quat` (w, x, y, z) with `half_extents`; 0 when all its corners project inside."""
    offset = float(np.hypot(pos[0] - center[0], pos[1] - center[1]))
    # No corner is farther from the box center than its half diagonal, whatever the orientation
    if offset + float(np.linalg.norm(half_extents)) <= radius:
        return 0.0
    w, x, y, z = quat
    rotation_xy = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        ]
    )
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = pos[:2] + (signs * half_extents) @ rotation_xy.T
    return max(0.0, float(np.max(np.linalg.norm(corners - center, axis=1))) - radius)


class _SimBus:
    """Minimal stand-in for the motors bus, for the code that talks to `robot.bus` directly."""

    def __init__(self, robot: "SO101SimFollower"):
        self._robot = robot
        self.motors = dict.fromkeys(MOTORS)

    def sync_read(self, data_name: str) -> dict[str, float]:
        if data_name == "Present_Position":
            return self._robot._read_positions()
        if data_name == "Present_Current":
            return self._robot._read_currents()
        raise NotImplementedError(f"{data_name} is not simulated")

    def sync_write(self, data_name: str, values: dict[str, float]) -> None:
        if data_name != "Goal_Position":
            raise NotImplementedError(f"{data_name} is not simulated")
        self._robot._set_targets(values)
        self._robot._step(GOAL_WRITE_DT)


class SO101SimFollower(Robot):
    """SO101 follower arm simulated in MuJoCo, with the units and interface of the real robot."""

    config_class = SO101SimFollowerConfig
    name = "so101_sim_follower"

    def __init__(self, config: SO101SimFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = _SimBus(self)
        # Only the keys are used by the environment (real robots store camera objects here)
        self.cameras = dict.fromkeys(CAMERAS)
        self._rng = np.random.default_rng(config.seed)
        self.model = None
        self.data = None
        self._renderer = None
        self._viewer = None
        self._placed_time = 0.0

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        cams = {cam: (self.config.image_height, self.config.image_width, 3) for cam in CAMERAS}
        return {**{f"{m}.pos": float for m in MOTORS}, **cams}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in MOTORS}

    @property
    def is_connected(self) -> bool:
        return self.model is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import mujoco

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self._joint_qpos = np.array([self.model.joint(m).qposadr[0] for m in MOTORS])
        self._joint_dof = np.array([self.model.joint(m).dofadr[0] for m in MOTORS])
        self._actuators = np.array([self.model.actuator(m).id for m in MOTORS])
        self._gripper_range = self.model.jnt_range[self.model.joint("gripper").id].copy()
        self._cube_qpos = self.model.joint("cube").qposadr[0]
        self._cube_dof = self.model.joint("cube").dofadr[0]
        self._renderer = mujoco.Renderer(
            self.model, height=self.config.image_height, width=self.config.image_width
        )
        if self.config.show_viewer:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self._viewer.lock():
                cam = self._viewer.cam
                cam.azimuth = self.config.viewer_azimuth_deg
                cam.elevation = self.config.viewer_elevation_deg
                cam.distance = self.config.viewer_distance
                cam.lookat[:] = self.config.viewer_lookat
        self.reset_scene([0.0, 0.0, 0.0, 90.0, 0.0, 5.0])
        logger.info(f"{self} connected.")

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.model = None
        self.data = None
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------ scene

    def _build_model(self):
        return self._build_spec().compile()

    def _build_spec(self):
        """The scene as an `MjSpec`, so subclasses can extend it before compiling."""
        import mujoco

        cfg = self.config
        spec = mujoco.MjSpec.from_file(cfg.xml_path)
        # Grasp-friendly solver settings, following SO101-Nexus (Apache-2.0,
        # https://github.com/johnsutor/so101-nexus): elliptic cones + noslip keep a held cube from sliding
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.iterations = 10
        spec.option.ls_iterations = 20
        spec.option.noslip_iterations = 3
        spec.visual.global_.offwidth = max(cfg.image_width, 640)
        spec.visual.global_.offheight = max(cfg.image_height, 480)
        spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]

        # Look like the real arm: white printed parts, black servos
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                geom.material = ""
                is_servo = geom.meshname.startswith("sts3215")
                geom.rgba = [0.12, 0.12, 0.12, 1] if is_servo else [0.93, 0.93, 0.9, 1]

        world = spec.worldbody
        spec.body("base").pos = [0.0, 0.0, cfg.robot_base_z]
        world.add_light(
            pos=[0.3, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.45, 0.45, 0.45]
        )
        world.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], rgba=[0.82, 0.82, 0.8, 1]
        )

        # Object to pick (the "cube"), with the density of the default 2 cm, 20 g cube
        half = cfg.object_half_extents
        cube = world.add_body(name="cube", pos=[0.18, 0.16, half[2]])
        cube.add_freejoint(name="cube")
        cube.add_geom(
            name="cube",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            mass=0.02 * float(np.prod(half)) / 0.01**3,
            rgba=[0.05, 0.05, 0.05, 1],
            friction=[1.0, 0.01, 0.001],
            condim=4,
        )

        # Ring (roll of tape): wall segments around the target center, on a mocap body so it can be
        # moved per episode through `data.mocap_pos` (it stays static otherwise)
        ring = world.add_body(name="ring", pos=[cfg.target_center[0], cfg.target_center[1], 0], mocap=True)
        n_seg = 20
        r_mid = cfg.target_inner_radius + cfg.target_wall_thickness / 2
        seg_half_len = np.pi * r_mid / n_seg * 1.05
        for i in range(n_seg):
            angle = 2 * np.pi * i / n_seg
            ring.add_geom(
                name=f"ring_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[cfg.target_wall_thickness / 2, seg_half_len, cfg.target_height / 2],
                pos=[r_mid * np.cos(angle), r_mid * np.sin(angle), cfg.target_height / 2],
                euler=[0, 0, angle],
                rgba=[0.85, 0.2, 0.15, 1] if i % 2 else [0.95, 0.75, 0.1, 1],
            )

        # Front camera looking at the workspace, like the real one
        pos = np.array(cfg.front_camera_pos)
        forward = np.array(cfg.front_camera_lookat) - pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        roll = np.deg2rad(cfg.front_camera_roll_deg)
        right, up = np.cos(roll) * right + np.sin(roll) * up, np.cos(roll) * up - np.sin(roll) * right
        world.add_camera(name="front", pos=pos.tolist(), xyaxes=[*right, *up], fovy=cfg.front_camera_fovy)

        # Grasping only goes through two flat finger pads: the jaw meshes' convex hulls overlap
        gripper = spec.body("gripper")
        jaw = spec.body("moving_jaw_so101_v1")
        for body in (gripper, jaw):
            for geom in body.geoms:
                geom.contype = 0
                geom.conaffinity = 0

        # Pad poses are defined in the tool frame (`gripperframe` site): fingers along +x, closing along z
        base = spec.compile()
        data = mujoco.MjData(base)
        data.qpos[base.joint("gripper").qposadr[0]] = base.jnt_range[base.joint("gripper").id][0]
        mujoco.mj_kinematics(base, data)
        site = base.site("gripperframe").id
        tool_rot = data.site_xmat[site].reshape(3, 3)
        tool_pos = data.site_xpos[site]
        # Pads span the finger up to its tip (~5 mm past the tool point), so a grasp at the tool point
        # holds the cube around its center of mass instead of above it (where it swings out)
        pad_half = [0.0135, 0.007, 0.002]

        def pad_pose_in(body_name: str, center_in_tool: list[float]) -> tuple[list[float], list[float]]:
            body_id = base.body(body_name).id
            body_rot = data.xmat[body_id].reshape(3, 3)
            world_pos = tool_pos + tool_rot @ np.array(center_in_tool)
            local_pos = body_rot.T @ (world_pos - data.xpos[body_id])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (body_rot.T @ tool_rot).flatten())
            return local_pos.tolist(), quat.tolist()

        # Contact parameters of the SO101-Nexus finger pads
        pad_kwargs = {
            "type": mujoco.mjtGeom.mjGEOM_BOX,
            "size": pad_half,
            "friction": [1.0, 0.05, 0.001],
            # condim 6 adds torsional/rolling resistance, so the held cube does not pivot on the pads
            "condim": 6,
            "solref": [0.01, 1],
            "solimp": [0.95, 0.99, 0.001, 0.5, 2],
            "rgba": [0.1, 0.1, 0.1, 0.4],
        }
        pos_fixed, quat_fixed = pad_pose_in("gripper", [-0.009, 0.0, -0.002])
        gripper.add_geom(name="fixed_pad", pos=pos_fixed, quat=quat_fixed, **pad_kwargs)
        pos_moving, quat_moving = pad_pose_in("moving_jaw_so101_v1", [-0.009, 0.0, 0.004])
        jaw.add_geom(name="moving_pad", pos=pos_moving, quat=quat_moving, **pad_kwargs)

        # Wrist camera at the URDF `wrist_camera_link` position, looking past the fingertips
        cam_pos = np.array([0.0025, 0.07344, 0.00659])
        lookat = np.array([-0.0079, 0.0, -0.13])
        forward = lookat - cam_pos
        forward /= np.linalg.norm(forward)
        right = np.array([1.0, 0.0, 0.0]) - forward[0] * forward
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        gripper.add_camera(name="wrist", pos=cam_pos.tolist(), xyaxes=[*right, *up], fovy=cfg.wrist_camera_fovy)

        return spec

    def reset_scene(self, joint_positions: list[float] | None = None) -> None:
        """Teleport the arm to `joint_positions` (robot units) and drop the cube at a random spot."""
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        if joint_positions is not None:
            targets = self._to_sim(dict(zip(MOTORS, joint_positions, strict=True)))
            self.data.qpos[self._joint_qpos] = targets
            self.data.ctrl[self._actuators] = targets

        xy = self._rng.uniform(self.config.cube_spawn_min, self.config.cube_spawn_max)
        max_yaw = np.deg2rad(self.config.cube_yaw_range_deg)
        yaw = self._rng.uniform(-max_yaw, max_yaw)
        self.data.qpos[self._cube_qpos : self._cube_qpos + 7] = [
            xy[0],
            xy[1],
            self.config.object_half_extents[2],
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]
        mujoco.mj_forward(self.model, self.data)
        self._step(0.2)
        self._placed_time = 0.0
        # Where the gripper must come back to when `return_home` is set
        self._home_ee = self.data.site_xpos[self.model.site("gripperframe").id].copy()

    def object_height(self) -> float:
        """Height of the cube center above the table (m); for evaluation, not visible to the policy."""
        return float(self.data.qpos[self._cube_qpos + 2])

    def ring_center(self) -> np.ndarray:
        """Current (x, y) of the ring; `config.target_center` unless the ring was moved."""
        return self.data.mocap_pos[self.model.body("ring").mocapid[0], :2]

    def footprint_overshoot(self, margin: float = 0.0) -> float:
        """How far (m) the object's farthest footprint corner is beyond `inner radius - margin` of the
        ring (0 when the whole outline is inside). Uses the object's full pose, so its yaw counts."""
        qpos = self.data.qpos[self._cube_qpos : self._cube_qpos + 7]
        return footprint_overshoot(
            qpos[:3], qpos[3:], self.config.object_half_extents, self.ring_center(),
            self.config.target_inner_radius - margin,
        )

    def _is_placed(self) -> bool:
        """The object is still, its whole outline (projected on the table) inside the ring; it may lie
        tilted, e.g. against the ring wall."""
        still = np.linalg.norm(self.data.qvel[self._cube_dof : self._cube_dof + 3]) < 0.02
        return bool(still and self.footprint_overshoot() == 0.0)

    def is_home(self) -> bool:
        """The gripper is back where it was after the last reset."""
        ee = self.data.site_xpos[self.model.site("gripperframe").id]
        return bool(np.linalg.norm(ee - self._home_ee) < self.config.home_tolerance)

    def is_success(self) -> bool:
        """The cube has stayed placed in the ring for `placement_dwell_s` (as in SO101-Nexus), and with
        `return_home` the gripper is back at its reset pose."""
        placed = self._placed_time >= self.config.placement_dwell_s
        return placed and (not self.config.return_home or self.is_home())

    # --------------------------------------------------------------- physics

    def _to_sim(self, positions: dict[str, float]) -> np.ndarray:
        """Robot units (degrees, gripper 0-100) -> joint angles in radians."""
        lo, hi = self._gripper_range
        values = []
        for motor in MOTORS:
            value = positions[motor]
            values.append(lo + value / 100.0 * (hi - lo) if motor == "gripper" else np.deg2rad(value))
        return np.array(values)

    def _read_positions(self) -> dict[str, float]:
        q = self.data.qpos[self._joint_qpos]
        lo, hi = self._gripper_range
        values = np.rad2deg(q)
        values[-1] = (q[-1] - lo) / (hi - lo) * 100.0
        return {motor: float(v) for motor, v in zip(MOTORS, values, strict=True)}

    def _read_currents(self) -> dict[str, float]:
        torques = np.abs(self.data.actuator_force[self._actuators]) * self.config.current_scale
        return {motor: float(v) for motor, v in zip(MOTORS, torques, strict=True)}

    def _set_targets(self, positions: dict[str, float]) -> None:
        self.data.ctrl[self._actuators] = self._to_sim(positions)

    def _step(self, duration: float) -> None:
        import mujoco

        n_steps = max(1, round(duration / self.model.opt.timestep))
        lock = self._viewer.lock() if self._viewer is not None else contextlib.nullcontext()
        with lock:
            for _ in range(n_steps):
                mujoco.mj_step(self.model, self.data)
                self._placed_time = self._placed_time + self.model.opt.timestep if self._is_placed() else 0.0
        if self._viewer is not None:
            self._viewer.sync()

    def _render(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    # ------------------------------------------------------------- interface

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict[str, Any] = {f"{m}.pos": v for m, v in self._read_positions().items()}
        for cam in CAMERAS:
            obs[cam] = self._render(cam)
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal = {m: float(action[f"{m}.pos"]) for m in MOTORS}
        self._set_targets(goal)
        self._step(1.0 / self.config.fps)
        return {f"{m}.pos": v for m, v in goal.items()}
