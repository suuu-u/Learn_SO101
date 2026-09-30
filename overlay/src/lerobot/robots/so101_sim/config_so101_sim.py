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

from dataclasses import dataclass, field

from ..config import RobotConfig


@RobotConfig.register_subclass("so101_sim_follower")
@dataclass
class SO101SimFollowerConfig(RobotConfig):
    """MuJoCo SO101 follower exposing the same interface and units as the real `so101_follower`.

    Joints are in degrees (new calibration: zero at the middle of the range) and the gripper in
    [0, 100] (0 = closed), so the real-robot pipeline (IK, bounds, leader teleop, cropping) runs
    unchanged. The scene is a pick-and-place task: move the cube into the ring.
    """

    # SO101 MJCF, relative to the working directory or absolute
    xml_path: str = "Simulation/SO101/so101_new_calib.xml"
    # Physics advanced per `send_action`, must match the env fps
    fps: int = 10
    # Open an interactive MuJoCo window to watch the scene (needed to teleoperate)
    show_viewer: bool = True
    # Initial view of that window. Keep `viewer_azimuth_deg` equal to the teleop `view_azimuth_deg`
    # so that mouse/arrow directions match the screen (0 = from behind the robot)
    viewer_azimuth_deg: float = 0.0
    viewer_elevation_deg: float = -40.0
    viewer_distance: float = 0.7
    viewer_lookat: list[float] = field(default_factory=lambda: [0.2, 0.05, 0.0])
    seed: int | None = None

    # Rendered camera images (same size as the real cameras, so the same crop parameters apply)
    image_width: int = 640
    image_height: int = 480
    # Front camera fitted to the real one (arm silhouettes in 7 poses); the look-at point only sets the
    # viewing direction. Roll: rotation about the viewing axis
    front_camera_pos: list[float] = field(default_factory=lambda: [0.4603, -0.0096, 0.3992])
    front_camera_lookat: list[float] = field(default_factory=lambda: [0.0562, 0.0811, -0.2228])
    front_camera_fovy: float = 41.2
    front_camera_roll_deg: float = 7.1
    wrist_camera_fovy: float = 70.0

    # Cube (side length in meters) and its random spawn area in the robot base frame
    cube_size: float = 0.02
    cube_spawn_min: list[float] = field(default_factory=lambda: [0.12, 0.10])
    cube_spawn_max: list[float] = field(default_factory=lambda: [0.24, 0.22])
    # Random cube rotation about z at reset, in ±degrees. The action space has no wrist rotation, so
    # large angles force corner grasps that tend to slip
    cube_yaw_range_deg: float = 15.0
    # Object to pick instead of the cube: its full (x, y, z) size in meters at zero yaw, x being the
    # gripper's closing direction (e.g. [0.01, 0.02, 0.005]). None keeps a cube of `cube_size`
    object_size: list[float] | None = None
    # Ring (roll of tape) the cube must end up in
    target_center: list[float] = field(default_factory=lambda: [0.24, -0.09])
    # Ring sized like the real roll of tape (inner / outer radius 3.6 / 5.0 cm fitted in the calibrated
    # front camera, 2.5 cm tall)
    target_inner_radius: float = 0.036
    target_wall_thickness: float = 0.014
    target_height: float = 0.025
    # Seconds the cube must stay still inside the ring before success is reported
    placement_dwell_s: float = 0.3
    # Success also requires the gripper back at its reset pose (within `home_tolerance` m) afterwards
    return_home: bool = False
    # Height of the robot base frame above the table surface (m). Negative when the arm reaches lower
    # than the table plane at z = 0 would allow, e.g. to match a real setup where the fingertips touch
    # the table at joint angles that leave a gap in the default scene
    robot_base_z: float = 0.0
    home_tolerance: float = 0.02

    @property
    def object_half_extents(self) -> list[float]:
        """Half sizes (x, y, z) of the object to pick."""
        size = self.object_size if self.object_size is not None else [self.cube_size] * 3
        return [s / 2 for s in size]

    # Present_Current emulation: |actuator torque| (N·m) times this factor, roughly the STS3215 mA/N·m
    current_scale: float = 900.0
