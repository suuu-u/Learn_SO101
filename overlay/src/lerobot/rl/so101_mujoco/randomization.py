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

"""Domain randomization of the MuJoCo SO101 scene: appearance, images and physics.

The envs of a worker process share one `MjModel`, and colors, lights, camera poses and physics
parameters live in the model. So each env samples its own values per episode and writes them into
the shared model right before using it (rendering / stepping the physics); envs run one after the
other, so they never see each other's values. Mocap positions (ring, distractors) live in each env's
`MjData` and need no such care.
"""

import colorsys

import numpy as np

from .configs import ObservationNoiseConfig, PhysicsRandomizationConfig, VisualRandomizationConfig

N_TABLE_MATERIALS = 6
N_DISTRACTORS = 4
# Where unused distractors wait, out of every camera's view
DISTRACTOR_PARK = [5.0, 5.0, -1.0]
# Table area the distractors are dropped on (robot base frame), kept clear of the base, the ring and
# the object's spawn area
DISTRACTOR_AREA_MIN = [0.04, -0.26]
DISTRACTOR_AREA_MAX = [0.40, 0.32]

# Pristine appearance / physics of each shared model, captured before any env modifies it
_DEFAULTS: dict[int, dict[str, np.ndarray]] = {}
_PHYSICS_DEFAULTS: dict[int, dict[str, np.ndarray]] = {}
# Models currently holding randomized values
_MODIFIED: set[int] = set()
_MODIFIED_PHYSICS: set[int] = set()


def add_visual_variety(spec, seed: int = 0) -> None:
    """Add procedural table materials and visual-only distractor objects (mocap, no collisions) to a
    scene spec, for `VisualRandomizer` to use."""
    import mujoco

    rng = np.random.default_rng(seed)
    builtin = mujoco.mjtBuiltin
    builtins = [builtin.mjBUILTIN_CHECKER, builtin.mjBUILTIN_GRADIENT, builtin.mjBUILTIN_FLAT]
    for i in range(N_TABLE_MATERIALS):
        texture = spec.add_texture(
            name=f"dr_table_tex_{i}",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=builtins[i % len(builtins)],
            width=256,
            height=256,
            rgb1=rng.uniform(0.3, 1.0, 3),
            rgb2=rng.uniform(0.0, 0.7, 3),
            mark=mujoco.mjtMark.mjMARK_RANDOM,
            random=float(rng.uniform(0.01, 0.15)),
            markrgb=rng.uniform(0.0, 1.0, 3),
        )
        repeat = float(rng.uniform(2, 12))
        material = spec.add_material(name=f"dr_table_mat_{i}", texrepeat=[repeat, repeat], texuniform=True)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name

    shapes = [
        (mujoco.mjtGeom.mjGEOM_BOX, [0.012, 0.02, 0.015]),
        (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.015, 0.02, 0.0]),
        (mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0.0, 0.0]),
        (mujoco.mjtGeom.mjGEOM_BOX, [0.03, 0.01, 0.008]),
    ]
    for i in range(N_DISTRACTORS):
        geom_type, size = shapes[i % len(shapes)]
        body = spec.worldbody.add_body(name=f"distractor_{i}", pos=DISTRACTOR_PARK, mocap=True)
        body.add_geom(
            name=f"distractor_{i}",
            type=geom_type,
            size=size,
            contype=0,
            conaffinity=0,
            rgba=[0.5, 0.5, 0.5, 1],
        )


# Point the side lamp aims at: the middle of the workspace (robot base frame)
LAMP_TARGET = [0.2, 0.0, 0.0]


def add_lighting_variety(spec) -> None:
    """Add a shadow-casting spot light (off until `VisualRandomizer` turns it on) and a glossy table
    material to a scene spec. Unused, they leave the rendered images unchanged."""
    import mujoco

    spec.worldbody.add_light(
        name="dr_lamp",
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        pos=[LAMP_TARGET[0], LAMP_TARGET[1], 1.0],
        dir=[0, 0, -1],
        castshadow=True,
        active=False,
        cutoff=60.0,
        exponent=2.0,
        diffuse=[0.0, 0.0, 0.0],
        specular=[0.3, 0.3, 0.3],
    )
    # The table keeps its geom color (a non-default geom rgba overrides the material's)
    spec.add_material(name="dr_gloss_table")


def _defaults(model) -> dict[str, np.ndarray]:
    key = id(model)
    if key not in _DEFAULTS:
        _DEFAULTS[key] = {
            "geom_rgba": model.geom_rgba.copy(),
            "geom_matid": model.geom_matid.copy(),
            "light_diffuse": model.light_diffuse.copy(),
            "light_dir": model.light_dir.copy(),
            "light_pos": model.light_pos.copy(),
            "light_active": model.light_active.copy(),
            "light_specular": model.light_specular.copy(),
            "mat_reflectance": model.mat_reflectance.copy(),
            "mat_specular": model.mat_specular.copy(),
            "mat_shininess": model.mat_shininess.copy(),
            "headlight_diffuse": np.array(model.vis.headlight.diffuse),
            "headlight_ambient": np.array(model.vis.headlight.ambient),
            "cam_pos": model.cam_pos.copy(),
            "cam_quat": model.cam_quat.copy(),
        }
    return _DEFAULTS[key]


def restore_appearance(model) -> None:
    """Undo a randomized appearance; for envs without randomization sharing the model."""
    if id(model) not in _MODIFIED:
        return
    d = _DEFAULTS[id(model)]
    model.geom_rgba[:] = d["geom_rgba"]
    model.geom_matid[:] = d["geom_matid"]
    _restore_lighting(model, d)
    model.vis.headlight.diffuse[:] = d["headlight_diffuse"]
    model.vis.headlight.ambient[:] = d["headlight_ambient"]
    model.cam_pos[:] = d["cam_pos"]
    model.cam_quat[:] = d["cam_quat"]
    _MODIFIED.discard(id(model))


def _restore_lighting(model, d: dict[str, np.ndarray]) -> None:
    for name in (
        "light_diffuse",
        "light_dir",
        "light_pos",
        "light_active",
        "light_specular",
        "mat_reflectance",
        "mat_specular",
        "mat_shininess",
    ):
        getattr(model, name)[:] = d[name]


def _named_ids(model, kind, prefix: str, count: int) -> list[int]:
    import mujoco

    ids = [mujoco.mj_name2id(model, kind, f"{prefix}{i}") for i in range(count)]
    return [i for i in ids if i >= 0]


class VisualRandomizer:
    def __init__(self, model, cfg: VisualRandomizationConfig, rng: np.random.Generator):
        import mujoco

        self.model = model
        self.cfg = cfg
        self.rng = rng
        self.defaults = _defaults(model)

        ring_body = model.body("ring").id
        ring = [g for g in range(model.ngeom) if model.geom_bodyid[g] == ring_body]
        self.ring_a, self.ring_b = ring[0::2], ring[1::2]
        self.cube = model.geom("cube").id
        self.table = model.geom("table").id
        # White printed parts of the arm (the servos are dark)
        rgba = self.defaults["geom_rgba"]
        self.robot_white = [
            g
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and rgba[g, 0] > 0.5
        ]
        self.cameras = {"front": model.camera("front").id, "wrist": model.camera("wrist").id}
        # Present only in scenes extended with `add_visual_variety`
        obj = mujoco.mjtObj
        self.table_materials = _named_ids(model, obj.mjOBJ_MATERIAL, "dr_table_mat_", N_TABLE_MATERIALS)
        self.distractor_geoms = _named_ids(model, obj.mjOBJ_GEOM, "distractor_", N_DISTRACTORS)
        self.distractor_mocap = [
            model.body(f"distractor_{i}").mocapid[0] for i in range(len(self.distractor_geoms))
        ]
        # Present only in scenes extended with `add_lighting_variety`
        self.lamp = mujoco.mj_name2id(model, obj.mjOBJ_LIGHT, "dr_lamp")
        self.gloss = mujoco.mj_name2id(model, obj.mjOBJ_MATERIAL, "dr_gloss_table")
        self.params: dict | None = None

    def _color(self, saturation: list[float], value: list[float]) -> np.ndarray:
        h = self.rng.uniform(0.0, 1.0)
        return np.array(colorsys.hsv_to_rgb(h, self.rng.uniform(*saturation), self.rng.uniform(*value)))

    def _uniform(self, bounds: list[float]) -> float:
        # Fixed ranges draw nothing, so older configs keep their random streams
        low, high = bounds
        return float(self.rng.uniform(low, high)) if low != high else float(low)

    def _jitter_quat(self, quat: np.ndarray) -> np.ndarray:
        import mujoco

        axis = self.rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(self.rng.uniform(-self.cfg.camera_rotation_deg, self.cfg.camera_rotation_deg))
        delta, out = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(delta, axis, angle)
        mujoco.mju_mulQuat(out, delta, quat)
        return out

    def sample(self) -> None:
        """Draw the appearance of the next episode."""
        cfg, d = self.cfg, self.defaults
        ring_a = self._color(cfg.ring_saturation, cfg.ring_value)
        single = self.rng.random() < cfg.ring_single_color_prob
        ring_b = ring_a if single else self._color(cfg.ring_saturation, cfg.ring_value)
        light_dir = d["light_dir"] + self.rng.normal(0.0, cfg.light_direction_std, d["light_dir"].shape)
        light_dir /= np.linalg.norm(light_dir, axis=-1, keepdims=True)
        cam_pos, cam_quat = d["cam_pos"].copy(), d["cam_quat"].copy()
        for name, jitter in (("front", cfg.front_camera_position), ("wrist", cfg.wrist_camera_position)):
            cam = self.cameras[name]
            cam_pos[cam] += self.rng.uniform(-jitter, jitter, 3)
            cam_quat[cam] = self._jitter_quat(cam_quat[cam])
        table_material = -1
        if self.table_materials and self.rng.random() < cfg.table_texture_prob:
            table_material = int(self.rng.choice(self.table_materials))
        self.params = {
            "ring_a": ring_a,
            "ring_b": ring_b,
            "cube": self._color(cfg.cube_saturation, cfg.cube_value),
            "table": self._color(cfg.table_saturation, cfg.table_value),
            "table_material": table_material,
            "robot": self.rng.uniform(*cfg.robot_brightness),
            "light": self.rng.uniform(*cfg.light_scale),
            "light_dir": light_dir,
            "cam_pos": cam_pos,
            "cam_quat": cam_quat,
            "distractors": [self._color([0.0, 1.0], [0.1, 1.0]) for _ in self.distractor_geoms],
            "light_specular": self._uniform(cfg.light_specular),
            "lamp": self._sample_lamp() if self.lamp >= 0 and self.rng.random() < cfg.lamp_prob else None,
            "table_gloss": tuple(
                self._uniform(bounds)
                for bounds in (cfg.table_reflectance, cfg.table_specular, cfg.table_shininess)
            ),
        }

    def _sample_lamp(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Position, direction (aimed at the workspace) and intensity of the side lamp."""
        cfg = self.cfg
        azimuth = self.rng.uniform(0.0, 2 * np.pi)
        elevation = np.deg2rad(self.rng.uniform(*cfg.lamp_elevation_deg))
        distance = self.rng.uniform(*cfg.lamp_distance)
        offset = distance * np.array(
            [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
        )
        intensity = float(self.rng.uniform(*cfg.lamp_intensity))
        return np.array(LAMP_TARGET) + offset, -offset / distance, intensity

    def place_distractors(self, data, keep_clear: list[tuple[np.ndarray, float]]) -> None:
        """Drop a random number of distractors on the table (in this env's `MjData`), away from the
        (center, radius) areas in `keep_clear`; the others wait out of view."""
        count = int(self.rng.integers(0, min(self.cfg.max_distractors, len(self.distractor_mocap)) + 1))
        for i, mocap in enumerate(self.distractor_mocap):
            data.mocap_pos[mocap] = DISTRACTOR_PARK
            if i >= count:
                continue
            for _ in range(20):
                xy = self.rng.uniform(DISTRACTOR_AREA_MIN, DISTRACTOR_AREA_MAX)
                if all(np.linalg.norm(xy - center) > radius for center, radius in keep_clear):
                    data.mocap_pos[mocap] = [xy[0], xy[1], self._half_height(self.distractor_geoms[i])]
                    break

    def _half_height(self, geom: int) -> float:
        """Height of the center of a geom resting on the table."""
        import mujoco

        size, geom_type = self.model.geom_size[geom], self.model.geom_type[geom]
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(size[1])
        return float(size[2])

    def apply(self) -> None:
        """Write this env's appearance into the (shared) model; call right before rendering."""
        m, d, p = self.model, self.defaults, self.params
        if p is None:
            return
        m.geom_rgba[:] = d["geom_rgba"]
        m.geom_matid[:] = d["geom_matid"]
        m.geom_rgba[self.ring_a, :3] = p["ring_a"]
        m.geom_rgba[self.ring_b, :3] = p["ring_b"]
        m.geom_rgba[self.cube, :3] = p["cube"]
        if p["table_material"] >= 0:
            m.geom_matid[self.table] = p["table_material"]
            m.geom_rgba[self.table] = [1.0, 1.0, 1.0, 1.0]
        else:
            m.geom_rgba[self.table, :3] = p["table"]
        m.geom_rgba[self.robot_white, :3] = np.clip(d["geom_rgba"][self.robot_white, :3] * p["robot"], 0, 1)
        for geom, color in zip(self.distractor_geoms, p["distractors"], strict=True):
            m.geom_rgba[geom, :3] = color
        _restore_lighting(m, d)
        m.light_diffuse[:] = np.clip(d["light_diffuse"] * p["light"], 0, 1)
        m.light_dir[:] = p["light_dir"]
        m.light_specular[:] = p["light_specular"]
        if p["lamp"] is not None:
            pos, direction, intensity = p["lamp"]
            m.light_active[self.lamp] = 1
            m.light_pos[self.lamp] = pos
            m.light_dir[self.lamp] = direction
            m.light_diffuse[self.lamp] = intensity
        if self.gloss >= 0 and p["table_material"] < 0:
            m.geom_matid[self.table] = self.gloss
            reflectance, specular, shininess = p["table_gloss"]
            m.mat_reflectance[self.gloss] = reflectance
            m.mat_specular[self.gloss] = specular
            m.mat_shininess[self.gloss] = shininess
        m.vis.headlight.diffuse[:] = np.clip(d["headlight_diffuse"] * p["light"], 0, 1)
        m.vis.headlight.ambient[:] = np.clip(d["headlight_ambient"] * p["light"], 0, 1)
        m.cam_pos[:] = p["cam_pos"]
        m.cam_quat[:] = p["cam_quat"]
        _MODIFIED.add(id(m))


class ImageCorruptor:
    """Per-episode photometric changes (per camera) and blur / noise / JPEG compression (shared) of the
    rendered (C, H, W) uint8 images."""

    def __init__(self, cfg: VisualRandomizationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.params: dict = {}
        self.photometric: dict[str, dict[str, float]] = {}

    def sample(self) -> None:
        cfg = self.cfg
        self.params = {
            "blur": float(self.rng.uniform(*cfg.blur_sigma)) if self.rng.random() < cfg.blur_prob else 0.0,
            "noise": float(self.rng.uniform(*cfg.noise_std)),
            "jpeg": (
                int(self.rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
                if self.rng.random() < cfg.jpeg_prob
                else 0
            ),
        }
        # Sampled lazily per camera (see `_photometric_params`)
        self.photometric = {}

    def _photometric_params(self, camera: str) -> dict[str, float]:
        if camera not in self.photometric:
            # Fixed ranges draw nothing, so older configs keep their random streams
            self.photometric[camera] = {}
            for name in ("exposure", "contrast", "saturation", "gamma"):
                low, high = getattr(self.cfg, name)
                self.photometric[camera][name] = float(self.rng.uniform(low, high)) if low != high else low
        return self.photometric[camera]

    def _apply_photometric(self, hwc: np.ndarray, camera: str) -> np.ndarray:
        p = self._photometric_params(camera)
        if all(value == 1.0 for value in p.values()):
            return hwc
        x = hwc.astype(np.float32) / 255.0 * p["exposure"]
        x = (x - x.mean()) * p["contrast"] + x.mean()
        gray = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        x = (x - gray[..., None]) * p["saturation"] + gray[..., None]
        x = np.clip(x, 0.0, 1.0) ** p["gamma"]
        return (x * 255.0 + 0.5).astype(np.uint8)

    def __call__(self, image: np.ndarray, camera: str = "") -> np.ndarray:
        import cv2

        p = self.params
        if not p:
            return image
        hwc = self._apply_photometric(np.ascontiguousarray(image.transpose(1, 2, 0)), camera)
        if p["blur"] > 0.0:
            hwc = cv2.GaussianBlur(hwc, (0, 0), p["blur"])
        if p["noise"] > 0.0:
            noisy = hwc.astype(np.float32) + self.rng.normal(0.0, p["noise"], hwc.shape)
            hwc = np.clip(noisy, 0, 255).astype(np.uint8)
        if p["jpeg"] > 0:
            ok, encoded = cv2.imencode(".jpg", hwc, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
            if ok:
                hwc = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        return np.ascontiguousarray(hwc.transpose(2, 0, 1))


def _physics_defaults(model) -> dict[str, np.ndarray]:
    key = id(model)
    if key not in _PHYSICS_DEFAULTS:
        _PHYSICS_DEFAULTS[key] = {
            "gainprm": model.actuator_gainprm.copy(),
            "biasprm": model.actuator_biasprm.copy(),
            "frictionloss": model.dof_frictionloss.copy(),
            "damping": model.dof_damping.copy(),
            "armature": model.dof_armature.copy(),
            "body_mass": model.body_mass.copy(),
            "body_inertia": model.body_inertia.copy(),
            "geom_friction": model.geom_friction.copy(),
        }
    return _PHYSICS_DEFAULTS[key]


def restore_physics(model) -> None:
    """Undo randomized physics; for envs without randomization sharing the model."""
    if id(model) not in _MODIFIED_PHYSICS:
        return
    d = _PHYSICS_DEFAULTS[id(model)]
    model.actuator_gainprm[:] = d["gainprm"]
    model.actuator_biasprm[:] = d["biasprm"]
    model.dof_frictionloss[:] = d["frictionloss"]
    model.dof_damping[:] = d["damping"]
    model.dof_armature[:] = d["armature"]
    model.body_mass[:] = d["body_mass"]
    model.body_inertia[:] = d["body_inertia"]
    model.geom_friction[:] = d["geom_friction"]
    _MODIFIED_PHYSICS.discard(id(model))


class PhysicsRandomizer:
    """Per-episode servo gains, joint friction / damping / armature, masses and frictions."""

    def __init__(self, model, cfg: PhysicsRandomizationConfig, rng: np.random.Generator):
        self.model = model
        self.cfg = cfg
        self.rng = rng
        self.defaults = _physics_defaults(model)
        base = model.body("base").id
        cube_body = model.body("cube").id
        self.robot_bodies = [b for b in range(model.nbody) if model.body_rootid[b] == base]
        self.robot_dofs = [d for d in range(model.nv) if model.dof_bodyid[d] in self.robot_bodies]
        self.cube_body = cube_body
        self.cube_geom = model.geom("cube").id
        self.pad_geoms = [model.geom("fixed_pad").id, model.geom("moving_pad").id]
        self.params: dict | None = None

    def _scale(self, bounds: list[float], size: int | None = None):
        return self.rng.uniform(bounds[0], bounds[1], size)

    def sample(self) -> None:
        cfg, nu, ndof = self.cfg, self.model.nu, len(self.robot_dofs)
        self.params = {
            "kp": self._scale(cfg.actuator_kp_scale, nu),
            "kv": self._scale(cfg.actuator_kv_scale, nu),
            "friction": self._scale(cfg.joint_friction_scale, ndof),
            "damping": self._scale(cfg.joint_damping_scale, ndof),
            "armature": self._scale(cfg.joint_armature_scale, ndof),
            "link_mass": self._scale(cfg.link_mass_scale, len(self.robot_bodies)),
            "object_mass": float(self._scale(cfg.object_mass_scale)),
            "object_friction": float(self._scale(cfg.object_friction_scale)),
            "pad_friction": float(self._scale(cfg.pad_friction_scale)),
        }

    def apply(self) -> None:
        """Write this env's physics into the (shared) model; call right before stepping it."""
        m, d, p = self.model, self.defaults, self.params
        if p is None:
            return
        # Position servos: force = kp * (target - q) - kv * qdot
        m.actuator_gainprm[:, 0] = d["gainprm"][:, 0] * p["kp"]
        m.actuator_biasprm[:, 1] = d["biasprm"][:, 1] * p["kp"]
        m.actuator_biasprm[:, 2] = d["biasprm"][:, 2] * p["kv"]
        dofs = self.robot_dofs
        m.dof_frictionloss[:] = d["frictionloss"]
        m.dof_damping[:] = d["damping"]
        m.dof_armature[:] = d["armature"]
        m.dof_frictionloss[dofs] = d["frictionloss"][dofs] * p["friction"]
        m.dof_damping[dofs] = d["damping"][dofs] * p["damping"]
        m.dof_armature[dofs] = d["armature"][dofs] * p["armature"]
        m.body_mass[:] = d["body_mass"]
        m.body_inertia[:] = d["body_inertia"]
        m.body_mass[self.robot_bodies] = d["body_mass"][self.robot_bodies] * p["link_mass"]
        m.body_inertia[self.robot_bodies] = d["body_inertia"][self.robot_bodies] * p["link_mass"][:, None]
        m.body_mass[self.cube_body] = d["body_mass"][self.cube_body] * p["object_mass"]
        m.body_inertia[self.cube_body] = d["body_inertia"][self.cube_body] * p["object_mass"]
        m.geom_friction[:] = d["geom_friction"]
        m.geom_friction[self.cube_geom, 0] = d["geom_friction"][self.cube_geom, 0] * p["object_friction"]
        m.geom_friction[self.pad_geoms, 0] = d["geom_friction"][self.pad_geoms, 0] * p["pad_friction"]
        _MODIFIED_PHYSICS.add(id(m))


class JointReadingNoise:
    """Encoder bias (per episode) and noise (per reading) on joint positions in robot units."""

    def __init__(self, cfg: ObservationNoiseConfig, rng: np.random.Generator, num_joints: int):
        self.cfg = cfg
        self.rng = rng
        self.num_joints = num_joints
        self.bias = np.zeros(num_joints)

    def _scales(self, joint: float, gripper: float) -> np.ndarray:
        scales = np.full(self.num_joints, joint)
        scales[-1] = gripper
        return scales

    def sample(self) -> None:
        scales = self._scales(self.cfg.joint_bias_deg, self.cfg.gripper_bias)
        self.bias = self.rng.uniform(-1.0, 1.0, self.num_joints) * scales

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        scales = self._scales(self.cfg.joint_noise_deg, self.cfg.gripper_noise)
        return positions + self.bias + self.rng.uniform(-1.0, 1.0, self.num_joints) * scales
