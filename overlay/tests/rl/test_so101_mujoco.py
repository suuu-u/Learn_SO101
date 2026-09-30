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

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")
pytest.importorskip("placo")

from lerobot.rl.so101_mujoco.configs import SO101PickPlaceEnvConfig  # noqa: E402
from lerobot.rl.so101_mujoco.env import SO101PickPlaceEnv  # noqa: E402
from lerobot.rl.so101_mujoco.networks import ActorCritic, VisionStudent  # noqa: E402


@pytest.fixture(scope="module")
def env():
    env = SO101PickPlaceEnv(SO101PickPlaceEnvConfig(), seed=0)
    yield env
    env.close()


def test_reset_observation(env):
    obs = env.reset()
    assert obs["state"].shape == (18,)
    assert obs["teacher"].dtype == np.float32
    # Joint velocities of the first observation are zero, like `JointVelocityProcessorStep`
    np.testing.assert_array_equal(obs["state"][6:12], 0.0)
    assert "front" not in obs


def test_ee_delta_moves_end_effector(env):
    env.reset()
    ee_before = env.robot.data.site_xpos[env._ee_site].copy()
    for _ in range(3):
        env.step(np.array([1.0, 0.0, 0.0, 1.0]))
    ee_after = env.robot.data.site_xpos[env._ee_site]
    assert ee_after[0] - ee_before[0] > 0.03


def test_discrete_gripper_semantics(env):
    env.reset()
    obs, *_ = env.step(np.array([0.0, 0.0, 0.0, 0.0]))
    for _ in range(3):
        obs, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0]))
    # 0 opens to max_gripper_pos, "stay" holds that target
    assert obs["state"][5] == pytest.approx(env.cfg.max_gripper_pos, abs=1.0)
    for _ in range(3):
        obs, *_ = env.step(np.array([0.0, 0.0, 0.0, 2.0]))
    assert obs["state"][5] < 10.0


def test_reward_range_and_truncation(env):
    env.reset()
    rng = np.random.default_rng(0)
    for step in range(env.max_episode_steps):
        _, reward, terminated, truncated, _ = env.step(np.r_[rng.uniform(-1, 1, 3), rng.integers(0, 3)])
        # [-1, 0] plus the IK-reference penalty
        assert -1.5 <= reward <= 0.0
        if terminated:
            break
    assert terminated or (truncated and step == env.max_episode_steps - 1)


def _place_cube(env, xyz):
    import mujoco

    env.robot.data.qpos[env.robot._cube_qpos : env.robot._cube_qpos + 3] = xyz
    mujoco.mj_forward(env.robot.model, env.robot.data)


def test_stage_grows_towards_ring():
    # The optional transport term (off by default)
    cfg = SO101PickPlaceEnvConfig()
    cfg.reward.transport_weight = 2.0
    env = SO101PickPlaceEnv(cfg, seed=0)
    env.reset()
    ring = env._ring_center
    grasped = (True, True)
    _place_cube(env, [ring[0] + 0.2, ring[1], 0.06])
    far = env._stage(grasped, placed=False)
    _place_cube(env, [ring[0] + 0.1, ring[1], 0.06])
    near = env._stage(grasped, placed=False)
    _place_cube(env, [ring[0], ring[1], 0.03])
    above = env._stage(grasped, placed=False)
    assert far < near < above
    assert env._stage((False, False), placed=True) == 6.0
    env.close()


def test_object_in_ring_and_not_held_counts_as_placed(env):
    env.reset()
    ring = env._ring_center
    # Released over the ring, not settled yet: already worth a placed object, so going back to the
    # object and grasping it again (reach + grasp + lift <= 3) only loses reward
    _place_cube(env, [ring[0], ring[1], 0.03])
    assert env._stage((False, False), placed=False) == 6.0
    assert env._stage((True, True), placed=False) <= 3.0
    # Outside the ring the default reward has no transport / release terms: only the reach term
    _place_cube(env, [ring[0] + 0.1, ring[1], 0.03])
    assert env._stage((False, False), placed=False) <= 1.0


def test_carry_region_rewards_lifting_over_dragging(env):
    ring = env._ring_center
    # Just outside the ring wall, dragging on the table is farther from the region than lifting
    dragged = env._carry_distance(np.array([ring[0], ring[1] + 0.06, 0.01]))
    lifted = env._carry_distance(np.array([ring[0], ring[1] + 0.06, 0.06]))
    assert dragged > lifted > 0.0
    # Inside the ring the cube is in the region at any height, so lowering it is free
    assert env._carry_distance(np.array([ring[0], ring[1], 0.06])) == 0.0
    assert env._carry_distance(np.array([ring[0], ring[1], 0.015])) == 0.0


def test_potential_shaping_pays_progress_not_holding():
    cfg = SO101PickPlaceEnvConfig()
    cfg.reward.potential_shaping = True
    env = SO101PickPlaceEnv(cfg, seed=0)
    env.reset()
    for _ in range(3):
        _, reward, *_ = env.step(np.array([0.0, 0.0, 0.0, 1.0]))
    assert abs(reward) < 0.05
    env.close()


def test_bound_penalty_except_floor(env):
    env.reset()
    for _ in range(15):
        env.step(np.array([0.0, 0.0, -1.0, 1.0]))
    # Pressing onto the floor bound is part of grasping and stays free
    assert env._bound_hits == 0
    # The workspace top is reachable, pushing into it counts
    for _ in range(15):
        env.step(np.array([0.0, 0.0, 1.0, 1.0]))
    assert env._bound_hits == 1


def test_ring_randomization_moves_ring_and_success_test():
    cfg = SO101PickPlaceEnvConfig(ring_position_range=0.05)
    env = SO101PickPlaceEnv(cfg, seed=0)
    centers = []
    for _ in range(5):
        env.reset()
        centers.append(env._ring_center.copy())
        # The placement test of the robot follows the moved ring
        np.testing.assert_array_equal(env.robot._ring_center, env._ring_center)
    centers = np.array(centers)
    assert np.all(np.abs(centers - env._ring_base) <= 0.05 + 1e-9)
    assert np.ptp(centers, axis=0).min() > 0.0
    env.close()


def test_low_ee_penalty_grows_and_resets(env):
    env.reset()
    counts = []
    for _ in range(20):
        env.step(np.array([0.0, 0.0, -1.0, 1.0]))
        counts.append(env._low_ee_steps)
    # The EE reaches the table within a few steps, then the time spent low keeps growing
    assert counts[-1] > env.cfg.reward.low_ee_grace_steps
    assert counts == sorted(counts)
    for _ in range(5):
        env.step(np.array([0.0, 0.0, 1.0, 1.0]))
    assert env._low_ee_steps == 0


def test_deploy_checks_env_config_against_training():
    import draccus

    from lerobot.robots import so101_sim  # noqa: F401
    from lerobot.teleoperators import so_leader  # noqa: F401
    from lerobot.rl.gym_manipulator import GymManipulatorConfig
    from lerobot.rl.so101_mujoco.configs import resolve_path
    from lerobot.rl.so101_mujoco.deploy import check_compatible

    path = resolve_path("src/lerobot/configs/env_config_so101_sim.json")
    env_cfg = draccus.parse(GymManipulatorConfig, config_path=path, args=[]).env
    check_compatible(SO101PickPlaceEnvConfig(), env_cfg)
    env_cfg.processor.inverse_kinematics.end_effector_step_sizes["x"] = 0.03
    with pytest.raises(ValueError, match="EE step sizes"):
        check_compatible(SO101PickPlaceEnvConfig(), env_cfg)


def test_bar_object_and_return_home():
    cfg = SO101PickPlaceEnvConfig(object_size=[0.01, 0.02, 0.005], return_home=True)
    env = SO101PickPlaceEnv(cfg, seed=0)
    env.reset()
    model = env.robot.model
    np.testing.assert_allclose(model.geom_size[env._cube_geom], [0.005, 0.01, 0.0025])
    # The bar lies flat on the table after the reset
    assert env.robot.object_height() == pytest.approx(0.0025, abs=1e-3)
    # A placed object is worth 6 + home_weight with the gripper home, 6 once it is `home_radius` away
    full = 6.0 + env.cfg.reward.home_weight
    assert env._stage((False, False), placed=True) == pytest.approx(full, abs=1e-6)
    env.robot._home_ee = env.robot._home_ee + [env.cfg.reward.home_radius, 0.0, 0.0]
    assert env._stage((False, False), placed=True) == pytest.approx(6.0, abs=1e-6)
    # Success needs the gripper home as well
    env.robot._placed_time = 1.0
    assert not env.robot.is_success()
    env.close()


def test_action_rate_penalty_prefers_smooth_actions():
    cfg = SO101PickPlaceEnvConfig()
    cfg.reward.action_rate_penalty = 1.0
    returns = {}
    for name, sign in (("smooth", lambda t: 1.0), ("jittery", lambda t: (-1.0) ** t)):
        env = SO101PickPlaceEnv(cfg, seed=0)
        env.reset()
        returns[name] = sum(env.step(np.array([0.0, 0.3 * sign(t), 0.0, 1.0]))[1] for t in range(6))
        env.close()
    assert returns["jittery"] < returns["smooth"] - 1.0


def test_physics_randomization_is_per_env_and_restored():
    from lerobot.rl.so101_mujoco.configs import PhysicsRandomizationConfig

    cfg = SO101PickPlaceEnvConfig(physics_randomization=PhysicsRandomizationConfig(enabled=True))
    first, second = SO101PickPlaceEnv(cfg, seed=0), SO101PickPlaceEnv(cfg, seed=1)
    plain = SO101PickPlaceEnv(SO101PickPlaceEnvConfig(), seed=2)
    model = first.robot.model
    assert second.robot.model is model and plain.robot.model is model
    first.reset()
    second.reset()
    gains = []
    for env in (first, second):
        env._use_physics()
        gains.append(model.actuator_gainprm[:, 0].copy())
    assert not np.allclose(gains[0], gains[1])
    # An env without randomization sharing the model gets the nominal physics back
    plain.reset()
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], model.actuator_biasprm[:, 1] * -1.0)
    first._use_physics()
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], gains[0])
    for env in (first, second, plain):
        env.close()


def test_action_delay_holds_the_first_commands():
    cfg = SO101PickPlaceEnvConfig(action_delay_max=2)
    env = SO101PickPlaceEnv(cfg, seed=0)
    for _ in range(10):
        env.reset()
        if len(env._pending_actions) == 2:
            break
    ee = env.robot.data.site_xpos[env._ee_site].copy()
    for _ in range(2):
        env.step(np.array([1.0, 0.0, 0.0, 1.0]))
    # The two first commands wait: the arm holds still
    assert np.linalg.norm(env.robot.data.site_xpos[env._ee_site] - ee) < 0.005
    env.step(np.array([1.0, 0.0, 0.0, 1.0]))
    assert env.robot.data.site_xpos[env._ee_site][0] - ee[0] > 0.01
    env.close()


def test_joint_reading_noise_reaches_the_state():
    from lerobot.rl.so101_mujoco.configs import ObservationNoiseConfig

    noisy = SO101PickPlaceEnv(SO101PickPlaceEnvConfig(observation_noise=ObservationNoiseConfig(enabled=True)), seed=0)
    clean = SO101PickPlaceEnv(SO101PickPlaceEnvConfig(), seed=0)
    error = noisy.reset()["state"][:5] - clean.reset()["state"][:5]
    assert np.abs(error).max() > 0.05
    assert np.abs(error).max() <= 1.0 + 0.3 + 0.2
    noisy.close()
    clean.close()


def test_student_ignores_masked_state_features():
    student = VisionStudent(18, 3, (64, 64), 16, [32], masked_state_dims=list(range(12, 18))).eval()
    images = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    state = torch.randn(1, 18)
    changed = state.clone()
    changed[0, 12:] += 100.0
    np.testing.assert_allclose(student(images, images, state)[0].detach(), student(images, images, changed)[0].detach())
    # The mask is rebuilt from the constructor, not stored in checkpoints
    assert "state_mask" not in student.state_dict()


def test_networks_shapes():
    policy = ActorCritic(40, 3, [32], [32], init_noise_std=0.5)
    actions, log_prob, value = policy.act(torch.randn(5, 40))
    assert actions.shape == (5, 4) and log_prob.shape == (5,) and value.shape == (5,)
    assert set(actions[:, -1].tolist()) <= {0.0, 1.0, 2.0}

    student = VisionStudent(18, 3, (128, 128), 16, [32], image_shift_pad=4)
    images = torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.uint8)
    continuous, logits = student(images, images, torch.randn(2, 18), augment=True)
    assert continuous.shape == (2, 3) and logits.shape == (2, 3)
    assert continuous.abs().max() <= 1.0


def test_photometric_changes_are_per_camera_and_off_by_default():
    from lerobot.rl.so101_mujoco.configs import VisualRandomizationConfig
    from lerobot.rl.so101_mujoco.randomization import ImageCorruptor

    image = np.random.default_rng(0).integers(0, 256, (3, 16, 16), dtype=np.uint8)
    corruptor = ImageCorruptor(VisualRandomizationConfig(), np.random.default_rng(0))
    corruptor.sample()
    assert np.array_equal(corruptor(image, "front"), image)

    cfg = VisualRandomizationConfig(
        exposure=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.7, 1.3], gamma=[0.85, 1.2]
    )
    corruptor = ImageCorruptor(cfg, np.random.default_rng(0))
    corruptor.sample()
    front, wrist = corruptor(image, "front"), corruptor(image, "wrist")
    assert front.shape == image.shape and front.dtype == np.uint8
    assert not np.array_equal(front, image) and not np.array_equal(front, wrist)
    # Fixed for the episode
    assert np.array_equal(corruptor(image, "front"), front)


def test_per_camera_image_sizes_reach_the_student():
    from lerobot.rl.so101_mujoco.configs import VisualRandomizationConfig

    cfg = SO101PickPlaceEnvConfig(
        front_resize=[38, 58],
        wrist_resize=[48, 64],
        visual_randomization=VisualRandomizationConfig(enabled=True),
    )
    env = SO101PickPlaceEnv(cfg, seed=0, render=True)
    obs = env.reset()
    assert obs["front"].shape == (3, 38, 58) and obs["wrist"].shape == (3, 48, 64)
    student = VisionStudent(18, 3, (38, 58), 16, [32], image_shift_pad=2, wrist_image_size=(48, 64))
    front, wrist = torch.from_numpy(obs["front"][None]), torch.from_numpy(obs["wrist"][None])
    continuous, logits = student(front, wrist, torch.randn(1, 18), augment=True)
    assert continuous.shape == (1, 3) and logits.shape == (1, 3)
    env.close()


def test_camera_latency_shows_an_older_frame():
    from lerobot.rl.so101_mujoco.configs import CameraTimingConfig

    def wrist_images(timing: CameraTimingConfig) -> list[np.ndarray]:
        cfg = SO101PickPlaceEnvConfig(image_resize=[48, 64], camera_timing=timing)
        env = SO101PickPlaceEnv(cfg, seed=0, render=True)
        images = [env.reset()["wrist"]]
        for _ in range(4):
            images.append(env.step(np.array([1.0, 0.0, 0.0, 1.0]))[0]["wrist"])
        env.close()
        return images

    live = wrist_images(CameraTimingConfig())
    # A latency just below two control periods: each frame is (almost) the one of two steps before
    late = wrist_images(CameraTimingConfig(latency_s=[0.19, 0.19]))
    assert np.array_equal(late[0], live[0]) and np.array_equal(late[1], live[0])
    for step in (3, 4):
        image = late[step].astype(int)
        assert np.abs(image - live[step - 2]).mean() < np.abs(image - live[step]).mean()
    # Motion blur averages renders over the exposure: the moving view differs from a sharp frame
    blurred = wrist_images(CameraTimingConfig(exposure_s=[0.05, 0.05], blur_samples=4))
    assert np.array_equal(blurred[0], live[0]) and not np.array_equal(blurred[3], live[3])


def test_lamp_and_glossy_table_are_restored_for_plain_envs():
    from lerobot.rl.so101_mujoco.configs import VisualRandomizationConfig

    vis = VisualRandomizationConfig(
        enabled=True, lamp_prob=1.0, light_specular=[0.8, 0.8], table_reflectance=[0.3, 0.3]
    )
    env = SO101PickPlaceEnv(SO101PickPlaceEnvConfig(visual_randomization=vis), seed=0, render=True)
    env.reset()
    model = env.robot.model
    lamp = model.light("dr_lamp").id
    assert model.light_active[lamp] == 1 and model.light_castshadow[lamp] == 1
    assert model.mat_reflectance[model.geom_matid[model.geom("table").id]] == pytest.approx(0.3)
    # Another env of the same process without randomization renders the plain scene
    from lerobot.rl.so101_mujoco.randomization import restore_appearance

    restore_appearance(model)
    assert model.light_active[lamp] == 0 and model.geom_matid[model.geom("table").id] == -1
    assert np.allclose(model.light_specular[0], 0.3)
    env.close()


def test_rendering_ignores_the_appearance_of_other_envs():
    from lerobot.rl.so101_mujoco.configs import VisualRandomizationConfig

    def first_wrist_frame() -> np.ndarray:
        env = SO101PickPlaceEnv(SO101PickPlaceEnvConfig(image_resize=[48, 64]), seed=0, render=True)
        frame = env.reset()["wrist"]
        env.close()
        return frame

    plain = first_wrist_frame()
    # An env with randomized lights and camera poses shares the model, then a plain env renders
    vis = VisualRandomizationConfig(enabled=True, light_direction_std=1.0, camera_rotation_deg=10.0)
    randomized = SO101PickPlaceEnv(SO101PickPlaceEnvConfig(visual_randomization=vis), seed=0, render=True)
    randomized.reset()
    randomized.close()
    assert np.array_equal(first_wrist_frame(), plain)
