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

"""OpenSO-101-style RL on the MuJoCo `so101_sim` scene: privileged PPO teacher -> vision student.

1. `python -m lerobot.rl.so101_mujoco.train_ppo`      PPO on privileged state (cube pose, contacts).
2. `python -m lerobot.rl.so101_mujoco.train_distill`  DAgger into a CNN student that only sees the
   front / wrist cameras and the 18-dim joint state of the HIL-SERL pipeline.
3. `python -m lerobot.rl.so101_mujoco.eval`           success rate, MuJoCo viewer, or video.
4. `python -m lerobot.rl.so101_mujoco.deploy`         run the student through the `gym_manipulator`
   pipeline, on the `so101_sim_follower` or the real arm (only the env config differs).

Requires `mujoco` and `placo` (the same kinematics as `gym_manipulator`).
"""
