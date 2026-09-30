# Learn_SO101：SO101 HIL-SERL & MuJoCo Sim2Real（基于 LeRobot）

在 [LeRobot](https://github.com/huggingface/lerobot) 上为 SO-101 机械臂做的强化学习实验代码与配置：

- **HIL-SERL**：真机 / MuJoCo 仿真上的人在回路 SAC，带奖励分类器、键盘 / leader 臂接管、分阶段奖励
- **MuJoCo PPO → 视觉蒸馏（sim2real）**：用特权状态训练 PPO 教师（参照 [OpenSO-101](https://github.com/jixinyan/OpenSO-101) 的 rsl_rl PPO 配置自行实现，不依赖 rsl_rl 包），再用 DAgger 蒸馏成只看前置 / 腕部相机的视觉学生，经与 HIL-SERL 相同的 `gym_manipulator` 管线部署到仿真或真机
- **模仿学习基线**：ACT、SmolVLA

任务：`pick up the black cubic to the circle`（把黑色方块夹起放进圆圈），控制频率 10 fps。

> English: RL code and configs for the SO-101 arm on top of LeRobot — HIL-SERL (real + MuJoCo sim), a privileged PPO teacher distilled into a camera-only student for sim2real, and ACT / SmolVLA baselines. Run `bash setup.sh`, then follow the commands below.

## 仓库结构

本仓库不是 LeRobot 的完整副本，只包含**新增和修改过的文件**，按 LeRobot 的目录结构放在 `overlay/` 下；`setup.sh` 会把 LeRobot 固定到上游 commit [`4aaff99b`](https://github.com/huggingface/lerobot/commit/4aaff99be4a1d81568c08c8f0296b41b40c99ec4) 并覆盖这些文件。

```
.
├── setup.sh                         克隆 LeRobot@4aaff99b → 覆盖 overlay/ → pip 安装
├── overlay/                         按 LeRobot 目录结构放置的新增 / 修改文件
│   ├── src/lerobot/configs/         ★ 全部实验配置（见下表）
│   ├── src/lerobot/rl/              HIL-SERL（actor / learner / buffer / gym_manipulator / eval_policy ...）
│   ├── src/lerobot/rl/so101_mujoco/ PPO 教师、DAgger 蒸馏、评估、部署
│   ├── src/lerobot/robots/so101_sim/  MuJoCo 仿真机械臂 so101_sim_follower
│   ├── src/lerobot/processor/, policies/gaussian_actor/, teleoperators/ ...
│   ├── Simulation/SO101/            SO101 的 URDF / MJCF / STL
│   └── tests/rl/test_so101_mujoco.py
├── patches/upstream_modifications.diff   对上游已有文件的全部改动（仅供查阅）
└── extras/
    ├── environment/pip_freeze.txt   原实验环境的完整依赖版本
    └── calibration/                 作者机械臂的标定文件（仅供参考，请自行标定）
```

### 配置文件（`overlay/src/lerobot/configs/`）

| 文件 | 用途 |
|---|---|
| `env_config_so101.json` | 真机环境：SO101 follower + leader 臂遥操作，录制 / 接管 |
| `env_config_so101_sim.json` | MuJoCo 仿真环境（`so101_sim_follower`）+ leader 臂 |
| `env_config_so101_deploy_hires.json` | 真机部署视觉学生（键盘接管，不缩放图像） |
| `env_config_gym_hil.json` | gym-hil Panda 仿真（键盘） |
| `reward_classifier_train_so101.json` | 训练奖励分类器 |
| `train_config_hilserl_so101.json` | HIL-SERL 真机训练（learner + actor） |
| `train_config_hilserl_so101_sim.json` | HIL-SERL 仿真训练（learner + actor） |
| `so101_mujoco/ppo_realscene.yaml` | PPO 教师训练（`teacher_realscene.pt` 的原始配置） |
| `so101_mujoco/distill_realcam.yaml` | DAgger 蒸馏（`student_realcam.pt` 的原始配置） |

## 已发布的模型与数据（Hugging Face）

| Hub 仓库 | 内容 |
|---|---|
| [`suuu3/so101_mujoco_sim2real`](https://huggingface.co/suuu3/so101_mujoco_sim2real) | `teacher_robust_best.pt`、`teacher_realscene.pt`（PPO 教师），`student_realcam.pt`（视觉学生，用于部署），以及训练 config / metrics |
| [`suuu3/so101_reward_classifier`](https://huggingface.co/suuu3/so101_reward_classifier) | HIL-SERL 奖励分类器 |
| [`suuu3/so101_act_test3`](https://huggingface.co/suuu3/so101_act_test3) | ACT（40k steps） |
| [`suuu3/so101_smolvla_test3`](https://huggingface.co/suuu3/so101_smolvla_test3) | SmolVLA（基于 `lerobot/smolvla_base` 微调 20k steps） |
| [`suuu3/so101_test3`](https://huggingface.co/datasets/suuu3/so101_test3)（dataset） | ACT / SmolVLA 训练数据，50 条示范 |

HIL-SERL 的录制数据与 SAC 策略未发布；配置中以 `YOUR_HF_USER/...` 表示，需自行录制（见 1.1）。

国内网络可用镜像下载：`export HF_ENDPOINT=https://hf-mirror.com`。

## 0. 安装

原实验环境：Ubuntu，Python 3.12，PyTorch 2.11 + CUDA 13.0，MuJoCo 3.8.1，placo 0.9.15，gym-hil 0.1.14（完整版本见 `extras/environment/pip_freeze.txt`）。

```bash
git clone https://github.com/suuu-u/Learn_SO101.git
cd Learn_SO101
conda create -n lerobot python=3.12 -y && conda activate lerobot
bash setup.sh            # 生成 ./lerobot 并安装；NO_INSTALL=1 bash setup.sh 只克隆不安装
cd lerobot               # 以下所有命令都在这里执行
```

配置里的 `Simulation/SO101/...`、`outputs/...` 都是相对 `lerobot/` 根目录的路径。

**只用仿真时**无需任何硬件；leader 臂接管需要一个 SO101 leader（或把 `teleop` 改成 `keyboard_hil`）。

**用真机时**：
- 用 `lerobot-find-port`、`lerobot-find-cameras` 找到自己的串口和相机，改掉配置中的 `/dev/ttyACM*` 与 `/dev/v4l/by-id/...`。
- 用 `lerobot-calibrate` 标定（配置中机械臂 `id` 为 `my_awesome_follower_arm` / `my_awesome_leader_arm`）。
- 前置相机裁剪 `[0, 0, 379, 575]` 和末端工作空间 `min [0.0673, -0.1982, 0.005]` / `max [0.3155, 0.2947, 0.1477]` 与相机摆放、桌面布置相关，换场景需用 `crop_dataset_roi` 与 `lerobot-find-joint-limits` 重新测定。
- 把配置中的 `YOUR_HF_USER` 换成自己的 HF 用户名：
  `grep -rl YOUR_HF_USER src/lerobot/configs | xargs sed -i 's/YOUR_HF_USER/<你的HF用户名>/g'`

## 1. HIL-SERL

```bash
# 1.1 录制示范 / 分类器数据（把配置里的 "mode" 设为 "record"）
python -m lerobot.rl.gym_manipulator --config_path src/lerobot/configs/env_config_so101.json       # 真机
python -m lerobot.rl.gym_manipulator --config_path src/lerobot/configs/env_config_so101_sim.json   # 仿真

# 1.2 交互式框选 ROI，裁剪并缩放到 128x128（生成 <repo_id>_cropped_resized）
python -m lerobot.rl.crop_dataset_roi --repo-id YOUR_HF_USER/so101_hilserl_classifier

# 1.3 训练奖励分类器（也可直接用已发布的 suuu3/so101_reward_classifier，真机配置默认指向它）
lerobot-train --config_path src/lerobot/configs/reward_classifier_train_so101.json

# 1.4 在线训练：learner 与 actor 各开一个终端
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so101_sim.json
python -m lerobot.rl.actor   --config_path src/lerobot/configs/train_config_hilserl_so101_sim.json
#     真机把 _so101_sim 换成 _so101；输出在 outputs/train/HIL_SERL/<时间戳>_{learner,actor}_<名称>/

# 1.5 评估（不更新权重，Space 键可随时接管）
python -m lerobot.rl.eval_policy --policy_path=outputs/train/HIL_SERL/<run>/checkpoints/last/pretrained_model --n_episodes=50 --headless=true
```

## 2. MuJoCo PPO 教师 → 视觉学生（sim2real）

```bash
hf download suuu3/so101_mujoco_sim2real --local-dir outputs/so101_mujoco   # 放到配置默认读取的位置
```

模型链路：

```
PPO（鲁棒随机化，6700 iter） ──► teacher_robust_best.pt
        │ 续训到 8200 iter（ppo_realscene.yaml，对齐真实场景）
        ▼
teacher_realscene.pt ──DAgger 蒸馏（distill_realcam.yaml）──► student_realcam.pt ──► 部署
                              ▲ 初始化自中间学生 student_hires.pt（由 teacher_robust_best 蒸馏，未发布）
```

训练过程中的滚动成功率（带域随机化）：`teacher_realscene` 约 0.54（第 7600 次迭代），`student_realcam` 约 0.48（第 200 次迭代）。

```bash
# 2.1 PPO 教师：从 teacher_robust_best.pt 续训（配置中 resume 已指向它）
python -m lerobot.rl.so101_mujoco.train_ppo --config_path src/lerobot/configs/so101_mujoco/ppo_realscene.yaml
#     从零训练：加 --resume null（teacher_robust_best 的原始配置未保留，从零结果可能不同）

# 2.2 DAgger 蒸馏视觉学生
#     原流程需先得到中间学生 student_hires.pt：
python -m lerobot.rl.so101_mujoco.train_distill --teacher_checkpoint outputs/so101_mujoco/teacher_robust_best.pt \
    --output_dir outputs/so101_mujoco/distill_hires --num_envs 16 --num_workers 4
cp outputs/so101_mujoco/distill_hires/<run>/model_best.pt outputs/so101_mujoco/student_hires.pt
python -m lerobot.rl.so101_mujoco.train_distill --config_path src/lerobot/configs/so101_mujoco/distill_realcam.yaml
#     或跳过中间学生，从头蒸馏：上一条命令加 --init_checkpoint null

# 2.3 仿真评估（成功率 / MuJoCo 可视化 / 录视频）
python -m lerobot.rl.so101_mujoco.eval --checkpoint outputs/so101_mujoco/student_realcam.pt
python -m lerobot.rl.so101_mujoco.eval --checkpoint outputs/so101_mujoco/student_realcam.pt --viewer true
python -m lerobot.rl.so101_mujoco.eval --checkpoint outputs/so101_mujoco/student_realcam.pt --video_path outputs/eval.mp4

# 2.4 部署：仿真与真机只差 env 配置
python -m lerobot.rl.so101_mujoco.deploy --headless true --n_episodes 50        # 仿真，默认 student_realcam.pt
python -m lerobot.rl.so101_mujoco.deploy --env_config src/lerobot/configs/env_config_so101_deploy_hires.json \
    --action_scale 0.5 --n_episodes 3                                            # 真机，建议先降低动作幅度
```

## 3. 模仿学习基线（ACT / SmolVLA）

```bash
hf download suuu3/so101_act_test3 --local-dir outputs/release/so101_act_test3
lerobot-train --config_path outputs/release/so101_act_test3/train_config.json   # 用原超参在 suuu3/so101_test3 上重训
```

SmolVLA 同理（`suuu3/so101_smolvla_test3`）。

## 4. 测试

```bash
pytest tests/rl/test_so101_mujoco.py -v
```

## 致谢与许可

- 基于 [huggingface/lerobot](https://github.com/huggingface/lerobot)（Apache-2.0），本仓库同样以 Apache-2.0 发布。
- SO101 URDF / MJCF 来自 [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)；电机参数参考 [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini)。
- PPO 超参与训练流程参考 [OpenSO-101](https://github.com/jixinyan/OpenSO-101)。
