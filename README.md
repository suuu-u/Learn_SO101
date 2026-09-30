# Learn_SO101：SO101 HIL-SERL & MuJoCo Sim2Real（基于 LeRobot）

在 [LeRobot](https://github.com/huggingface/lerobot) 上为 SO-101 机械臂做的强化学习实验代码与配置：

- **模仿学习基线**：ACT、SmolVLA（遥操作采集示范 → 训练 → 真机评估）
- **HIL-SERL**：真机 / MuJoCo 仿真上的人在回路 SAC，带奖励分类器、键盘 / leader 臂接管、分阶段奖励
- **MuJoCo PPO → 视觉蒸馏（sim2real）**：用特权状态训练 PPO 教师（参照 [OpenSO-101](https://github.com/jixinyan/OpenSO-101) 的 rsl_rl PPO 配置自行实现，不依赖 rsl_rl 包），再用 DAgger 蒸馏成只看前置 / 腕部相机的视觉学生，经与 HIL-SERL 相同的 `gym_manipulator` 管线部署到仿真或真机

任务：把黑色方块夹起放进圆圈。模仿学习数据集的任务描述为 `Grab the black cuboid to the circle`（30 fps）；强化学习部分为 `pick up the black cubic to the circle`（10 fps）。

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
| `so101_mujoco/ppo_realscene.yaml` | PPO 教师训练（与 `teacher_realscene.pt` 相同的环境与超参，从零训练） |
| `so101_mujoco/distill_realcam.yaml` | DAgger 蒸馏（与 `student_realcam.pt` 相同的环境与超参，教师为 `teacher_realscene.pt`） |

## 模型与数据（Hugging Face）

| Hub 仓库 | 内容 |
|---|---|
| [`suuu3/so101_mujoco_sim2real`](https://huggingface.co/suuu3/so101_mujoco_sim2real) | `teacher_realscene.pt`（PPO 教师），`student_realcam.pt`（视觉学生，用于部署），以及训练 config / metrics |
| [`suuu3/so101_reward_classifier`](https://huggingface.co/suuu3/so101_reward_classifier) | HIL-SERL 奖励分类器 |
| [`suuu3/so101_act_test3`](https://huggingface.co/suuu3/so101_act_test3) | ACT（40k steps，chunk_size=100） |
| [`suuu3/so101_smolvla_test3`](https://huggingface.co/suuu3/so101_smolvla_test3) | SmolVLA（基于 `lerobot/smolvla_base` 微调 20k steps，chunk_size=50） |
| [`suuu3/so101_test3`](https://huggingface.co/datasets/suuu3/so101_test3)（dataset） | ACT / SmolVLA 训练数据，50 条示范 |

国内网络可用镜像下载：`export HF_ENDPOINT=https://hf-mirror.com`。

## 0. 安装

### 0.1 安装依赖

原实验环境：Ubuntu，Python 3.12，PyTorch 2.11 + CUDA 13.0，MuJoCo 3.8.1，placo 0.9.15，gym-hil 0.1.14（完整版本见 `extras/environment/pip_freeze.txt`）。

```bash
git clone https://github.com/suuu-u/Learn_SO101.git
cd Learn_SO101
conda create -n lerobot python=3.12 -y && conda activate lerobot
bash setup.sh            # 生成 ./lerobot 并安装；NO_INSTALL=1 bash setup.sh 只克隆不安装
cd lerobot               # 以下所有命令都在这里执行
```

配置里的 `Simulation/SO101/...`、`outputs/...` 都是相对 `lerobot/` 根目录的路径。

### 0.2 验证安装

```bash
pip install pytest
pytest tests/rl/test_so101_mujoco.py -v
```

运行 24 个单元测试，检查 MuJoCo 仿真环境与网络代码是否正常，全部显示 `PASSED` 即安装成功（约 10 秒，不训练模型、不需要真机）。测试内容包括：

- 环境：reset 后状态为 18 维，末端位移指令能移动机械臂，夹爪开 / 不动 / 关的语义
- 奖励：取值范围与回合截断，抬起比拖动得分高，越界惩罚，方块放进圆圈并松开才算成功
- 域随机化：物理参数按环境独立随机且可还原，动作延迟、关节读数噪声、相机延迟生效
- 网络与部署：教师 / 学生网络的输入输出维度，部署前的 env 配置一致性检查

若显示 `SKIPPED`，说明 `mujoco` 或 `placo` 没有装好；若有 `FAILED`，多半是依赖版本与原实验环境不一致，可对照 `extras/environment/pip_freeze.txt`。这些测试只覆盖仿真部分，真机相关功能需要按下面的说明接好硬件后再验证。

### 0.3 硬件与配置准备

**只用仿真时**无需任何硬件；leader 臂接管需要一个 SO101 leader（或把 `teleop` 改成 `keyboard_hil`）。

**用真机时**：
- 用 `lerobot-find-port`、`lerobot-find-cameras` 找到自己的串口和相机，改掉配置中的 `/dev/ttyACM*` 与 `/dev/v4l/by-id/...`。
- 用 `lerobot-calibrate` 标定（配置中机械臂 `id` 为 `my_awesome_follower_arm` / `my_awesome_leader_arm`）。
- 前置相机裁剪 `[0, 0, 379, 575]` 和末端工作空间 `min [0.0673, -0.1982, 0.005]` / `max [0.3155, 0.2947, 0.1477]` 与相机摆放、桌面布置相关，换场景需用 `crop_dataset_roi` 与 `lerobot-find-joint-limits` 重新测定。
- 把配置中的 `YOUR_HF_USER` 换成自己的 HF 用户名：
  `grep -rl YOUR_HF_USER src/lerobot/configs | xargs sed -i 's/YOUR_HF_USER/<你的HF用户名>/g'`

## 1. 模仿学习基线（ACT / SmolVLA）

使用 SO101 follower + leader 臂遥操作采集示范，前置（`front`）与侧面（`side`）两个 640×480 相机，30 fps。已发布的数据集 `suuu3/so101_test3` 为 50 条示范，可跳过 1.1 直接训练。

### 1.1 采集示范数据

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: <前置相机>, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: <侧面相机>, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=<你的HF用户名>/so101_test3 \
  --dataset.single_task="Grab the black cuboid to the circle" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --display_data=true
```

录制时 → 键提前结束当前回合，← 键重录当前回合，Esc 结束采集。默认录完上传到 Hub，只存本地加 `--dataset.push_to_hub=false`。

### 1.2 训练

```bash
# ACT：从零训练（原实验：40k steps，batch 8，lr 1e-5）
lerobot-train \
  --dataset.repo_id=suuu3/so101_test3 \
  --policy.type=act --policy.device=cuda \
  --steps=40000 --batch_size=8 --save_freq=10000 \
  --output_dir=outputs/train/act_so101 --job_name=act_so101 \
  --policy.push_to_hub=false

# SmolVLA：在 lerobot/smolvla_base 上微调（原实验：20k steps，batch 64）
#   预训练模型的相机名是 camera1/camera2，需要用 rename_map 把 front/side 对应过去
lerobot-train \
  --dataset.repo_id=suuu3/so101_test3 \
  --policy.path=lerobot/smolvla_base --policy.device=cuda \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  --steps=20000 --batch_size=64 --save_freq=5000 \
  --output_dir=outputs/train/smolvla_so101 --job_name=smolvla_so101 \
  --policy.push_to_hub=false
```

也可以直接复用原实验的完整配置：`hf download suuu3/so101_act_test3 --local-dir outputs/release/so101_act_test3`，然后 `lerobot-train --config_path outputs/release/so101_act_test3/train_config.json`。

### 1.3 真机评估

`--policy.path` 可以是 Hub 上的模型（如下），也可以是本地 checkpoint：`outputs/train/<run>/checkpoints/last/pretrained_model`。

```bash
# ACT：每回合结束后有 reset 时间摆回方块，评估过程录成 eval_ 数据集便于回看
lerobot-rollout --strategy.type=episodic \
  --policy.path=suuu3/so101_act_test3 \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: <前置相机>, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: <侧面相机>, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=<你的HF用户名>/eval_act_so101 \
  --dataset.single_task="Grab the black cuboid to the circle" \
  --dataset.num_episodes=10 --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false

# SmolVLA：同上，换模型并加相同的 rename_map
lerobot-rollout --strategy.type=episodic \
  --policy.path=suuu3/so101_smolvla_test3 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' \
  ...（其余参数同 ACT，dataset.repo_id 改为 eval_smolvla_so101）
```

只想看效果、不录数据：把 `--strategy.type=episodic` 和 `--dataset.*` 换成 `--strategy.type=base --task="Grab the black cuboid to the circle" --duration=60`。

### 1.4 动作分块：`chunk_size` 与 `n_action_steps`

两个模型都采用 action chunking：一次推理预测未来一段动作，而不是只预测下一步。

- `chunk_size`：每次推理预测多少步动作（训练时确定，推理时不能改）
- `n_action_steps`：预测出的动作中实际执行多少步后再重新推理（推理时可改，需 ≤ `chunk_size`）

| 模型 | chunk_size | n_action_steps | 推理频率（30 fps） |
|---|---|---|---|
| ACT | 100 | 100 | 每 100 步推理一次，约 3.3 s |
| SmolVLA | 50 | 50 | 每 50 步推理一次，约 1.7 s；每次推理内部做 10 步 flow matching 积分 |

两者都设为 `n_action_steps = chunk_size`：整段动作执行完才看下一次画面。推理次数少、动作连贯，但执行期间是开环的，方块被碰歪不会及时纠正，段与段之间可能有跳变。推理时可以在评估命令后加参数调整：

```bash
--policy.n_action_steps=25                                    # 执行 25 步就重新推理，反应更灵敏，推理次数 ×4
--policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01   # 仅 ACT：每步推理，对重叠预测做指数加权平均，最平滑
--inference.type=rtc                                          # 仅 SmolVLA 等较慢模型：Real-Time Chunking，异步推理下一段
```

## 2. HIL-SERL

```bash
# 2.1 录制示范 / 分类器数据（把配置里的 "mode" 设为 "record"）
python -m lerobot.rl.gym_manipulator --config_path src/lerobot/configs/env_config_so101.json       # 真机
python -m lerobot.rl.gym_manipulator --config_path src/lerobot/configs/env_config_so101_sim.json   # 仿真

# 2.2 交互式框选 ROI，裁剪并缩放到 128x128（生成 <repo_id>_cropped_resized）
python -m lerobot.rl.crop_dataset_roi --repo-id YOUR_HF_USER/so101_hilserl_classifier

# 2.3 训练奖励分类器（也可直接用已发布的 suuu3/so101_reward_classifier，真机配置默认指向它）
lerobot-train --config_path src/lerobot/configs/reward_classifier_train_so101.json

# 2.4 在线训练：learner 与 actor 各开一个终端
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so101_sim.json
python -m lerobot.rl.actor   --config_path src/lerobot/configs/train_config_hilserl_so101_sim.json
#     真机把 _so101_sim 换成 _so101；输出在 outputs/train/HIL_SERL/<时间戳>_{learner,actor}_<名称>/

# 2.5 评估（不更新权重，Space 键可随时接管）
python -m lerobot.rl.eval_policy --policy_path=outputs/train/HIL_SERL/<run>/checkpoints/last/pretrained_model --n_episodes=50 --headless=true
```

## 3. MuJoCo PPO 教师 → 视觉学生（sim2real）

```bash
hf download suuu3/so101_mujoco_sim2real --local-dir outputs/so101_mujoco   # 放到配置默认读取的位置
```

模型链路：

```
PPO 教师（特权状态：方块位姿、接触等，ppo_realscene.yaml）──► teacher_realscene.pt
        │ DAgger 蒸馏（distill_realcam.yaml）
        ▼
视觉学生（前置 + 腕部相机 + 18 维关节状态）──► student_realcam.pt ──► 部署
```

训练过程中的滚动成功率（带域随机化）：`teacher_realscene` 约 0.54，`student_realcam` 约 0.48。发布的权重经过分阶段训练，按下面的配置从零训练结果可能略有差异。

```bash
# 3.1 PPO 教师：从零训练 8200 次迭代（配置中 resume: null）
python -m lerobot.rl.so101_mujoco.train_ppo --config_path src/lerobot/configs/so101_mujoco/ppo_realscene.yaml
#     在发布的教师上继续训练：加 --resume outputs/so101_mujoco/teacher_realscene.pt --max_iterations <大于 7600 的总迭代数>

# 3.2 DAgger 蒸馏视觉学生（教师为 teacher_realscene.pt，学生从头初始化，配置中 init_checkpoint: null）
python -m lerobot.rl.so101_mujoco.train_distill --config_path src/lerobot/configs/so101_mujoco/distill_realcam.yaml
#     用自己训练的教师：加 --teacher_checkpoint outputs/so101_mujoco/ppo_realscene/<run>/model_best.pt
#     在发布的学生上微调：加 --init_checkpoint outputs/so101_mujoco/student_realcam.pt

# 3.3 仿真评估（成功率 / MuJoCo 可视化 / 录视频）
python -m lerobot.rl.so101_mujoco.eval --checkpoint outputs/so101_mujoco/student_realcam.pt
python -m lerobot.rl.so101_mujoco.eval --checkpoint outputs/so101_mujoco/student_realcam.pt --viewer true
python -m lerobot.rl.so101_mujoco.eval --checkpoint outputs/so101_mujoco/student_realcam.pt --video_path outputs/eval.mp4

# 3.4 部署：仿真与真机只差 env 配置
python -m lerobot.rl.so101_mujoco.deploy --headless true --n_episodes 50        # 仿真，默认 student_realcam.pt
python -m lerobot.rl.so101_mujoco.deploy --env_config src/lerobot/configs/env_config_so101_deploy_hires.json \
    --action_scale 0.5 --n_episodes 3                                            # 真机，建议先降低动作幅度
```

## 致谢与许可

- 基于 [huggingface/lerobot](https://github.com/huggingface/lerobot)（Apache-2.0），本仓库同样以 Apache-2.0 发布。
- SO101 URDF / MJCF 来自 [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)；电机参数参考 [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini)。
- PPO 超参与训练流程参考 [OpenSO-101](https://github.com/jixinyan/OpenSO-101)。
