# ALOHA ROS × LeRobot 中文使用指南

[日本語版](./README_JA.md)

从下载代码到机器人跑起模型的完整流程：环境配置 → 启动机器人 → 标定 → 遥操作 → 采集数据 → 转换 → 训练 → 部署运行 → HIL 补数据 → 休眠断电。

适用环境：本仓库 `aloha2` 分支；机器人 PC 为 Ubuntu 20.04 + ROS Noetic + Interbotix；四臂 ALOHA（2 条 WidowX `wx250s` 主臂，2 条 ViperX `vx300s` 从臂）+ 4 路 USB 相机（`cam_high`、`cam_low`、`cam_left_wrist`、`cam_right_wrist`）。

本机路径约定（换机器时替换成自己的）：

| 用途             | 路径                                                        |
| ---------------- | ----------------------------------------------------------- |
| 代码             | `/mnt/server_ssd2/Programs_server/lerobot`                |
| LeRobot 数据集   | `/home/robot/Dataset_and_Checkpoint/lerobot-dataset/`     |
| ALOHA2 原始 hdf5 | `/home/robot/Dataset_and_Checkpoint/dataset/<task_name>/` |
| 部署用的模型     | `/home/robot/Dataset_and_Checkpoint/checkpoints/`         |

---

## 0. 先选数据路线

数据有两种来源，**训练时用的是哪条路线的数据，部署时就必须用对应的参数**，否则模型看到的夹爪状态含义和训练时不一样，夹爪会乱动。

|                       | 路线 A：LeRobot 直接录                        | 路线 B：ALOHA2 hdf5 + 转换                                                  |
| --------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| 录制脚本              | `aloha_ros_record.py` 或 `lerobot-record` | act-triple-plus-aloha2 的 `record_episodes_aloha2.py`                     |
| 产物                  | LeRobot v3.0 数据集                           | `episode_N.hdf5`，再用 act-triple 的 `convert_aloha_hdf5_to_lerobot_v30.py` 转成 v3.0 |
| 夹爪状态              | 关节角归一化到 [0, 1]                         | ALOHA2 qpos 的归一化方式                                                    |
| 额外特征              | 无                                            | `observation.velocity`、`observation.effort`                            |
| `aloha_ros_eval.py` | `--data_format lerobot`（默认）             | `--data_format aloha2`                                                    |
| `lerobot-rollout`   | 不加夹爪参数                                  | `--robot.gripper_state_format=aloha2`                                     |
| 已有例子              | apple_grasping（ACT，30 Hz）                  | packaging_simple_2（pi05、MolmoAct2，50 Hz）                                |

两条路线里动作（action）的夹爪含义相同，都是归一化的开合度（0 = 闭合，1 = 张开）。动作都是**绝对**的关节目标值，这也是 LeRobot 的默认做法，本文所有训练命令都按绝对动作写。

---

## 1. 下载代码

```bash
cd /mnt/server_ssd2/Programs_server
git clone git@github.com:guningquan/lerobot.git
cd lerobot
git checkout aloha2
git remote add upstream https://github.com/huggingface/lerobot.git   # 以后同步官方更新用
```

---

## 2. 环境配置

### 2.1 前置条件

- ROS Noetic 和 Interbotix 工作空间 `~/interbotix_ws_noetic` 已编译好，其中有 `aloha` 包（提供 `4arms_tactile_teleop.launch`）。
- 已安装 [uv](https://docs.astral.sh/uv/)。

### 2.2 Python 环境（uv，Python 3.12）

```bash
cd /mnt/server_ssd2/Programs_server/lerobot
uv venv --python 3.12                       # 已有 .venv 可跳过
uv sync --locked --extra core_scripts --extra training --extra pi --extra molmoact2
uv pip install rospkg catkin_pkg netifaces h5py
```

- `pi` 是 pi0/pi0.5 的依赖，`molmoact2` 是 MolmoAct2 的依赖；只训练 ACT 可以不装这两个。
- 最后一行让 Python 3.12 能 `import rospy`。这几个包不在 `uv.lock` 里，**每次 `uv sync` 都会把它们删掉**，sync 完要重新执行一次这行（或者改用 `uv sync --inexact`）。缺了它们会报 `ModuleNotFoundError: No module named 'rospkg'`。
- 训练 pi05 要从 Hugging Face 下载 `lerobot/pi05_base` 和 `google/paligemma-3b-pt-224` 的分词器；后者需要先在网页上同意 PaliGemma 的许可，再 `hf auth login`。只部署不训练不需要，checkpoint 里自带分词器。

### 2.3 每个新终端的初始化

```bash
source ~/interbotix_ws_noetic/devel/setup.bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/lerobot
```

顺序不能反：`setup.bash` 设置的 `PYTHONPATH` 让 venv 能找到 rospy 和 interbotix，也不要手动 `unset PYTHONPATH`。运行时开头出现的几条 `/opt/ros/noetic/... SyntaxWarning: invalid escape sequence` 是 ROS 的 Python 3.8 代码在 3.12 下的警告，可以忽略。

### 2.4 常用变量

下面的命令都用到这几个变量。**它们只在当前终端有效**，新开终端要重新设置；变量为空时会报 `Couldn't parse '' into an int` 这类错误。建议存成 `~/aloha_env.sh`，每次 `source ~/aloha_env.sh`。

```bash
FPS=50                                  # 与数据集的 fps 一致：packaging 系列 50，apple 系列 30
TASK="packaging the items"
DATA=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/<数据集名>
CKPT=/home/robot/Dataset_and_Checkpoint/checkpoints/<模型目录>      # 里面直接是 config.json、model.safetensors
CAMS="{cam_high: {type: ros_image, topic: /usb_cam_high/image_raw, width: 640, height: 480, fps: $FPS}, cam_low: {type: ros_image, topic: /usb_cam_low/image_raw, width: 640, height: 480, fps: $FPS}, cam_left_wrist: {type: ros_image, topic: /usb_cam_left_wrist/image_raw, width: 640, height: 480, fps: $FPS}, cam_right_wrist: {type: ros_image, topic: /usb_cam_right_wrist/image_raw, width: 640, height: 480, fps: $FPS}}"

echo $FPS $DATA $CKPT; echo "$CAMS"    # 检查都不是空的
```

`CAMS` 写全四个相机即可：模型只取自己训练时用过的相机，多出来的相机只会被录进数据集。

---

## 3. 启动机器人（终端 1，全程不要关）

```bash
source ~/interbotix_ws_noetic/devel/setup.bash
roslaunch aloha 4arms_tactile_teleop.launch
```

它会启动四条臂（`master_left`、`master_right`、`puppet_left`、`puppet_right`）和四路相机话题 `/usb_cam_{high,low,left_wrist,right_wrist}/image_raw`。相机由 launch 独占，LeRobot 通过 `ros_image` 类型读话题，图像已经是 RGB，不需要再做 BGR/RGB 转换。

检查相机：`rostopic hz /usb_cam_high/image_raw`。

以下所有命令都在**另一个终端**里运行（先做 2.3 的初始化）。

---

## 4. 夹爪标定（换夹爪或重新装配后做一次）

```bash
python examples/aloha_ros/calibrate_master_gripper.py --side left
python examples/aloha_ros/calibrate_master_gripper.py --side right
python examples/aloha_ros/calibrate_puppet_gripper.py --side left
python examples/aloha_ros/calibrate_puppet_gripper.py --side right
```

每个脚本：把夹爪完全合上、完全张开，然后 Ctrl+C，把打印的 min/max 填进 `examples/aloha_ros/aloha_scripts/constants.py` 的 `MASTER_/PUPPET_<SIDE>_GRIPPER_JOINT_CLOSE / _OPEN`。从臂脚本会先把夹爪（只有夹爪）切到 PWM 模式并断扭矩，方便用手掰。

注意：标定值决定夹爪的归一化，改了之后，用旧标定录的数据和训练的模型，夹爪含义会有偏差。

---

## 5. 遥操作测试

```bash
python examples/aloha_ros/aloha_ros_teleop.py --fps $FPS                    # 双臂
python examples/aloha_ros/aloha_ros_teleop_one_arm.py --side right --fps $FPS   # 单臂
```

按键：→ 开始 / 暂停后继续；← 暂停并锁住（从臂停止跟随，主臂保持不动）；ESC 或 q 停止（四条臂回休息位并断电）。

也可以用 LeRobot 自带的命令：

```bash
lerobot-teleoperate \
  --robot.type=aloha_ros --robot.init_ros_node=false --robot.fps=$FPS --robot.cameras="$CAMS" \
  --teleop.type=aloha_teleop_ros --teleop.fps=$FPS \
  --fps=$FPS --display_data=true
```

---

## 6. 采集数据

### 6.1 路线 A：直接录成 LeRobot 数据集

**方式一：本仓库的脚本。** 先改 `examples/aloha_ros/aloha_ros_record.py` 开头的配置：`NUM_EPISODES`、`EPISODE_TIME_SEC`、`TASK_DESCRIPTION`、`HF_REPO_ID`、`DATASET_ROOT`。`DATASET_ROOT` 已有数据集时会接着往里录（fps 必须相同）。

```bash
python examples/aloha_ros/aloha_ros_record.py --fps $FPS
```

每条开始前四条臂都回到起始位，按 → 开始录。录制中：→ 提前结束并保存；← 丢弃这一条重录；ESC 停止（当前这条丢弃）。

**方式二：`lerobot-record`。**

```bash
lerobot-record \
  --robot.type=aloha_ros --robot.fps=$FPS --robot.cameras="$CAMS" \
  --teleop.type=aloha_teleop_ros --teleop.init_ros_node=false --teleop.fps=$FPS \
  --dataset.repo_id=local/<数据集名> --dataset.root=$DATA --dataset.no_stamp=true \
  --dataset.fps=$FPS --dataset.num_episodes=50 --dataset.episode_time_s=15 --dataset.reset_time_s=10 \
  --dataset.single_task="$TASK" --dataset.push_to_hub=false \
  --display_data=true
```

### 6.2 路线 B：ALOHA2 原始 hdf5

在 act-triple-plus-aloha2 仓库里录（用它的 conda 环境 `aloha`，不是本仓库的 venv）：

```bash
conda activate aloha
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/record_episodes_aloha2.py --task_name packaging_simple_2 --start_idx 0 --end_idx 49
```

任务的保存目录、每条长度（`episode_len`）、相机列表在该仓库 `aloha_scripts/constants.py` 的 `TASK_CONFIGS` 里配置，详见该仓库的 `aloha_scripts/ALOHA_SCRIPTS_USAGE_CN.md`。

---

## 7. 转换 hdf5 → LeRobot v3.0（路线 B）

数据的采集和转换统一放在 act-triple-plus-aloha2 仓库：转换脚本是 `aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py`。它要 import lerobot，所以用**本仓库的 venv** 运行，不是 act-triple 的 conda 环境 `aloha`：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py \
  --raw-dir /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 \
  --task "packaging the items" \
  --fps 50
```

- `--raw-dir` 可以给**多个目录**，按顺序合成一个数据集。比如原示教 + 补录 + `hil_rollout.py` 录的 HIL（先用 `hil_split_episodes.py` 切出纠正段）。同一个目录写两次就转两份，可以用来提高少量数据的占比。`--task` 给一个就是所有目录共用，也可以每个目录给一个。
- 所有目录的相机、图像尺寸、关节维度必须一致，不一致会直接报错。
- 一步得到 v3.0 数据集，`meta/stats.json` 里已经算好精确的分位数（q01/q10/q50/q90/q99）。LeRobot 自带的合并和重算统计只保留每条 episode 分位数的包络，范围偏宽，所以多份数据要一起转，不要分开转完再合并。
- `meta/source_episodes.tsv` 记录每条 episode 来自哪个目录、哪个文件。
- 关节名为 `left_waist.pos … right_gripper.pos`（左臂在前），和机器人的命名一致；动作按录制时的绝对值保存。
- 颜色：图像按 `cv2.imdecode` 的结果原样存，已和 openpi 的 `convert_aloha_data_to_lerobot_no_bgr2rgb.py` 核对过是正确的 RGB，**不要**再加 BGR→RGB 转换。
- 其他参数：`--cameras` 只保留部分相机，`--episodes` 只转部分 episode（只能配一个目录），`--use-images` 存无损 PNG（约大 40 倍），`--overwrite` 覆盖已有输出。

查看数据：

```bash
lerobot-dataset-viz --repo-id local/packaging_simple_2_v30 \
  --root /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 --episode-index 0
```

---

## 8. 训练（在服务器或云 GPU 上）

把数据集拷到训练机：

```bash
rsync -a --info=progress2 /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 server:/path/to/datasets/
```

以下命令在训练机的 lerobot 环境里运行，`--dataset.root` 换成训练机上的路径。

### 8.1 ACT（路线 A 的例子）

```bash
lerobot-train \
  --dataset.repo_id=local/aloha2_apple --dataset.root=$DATA \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_aloha2_apple --job_name=act_aloha2_apple \
  --steps=100000 --batch_size=8 --save_freq=10000 --wandb.enable=false
```

### 8.2 pi0.5（路线 B）

```bash
lerobot-train \
  --dataset.repo_id=local/packaging_simple_2_v30 --dataset.root=/path/to/packaging_simple_2_v30 \
  --dataset.image_transforms.enable=true \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=32 --steps=20000 --save_freq=5000 --num_workers=8 \
  --output_dir=outputs/pi05_packaging_simple_2_default --job_name=pi05_packaging_simple_2_default \
  --wandb.enable=false
```

batch 32 的全量微调需要大显存的卡，3090 这种 24 GB 的卡不够。

### 8.3 MolmoAct2（路线 B）

```bash
lerobot-train \
  --dataset.repo_id=local/packaging_simple_2_v30 --dataset.root=/path/to/packaging_simple_2_v30 \
  --dataset.image_transforms.enable=true \
  --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2 \
  --policy.image_keys='["observation.images.cam_high","observation.images.cam_left_wrist","observation.images.cam_right_wrist"]' \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=true \
  --policy.push_to_hub=false \
  --batch_size=8 --steps=20000 --save_freq=2000 \
  --output_dir=outputs/molmoact2_packaging --job_name=molmoact2_packaging --wandb.enable=false
```

### 8.4 训练产物和续训

- checkpoint 在 `<output_dir>/checkpoints/<步数>/pretrained_model/`（`last` 指向最新一个）。部署只需要 `pretrained_model` 这个目录，拷回机器人：
  ```bash
  rsync -a server:/path/outputs/pi05_xxx/checkpoints/020000/pretrained_model/ \
    /home/robot/Dataset_and_Checkpoint/checkpoints/pi05_xxx_020000/
  ```

  打包成 `.tar.gz` 传输的，解压前先 `gzip -t xxx.tar.gz` 确认文件完整（下载中断的包能列出部分文件，但解压会报 `Unexpected EOF`）。
- 中断后续训：
  ```bash
  lerobot-train --config_path=outputs/<job>/checkpoints/last/pretrained_model/train_config.json --resume=true
  ```

---

## 9. 部署运行

两个入口：

|          | `aloha_ros_eval.py`                 | `lerobot-rollout`       |
| -------- | ------------------------------------- | ------------------------- |
| 推理方式 | 同步（每个 chunk 执行完再推理下一个） | 同步，或 RTC              |
| 快捷键   | → / ← / ESC，按 episode 跑          | 取决于 strategy（见 9.2） |
| 录数据   | 可选（`--dataset_root`）            | episodic、dagger 必须录   |

### 9.1 `aloha_ros_eval.py`（同步，推荐日常评估）

```bash
# 路线 B 的模型（pi05、MolmoAct2）
python examples/aloha_ros/aloha_ros_eval.py --data_format aloha2 --fps 50 \
  --policy_path $CKPT --task "$TASK" --num_episodes 5 --episode_time_s 180

# 路线 A 的模型（如 ACT）
python examples/aloha_ros/aloha_ros_eval.py --fps 30 --policy_path $CKPT --num_episodes 5 --episode_time_s 30
```

按键（在运行脚本的终端里按）：

| 按键            | 作用                                                         |
| --------------- | ------------------------------------------------------------ |
| →（或 `n`）  | 在起始位等待时：开始这一轮；运行中：提前结束这一轮（算完成） |
| ←（或 `r`）  | 丢弃这一轮，回起始位重跑，不计入 `--num_episodes`          |
| ESC（或 `q`） | 停止：机械臂分两段回休息位并断电                             |

- 运行时只显示一行进度条（秒数 + 最近一次推理耗时）。
- 启动时会先做一次预热推理（pi05 约 10–20 s），之后正常推理约 0.2 s（3090）。同步推理每执行完一个 chunk 会停顿一次等推理，这是正常现象；想要连续运动用 9.3 的 RTC。
- 脚本会检查模型输入和关节顺序与 `--data_format` 是否匹配，不匹配会直接报错并提示该用哪个格式。
- 视频默认存到模型目录下（脚本顶部 `SAVE_VIDEOS`、`VIDEO_OUTPUT_DIR`）；`--dataset_root <目录>` 会把评估过程另存成 LeRobot 数据集。
- ACT 想开时间集成：把脚本顶部 `POLICY_TEMPORAL_ENSEMBLE_COEFF` 设为 0.01、`POLICY_N_ACTION_STEPS` 设为 1。
- 启动时出现 `[ERROR] Please set the gripper's 'operating mode' to 'pwm' or 'current'` 可以忽略，是 Interbotix 初始化夹爪时的提示。

### 9.2 `lerobot-rollout` 的几种模式

| `--strategy.type`               | 控制方式                                                          | 录数据           |
| --------------------------------- | ----------------------------------------------------------------- | ---------------- |
| `base`                          | 无快捷键，跑满 `--duration` 秒或 Ctrl+C 停止                    | 不录             |
| `base` + `--interactive=true` | 终端输入 `/start` 开始、`/reset` 停下回初始位、`/stop` 结束 | 不录             |
| `episodic`                      | → 提前结束本轮或重置阶段，← 丢弃重录，ESC 停止                  | 必须录           |
| `dagger`                        | 键盘或踏板，人工接管（见第 10 节）                                | 必须录，要接主臂 |

- 录数据的模式，`--dataset.repo_id` 的名字必须以 `rollout_` 开头。
- 接着往已有数据集里录加 `--resume=true`；这时相机和 fps 必须和原数据集一致。
- 路线 B 的模型一定要加 `--robot.gripper_state_format=aloha2`；路线 A 不加。
- ACT 想开时间集成：`--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1`。

只跑策略、不录数据（同步推理）：

```bash
lerobot-rollout --strategy.type=base \
  --policy.path=$CKPT \
  --robot.type=aloha_ros --robot.fps=$FPS --robot.gripper_state_format=aloha2 --robot.cameras="$CAMS" \
  --fps=$FPS --duration=180 --task="$TASK" --display_data=true
```

### 9.3 RTC 实时分块推理（选配）

**选配**：不开 RTC 也能正常运行（9.1、9.2 都是同步推理）。只有 pi05 这类推理慢的模型、又在意 chunk 之间的停顿时才需要。

论文：Kevin Black, Manuel Y. Galliker, Sergey Levine. *Real-Time Execution of Action Chunking Flow Policies*, 2025. <https://arxiv.org/abs/2506.07339>（Physical Intelligence 的介绍页：<https://www.physicalintelligence.company/research/real_time_chunking>）

RTC 在后台提前算下一个 chunk，并让新 chunk 的开头贴合旧轨迹，执行中不停顿。只能用 `lerobot-rollout`。下面是带快捷键、录数据的 episodic 版本：

```bash
lerobot-rollout --strategy.type=episodic \
  --policy.path=$CKPT \
  --inference.type=rtc --inference.rtc.execution_horizon=30 --inference.queue_threshold=30 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=aloha_ros --robot.fps=50 --robot.gripper_state_format=aloha2 --robot.cameras="$CAMS" \
  --dataset.repo_id=local/rollout_pi05_eval \
  --dataset.root=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_pi05_eval \
  --dataset.num_episodes=5 --dataset.episode_time_s=180 --dataset.reset_time_s=30 \
  --dataset.push_to_hub=false --dataset.single_task="$TASK" \
  --fps=50 --task="$TASK"
```

不想录数据：`--strategy.type=base`，去掉所有 `--dataset.*`，加 `--duration=180`。

参数怎么定（50 Hz、pi05、3090 实测）：

- guided RTC 每个去噪步多一次反向传播，推理约 240–290 ms（普通推理 200 ms），加上取图和预处理，延迟约 15 个 tick。
- `execution_horizon` 必须明显大于这个延迟。设成默认的 10 时，新旧 chunk 的切换点落在引导范围之外，每次切换都会跳一下，反而比同步推理更难受。
- `queue_threshold=30`：队列剩 30 步时开始算下一个 chunk，算完约执行掉 15 步，切换点落在 15–30 步的过渡区里。GPU 同时跑别的程序、延迟变大时，两个值一起调到 35。
- 训练时没开 RTC 的 checkpoint（`rtc_training_max_delay=0`，目前的模型都是）只能用默认的 guided 模式。

---

## 10. HIL 人工接管补数据（DAgger）

策略自动跑，出错时人用主臂接管纠正，把纠正的片段加进训练数据再训练。有两种方式，**两种都能用，也可以混在一起用**：

| | 方式 A：`lerobot-rollout` dagger | 方式 B：act-triple 的 `hil_rollout.py` |
|---|---|---|
| 被改进的模型 | LeRobot 训练的 checkpoint（pi05、MolmoAct2、ACT），在本机 GPU 上推理 | openpi 或 N0-VTLA 的模型，在服务器上用 `serve_policy.py` 推理 |
| 运行环境 | 本仓库的 venv | act-triple 的 conda 环境 `aloha` |
| 录什么 | 只录纠正段，每次纠正存成一条 episode | 整条 episode 都录，每帧带 phase / intervention 标签，录完再切 |
| 数据格式 | LeRobot v3.0 数据集（`rollout_*`） | ALOHA2 原始 hdf5（N0 是 VTLA 目录） |
| 能用来训练 | 只能用于 LeRobot | LeRobot、ACT（act-triple）、openpi 都能用 |
| 并进 LeRobot 训练集 | `merge_hil_into_lerobot_v30.py`（10.1） | `hil_split_episodes.py` 切出纠正段，再用 `convert_aloha_hdf5_to_lerobot_v30.py` 和示教一起转（10.2） |

怎么选：要改进的是 LeRobot 训练的模型，用方式 A；要改进的是 openpi / N0 的模型，用方式 B。DAgger 的纠正针对的是当时跑的那个模型的失败，用来训练同一个模型效果最直接；拿去训练别的模型也算有效的示教数据，但纠正的不是那个模型自己的失败。方式 B 的数据是原始格式，以后给哪个框架用都行；方式 A 的数据只能用于 LeRobot，没有工具把它转回 hdf5。

两种方式的接管都是离合式，键位相同：→ 暂停 / 交还策略，← 开始 / 结束纠正。

### 10.1 方式 A：`lerobot-rollout` dagger（LeRobot 模型）

只录纠正段，每次纠正存成一条 episode，带 `intervention` 标记（纠正模式下全是 True）。

接管有两个版本，区别只在「开始纠正」那一刻主臂和从臂怎么交接：

| | 版本一：修改前（非离合，从臂缓入） | 版本二：修改后（离合式） |
|---|---|---|
| 主臂释放 | 0.6 s 内变软后断扭矩 | 0.3 s 内变软后断扭矩，再等主臂稳定（最多 1 s） |
| 从臂怎么跟 | 0.3 s 内从保持的姿态缓入到主臂的**绝对**位置 | 记下主臂和从臂的差值，按「主臂 − 差值」跟随 |
| 主臂松手下沉时 | 从臂跟着沉到主臂的位置，会动一下 | 差值把下沉抵消掉，从臂不动 |
| 不录的开头 | 0.5 s | 0.2 s |
| 关键参数 | `correction_blend_s=0.3`、`correction_skip_s=0.5`、`release_ramp_s=0.6` | `relative_on_release=true`、`correction_blend_s=0`、`correction_skip_s=0.2`、`release_ramp_s=0.3` |

两个版本录出来的数据格式完全一样，可以录进同一个数据集；想对比两种接管的效果，就用不同的 `--dataset.root`。接着录已有的 HIL 数据集时加 `--resume=true`。

#### 版本一：修改前（非离合，从臂缓入）

```bash
lerobot-rollout --strategy.type=dagger --strategy.record_autonomous=false \
  --strategy.keyboard.pause_resume=right --strategy.keyboard.correction=left \
  --strategy.release_teleop_on_resume=false --strategy.num_episodes=20 \
  --strategy.correction_blend_s=0.3 --strategy.correction_skip_s=0.5 \
  --robot.type=aloha_ros --robot.fps=$FPS --robot.gripper_state_format=aloha2 --robot.cameras="$CAMS" \
  --teleop.type=aloha_teleop_ros --teleop.init_ros_node=false --teleop.fps=$FPS \
  --teleop.release_on_connect=false --teleop.release_ramp_s=0.6 \
  --policy.path=$CKPT \
  --fps=$FPS --dataset.repo_id=local/rollout_packaging_hil --dataset.no_stamp=true \
  --dataset.root=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --dataset.fps=$FPS --dataset.single_task="$TASK" --dataset.push_to_hub=false \
  --display_data=true
```

#### 版本二：修改后（离合式）

```bash
lerobot-rollout --strategy.type=dagger --strategy.record_autonomous=false \
  --strategy.keyboard.pause_resume=right --strategy.keyboard.correction=left \
  --strategy.release_teleop_on_resume=false --strategy.num_episodes=20 \
  --strategy.correction_blend_s=0 --strategy.correction_skip_s=0.2 \
  --robot.type=aloha_ros --robot.fps=$FPS --robot.gripper_state_format=aloha2 --robot.cameras="$CAMS" \
  --teleop.type=aloha_teleop_ros --teleop.init_ros_node=false --teleop.fps=$FPS \
  --teleop.release_on_connect=false --teleop.relative_on_release=true --teleop.release_ramp_s=0.3 \
  --policy.path=$CKPT \
  --fps=$FPS --dataset.repo_id=local/rollout_packaging_hil --dataset.no_stamp=true \
  --dataset.root=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --dataset.fps=$FPS --dataset.single_task="$TASK" --dataset.push_to_hub=false \
  --display_data=true
```

#### 操作流程和参数说明

两个版本的按键相同：

1. 启动后策略自动运行，主臂保持锁定不动。
2. 看到要出错时按 **→ 暂停**：从臂停住，主臂被带到从臂当前的姿态。
3. 手握住主臂，按 **← 开始纠正**：主臂逐渐变软后释放，按上表的方式交接，然后开始录制。
4. 纠正完按 **← 结束纠正**：这一段存成一条 episode。
5. 按 **→ 交还策略**，继续自动运行。ESC 结束整个会话。

| 参数 | 作用 |
|---|---|
| `--teleop.relative_on_release` | `true` = 离合式（版本二）：主臂释放、等它稳定下来后，记下主臂和从臂的差值，从臂按「主臂 − 差值」跟随。不加（默认 `false`）= 版本一，从臂直接跟主臂的绝对位置 |
| `--teleop.release_ramp_s` | 主臂在这段时间内把位置增益降到 0 再断扭矩，避免一下子松掉 |
| `--strategy.correction_blend_s` | 开始纠正时从臂缓入到主臂位置的时间。版本一靠它避免从臂一下跳过去；离合模式下从臂本来就不动，设 0 |
| `--strategy.correction_skip_s` | 每次纠正的开头这么多秒不录，避开交接时的过渡动作。纠正时间短于它的不保存、不计数 |
| `--strategy.release_teleop_on_resume=false` | 交还策略后主臂保持锁定，不会因为断扭矩而掉下来 |
| `--strategy.num_episodes` | 录满这么多次纠正后自动结束 |

#### 录完之后：数据处理和训练

dagger 只往自己的数据集（上面的 `rollout_packaging_hil`）里写：每结束一次纠正，就存成一条 episode，同时更新这个数据集自己的 `meta/`。**它不会动你训练用的数据集。** 想用这些数据，要先合并成一个新数据集，再训练。

**第 1 步：合并。** 用 act-triple 的 `aloha_scripts/merge_hil_into_lerobot_v30.py`（用本仓库的 venv 运行）：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/merge_hil_into_lerobot_v30.py \
  --base /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30 \
  --hil /home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_v30_hil
```

- 输出是新数据集，`--base` 和 `--hil` 都不会被修改。`--hil` 可以给多个；同一个写两次就加两份，可以提高 HIL 的占比。结束时会打印 HIL 帧占总帧数的比例。
- 两种数据集的特征不同，脚本会先对齐再合并：
  - 去掉 HIL 里的 `intervention` 和 `cam_low`；哪些 episode 来自 HIL 记在 `meta/source_episodes.tsv` 里。
  - 去掉 base 里的 `observation.velocity`、`observation.effort`。dagger 不录这两项，pi05、MolmoAct2、ACT 也都只用 `observation.state`。
  - `robot_type` 统一成 base 的。
- 视频直接拼接，不重新编码；`meta/stats.json` 按合并后的全部数据重新算精确分位数。
- fps、关节名顺序、图像尺寸必须一致，否则报错。HIL 必须是加了 `--robot.gripper_state_format=aloha2` 录的（上面两条命令都有），这一点脚本检查不了。

**第 2 步：训练。** 两种方式：

- 从头训：把第 8 节命令里的数据集换成合并后的数据集。
- 接着旧模型练：
  ```bash
  lerobot-train --policy.path=$CKPT \
    --dataset.repo_id=local/packaging_simple_2_v30_hil --dataset.root=/path/to/packaging_simple_2_v30_hil \
    --dataset.image_transforms.enable=true \
    --batch_size=32 --steps=5000 --save_freq=1000 \
    --output_dir=outputs/pi05_packaging_hil --job_name=pi05_packaging_hil --wandb.enable=false
  ```
  `--policy.path` 从旧模型的权重开始，归一化统计自动换成新数据集的。这里不要用 `--resume=true`，它只能在原来那次训练、原来的数据集上续训。旧模型的输入里有 velocity/effort，新数据集里没有，照样能训，已经用 ACT 实测过这个流程。

### 10.2 方式 B：act-triple 的 `hil_rollout.py`（openpi / N0 模型）

**采集**（服务器先起好 `serve_policy.py`；按键、进度条和参数见 act-triple 的 `aloha_scripts/README_run_robot.md` 第 1、3 节）：

```bash
conda activate aloha
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python3 aloha_scripts/hil_rollout.py --task_name packaging_simple_2 --policy openpi \
  --host 10.240.77.20 --port 8000 --prompt "packaging the items" --num_episodes 20
```

数据存到 `/home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil/episode_N.hdf5`，格式和示教的 hdf5 相同，另外多了每帧的 HIL 标签。

**第 1 步：切出纠正段**（还在 `aloha` 环境里）。整条 episode 里约三成是暂停，不能直接拿去训练：

```bash
python3 aloha_scripts/hil_split_episodes.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections --keep intervention
```

**第 2 步（可选）：量一下占比。** 打印纠正段占原示教的比例，以及要达到 10% / 20% 需要复制几份：

```bash
python3 aloha_scripts/hil_compare_lengths.py \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
  /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil
```

**第 3 步：和示教一起转成 LeRobot v3.0**（换到本仓库的 venv）。纠正段占比太低时，把 `_hil_corrections` 多写几次，每写一次多一份：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py \
  --raw-dir /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
            /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_v30 \
  --task "packaging the items" --fps 50
```

全部数据都来自 hdf5，所以这份数据集保留 velocity/effort，和第 7 节转出的数据集特征完全相同。

**第 4 步：训练。** 和 10.1 第 2 步一样，从头训或用 `--policy.path` 接着旧模型练，数据集换成 `packaging_simple_2_hil_v30`。

同一批切好的纠正段也能直接给 act-triple 的 ACT（读 hdf5）和 openpi（用 openpi 的转换脚本转成 v2.1）训练，做法见 act-triple 的 `aloha_scripts/README_run_robot.md` 第 4 节。

### 10.3 两种一起用

可以。先用转换脚本把示教和方式 B 的纠正段一起转成 v3.0，再用合并脚本把方式 A 的数据集并上去：

```bash
source /mnt/server_ssd2/Programs_server/lerobot/.venv/bin/activate
cd /mnt/server_ssd2/Programs_server/act-triple-plus-aloha2
python aloha_scripts/convert_aloha_hdf5_to_lerobot_v30.py \
  --raw-dir /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2 \
            /home/robot/Dataset_and_Checkpoint/dataset/packaging_simple_2_hil_corrections \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_v30 \
  --task "packaging the items" --fps 50
python aloha_scripts/merge_hil_into_lerobot_v30.py \
  --base /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_v30 \
  --hil /home/robot/Dataset_and_Checkpoint/lerobot-dataset/rollout_packaging_hil \
  --output-dir /home/robot/Dataset_and_Checkpoint/lerobot-dataset/packaging_simple_2_hil_all_v30
```

- 顺序不能反：hdf5 只能走转换脚本，LeRobot 数据集只能走合并脚本。
- 两种数据能放在一起，是因为关节顺序（左臂在前）、夹爪状态的含义（ALOHA2 qpos）、50 Hz 都一致，两个仓库的夹爪标定常量也完全相同。
- 只要并进了方式 A 的数据，结果里就没有 velocity/effort。
- 反过来不行：方式 A 的数据没法用于 act-triple 的 ACT 或 openpi。

---

## 11. 休眠 / 断电

```bash
python examples/aloha_ros/aloha_scripts/sleep_modify.py --sleep             # 四条臂回休息位，保持上电（中途休息）
python examples/aloha_ros/aloha_scripts/sleep_modify.py --shut_down         # 四条臂回休息位后断电（收工）
python examples/aloha_ros/aloha_scripts/sleep_modify.py --shut_down_puppet  # 只让从臂回休息位并断电
```

回休息位分两段走（先到旧的休息位，再到实测的最终位置），然后才断扭矩，避免机械臂掉下来砸到。

---

## 12. 常见问题

| 现象                                                 | 原因和处理                                                                                |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `Couldn't parse '' into an int`                    | `$FPS`、`$CAMS` 等变量在当前终端是空的，按 2.4 重新设置                               |
| `No module named 'rospkg'`                         | `uv sync` 删掉了 ROS 依赖，重新执行 `uv pip install rospkg catkin_pkg netifaces h5py` |
| 夹爪动作异常（该张不张、该合不合）                   | 数据路线和部署参数不一致，见第 0 节；或者标定变了                                         |
| 同步推理每秒顿一下                                   | 正常：每个 chunk 执行完要等一次推理。要连续运动可选用 RTC                                    |
| RTC 周期性跳动                                       | `execution_horizon` 太小，见 9.3                                                        |
| `--resume=true` 接着录失败                         | 相机或 fps 和已有数据集不一致，`CAMS` 要写全数据集里的相机                              |
| `.tar.gz` 解压报 `Unexpected EOF`                | 下载不完整，重新下载，先 `gzip -t` 校验                                                 |
| 启动时 `Please set the gripper's 'operating mode'` | Interbotix 的提示，可以忽略                                                               |
