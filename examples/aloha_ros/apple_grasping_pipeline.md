# Apple Grasping 完整流程指南

本文档详细说明从相机测试、遥操作采集数据、数据可视化、训练 ACT 模型到自动化部署推理的完整流程。

---

## 硬件与软件环境

| 组件 | 说明 |
|------|------|
| 机械臂 | 双手 ALOHA (VX300S)，通过 ROS InterbotixManipulatorXS 控制 |
| 相机 | 4 台 Intel RealSense (640x480, 30FPS) |
| 控制器 | 主手机械臂遥操作 |
| 策略 | ACT (Action Chunking Transformer) |

**相机配置：**

| 相机名称 | 序列号 | 位置 |
|----------|--------|------|
| cam_high | 109422062625 | 顶部视角 |
| cam_right_wrist | 943222070893 | 右手腕 |
| cam_left_wrist | 134222077139 | 左手腕 |
| cam_low | 936322072119 | 底部视角 |

---

## Step 0: 相机测试

在采集数据前，先确认所有相机正常连接并能捕获图像。

```bash
# 检测所有 RealSense 相机
lerobot-find-cameras realsense

# 检测所有 OpenCV 可识别的相机，图像保存到 outputs/captured_images
lerobot-find-cameras opencv
```

**检查要点：**
- 确认 4 台 RealSense 相机全部被识别
- 确认每台相机的序列号与上表一致
- 检查 `outputs/captured_images/` 下的图像质量是否正常

---

## Step 1: 遥操作采集数据

通过 ROS 启动机械臂，使用主手遥操作控制从手执行 "抓取苹果" 任务，同时录制视频和关节数据。

### 1.1 启动遥操作（单臂测试）

```bash
python examples/aloha_ros/aloha_ros_teleop_one_arm.py
```

### 1.2 启动遥操作（双臂完整版）

```bash
python examples/aloha_ros/aloha_ros_teleop.py
```

### 1.3 机械臂休眠 / 关机

```bash
# 让机械臂回到休眠位置（安全锁定）
python examples/aloha_ros/aloha_scripts/sleep.py --sleep

# 关闭机械臂扭矩（彻底下电）
python examples/aloha_ros/aloha_scripts/sleep.py --shut_down
```

### 1.4 录制演示视频

```bash
python examples/aloha_ros/aloha_ros_record.py
```

该脚本的配置（在 `aloha_ros_record.py` 中定义）：
- `NUM_EPISODES = 50` — 共采集 50 个 episode
- `EPISODE_TIME_SEC = 15` — 每个 episode 录制 15 秒
- `TASK_DESCRIPTION = "Grasp an apple"` — 任务描述
- 数据保存到 `/home/robot/Dataset_and_Checkpoint/lerobot-dataset/apple_grasping`

---

## Step 2: 训练 ACT 模型

使用采集好的数据集训练 ACT 策略模型。

```bash
lerobot-train \
  --dataset.repo_id=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/apple_grasping \
  --policy.type=act \
  --output_dir=/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint/apple_grasping \
  --job_name=act_apple_grasping \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=theodoreliu/act_policy
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--dataset.repo_id` | 数据集路径（本地目录或 HuggingFace Hub ID） |
| `--policy.type=act` | 使用 ACT (Action Chunking Transformer) 策略 |
| `--output_dir` | 模型 checkpoint 保存目录 |
| `--job_name` | 训练任务名称，用于 wandb 日志和 checkpoint 命名 |
| `--policy.device=cuda` | 使用 GPU 训练 |
| `--wandb.enable=false` | 关闭 wandb 日志（需要时设为 true） |
| `--policy.repo_id` | HuggingFace Hub 上传路径（`theodoreliu/act_policy`） |

**产出物：**
- 本地 checkpoint：`/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint/apple_grasping/checkpoints/last/pretrained_model`
- 远程模型（如果 push_to_hub=true）：`https://huggingface.co/theodoreliu/act_policy`

---

## 数据可视化

在训练前后，可以使用 lerobot 内置工具查看数据集：

```bash
# 数据集可视化
lerobot-dataset-viz --dataset.repo_id=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/apple_grasping

# 查看数据集信息
lerobot-info --dataset.repo_id=theodoreliu/eval_act_apple_grasping

# 数据集编辑（如需删除错误 episode）
lerobot-edit-dataset --dataset.repo_id=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/apple_grasping
```

---

## Step 3: 自动化部署推理

使用训练好的 ACT 策略，自动执行 "抓取苹果" 任务并录制结果。

### 方案 A：上传到 HuggingFace Hub

模型和数据集都上传到 HuggingFace Hub，适合分享和远程部署：

```bash
lerobot-record \
  --robot.type=aloha_ros \
  --robot.puppet_left_robot_name=puppet_left \
  --robot.puppet_right_robot_name=puppet_right \
  --robot.robot_model=vx300s \
  --robot.init_ros_node=true \
  --robot.cameras='{
    cam_high:        {type: intelrealsense, serial_number_or_name: "109422062625", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: "943222070893", width: 640, height: 480, fps: 30},
    cam_left_wrist:  {type: intelrealsense, serial_number_or_name: "134222077139", width: 640, height: 480, fps: 30},
    cam_low:         {type: intelrealsense, serial_number_or_name: "936322072119", width: 640, height: 480, fps: 30}
  }' \
  --display_data=true \
  --dataset.repo_id=theodoreliu/eval_act_apple_grasping \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=15 \
  --dataset.single_task="apple_grasping" \
  --policy.path=theodoreliu/act_policy
```

### 方案 B：本地部署

模型和数据集都保存在本地，不上传到 Hub：

```bash
lerobot-record \
  --robot.type=aloha_ros \
  --robot.puppet_left_robot_name=puppet_left \
  --robot.puppet_right_robot_name=puppet_right \
  --robot.robot_model=vx300s \
  --robot.init_ros_node=true \
  --robot.cameras='{
    cam_high:        {type: intelrealsense, serial_number_or_name: "109422062625", width: 640, height: 480, fps: 30},
    cam_right_wrist: {type: intelrealsense, serial_number_or_name: "943222070893", width: 640, height: 480, fps: 30},
    cam_left_wrist:  {type: intelrealsense, serial_number_or_name: "134222077139", width: 640, height: 480, fps: 30},
    cam_low:         {type: intelrealsense, serial_number_or_name: "936322072119", width: 640, height: 480, fps: 30}
  }' \
  --display_data=true \
  --dataset.repo_id=theodoreliu/eval_act_apple_grasping \
  --dataset.root=/home/robot/Dataset_and_Checkpoint/lerobot-dataset/eval_apple_grasping \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=15 \
  --dataset.single_task="apple_grasping" \
  --policy.path=/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint/apple_grasping/checkpoints/last/pretrained_model
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--robot.type=aloha_ros` | 机器人类型为基于 ROS 的 ALOHA 双臂 |
| `--robot.puppet_*` | 从手（执行端）ROS topic 名称 |
| `--robot.init_ros_node=true` | 自动初始化 ROS 节点 |
| `--robot.cameras` | 4 台 RealSense 相机的序列号与参数 |
| `--display_data=true` | 实时显示相机画面 |
| `--dataset.repo_id` | 评估数据集的 HuggingFace ID |
| `--dataset.root` | 本地数据保存路径（方案 B 使用） |
| `--dataset.push_to_hub=false` | 不上传到 Hub（方案 B） |
| `--dataset.num_episodes=10` | 执行 10 个推理 episode |
| `--dataset.episode_time_s=15` | 每个 episode 最长 15 秒 |
| `--dataset.single_task` | 任务描述标签 |
| `--policy.path` | 模型路径（Hub ID 或本地路径） |

---

## 整体流程图

```
┌─────────────────────────────────────────────────────────┐
│  Step 0: 相机测试                                        │
│  lerobot-find-cameras realsense / opencv                 │
│  → 确认 4 台 RealSense 相机正常工作                       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: 遥操作采集数据                                   │
│  aloha_ros_teleop.py → 主手控制从手执行抓取苹果任务         │
│  aloha_ros_record.py → 录制 50 个 episode 的演示数据       │
│  sleep.py → 任务结束后的休眠/关机                   │
│  → 产出：Dataset (图像 + 关节状态 + 动作)                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  数据可视化 (可选)                                        │
│  lerobot-dataset-viz → 查看/检查采集的数据集质量           │
│  lerobot-info → 查看数据集统计信息                        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: 训练 ACT 模型                                   │
│  lerobot-train --policy.type=act ...                     │
│  → 使用 GPU 训练，产出 checkpoint                        │
│  → 自动上传到 HuggingFace Hub (theodoreliu/act_policy)    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: 自动化部署推理                                   │
│  lerobot-record ... --policy.path=<模型路径>               │
│  → 加载训练好的策略，自动执行抓取苹果任务                   │
│  → 方案 A: 云端模型 + 云端数据集                          │
│  → 方案 B: 本地模型 + 本地数据集                          │
└─────────────────────────────────────────────────────────┘
```

---

## 关键路径汇总

| 用途 | 路径 |
|------|------|
| 训练数据集 | `/home/robot/Dataset_and_Checkpoint/lerobot-dataset/apple_grasping` |
| 模型 checkpoint | `/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint/apple_grasping` |
| 评估数据集（本地） | `/home/robot/Dataset_and_Checkpoint/lerobot-dataset/eval_apple_grasping` |
| HuggingFace 模型 | `theodoreliu/act_policy` |
| HuggingFace 数据集 | `theodoreliu/eval_act_apple_grasping` |
| 评估视频 | `/home/robot/videos/deploy_third_person/lerobot/act/act_apple_grasping/` |

---

## 常用调试命令

```bash
# 查找可用的 RealSense 相机序列号
lerobot-find-cameras realsense

# 查看已录制数据集的统计信息
lerobot-info --dataset.repo_id=/path/to/dataset

# 可视化检查数据集
lerobot-dataset-viz --dataset.repo_id=/path/to/dataset

# 只使用单个机械臂测试
python examples/aloha_ros/aloha_ros_teleop_one_arm.py
```
