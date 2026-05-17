# ALOHA2 + XENSE Tactile — Full Setup Summary

> 最后更新: 2026-05-10
> 当前分支: aloha
> 状态: 录制 pipeline ✅ | 训练 pipeline ✅ (smoke test 通过) | 仅 1 episode 数据，需录更多

---

## 硬件清单


| 设备            | 数量  | 型号 / 标识                                |
| ------------- | --- | -------------------------------------- |
| 主臂 (Leader)   | 2   | WidowX 250 (master_left, master_right) |
| 从臂 (Follower) | 2   | ViperX-300 (puppet_left, puppet_right) |
| 相机            | 4   | Intel RealSense D415/D435/D435i        |
| 触觉传感器         | 4   | XENSE G1-WS 光子触觉                       |


### 相机串号


| 名称              | 串号                   | 后端                             |
| --------------- | -------------------- | ------------------------------ |
| cam_high        | 109422062625 (D415)  | RealSense                      |
| cam_low         | 936322072119 (D435)  | RealSense                      |
| cam_left_wrist  | 134222077139 (D435i) | RealSense                      |
| cam_right_wrist | 943222070893 (D435i) | OpenCV (librealsense v4l2 bug) |


### 触觉传感器


| 位置              | 串号       | 模式          |
| --------------- | -------- | ----------- |
| left_fingertip  | OG000635 | FULL (17ch) |
| left_knuckle    | OG000707 | FULL (17ch) |
| right_fingertip | OG000614 | FULL (17ch) |
| right_knuckle   | OG000706 | FULL (17ch) |


### 夹爪实测校准值 (2026-05-08)


| 夹爪  | 最小值 (闭) | 最大值 (开) | 范围     |
| --- | ------- | ------- | ------ |
| 左主臂 | -0.2393 | 0.6796  | 0.9189 |
| 右主臂 | -0.0537 | 0.8468  | 0.9004 |
| 左从臂 | -1.7871 | -0.2562 | 1.5309 |
| 右从臂 | -1.6168 | -0.0430 | 1.5739 |


---

## 软件环境


| 项目        | 值                                           |
| --------- | ------------------------------------------- |
| Python 环境 | `conda activate lerobot` (Python 3.10.19)   |
| ROS       | Noetic                                      |
| 框架        | LeRobot 0.4.3 (`pip install -e .` editable) |
| XENSE SDK | xensesdk 1.7.0                              |
| 触觉模式      | **FULL (17 通道 float32)**                    |
| GPU       | NVIDIA RTX 3090, CUDA 12.8                  |


### FULL 模式通道详情 (17 channels)

```
rectify(1) + difference(1) + depth(1) + force(3) + force_norm(3) + force_resultant(6) + marker2d(2) = 17
索引:  rectify=[0], difference=[1], depth=[2], force=[3,4,5], force_norm=[6,7,8],
       force_resultant=[9..14], marker2d=[15,16]
```

### 账户


| 服务          | 用户名         |
| ----------- | ----------- |
| GitHub      | theotaoliu  |
| HuggingFace | theodoreliu |
| W&B         | theoliu     |


---

## 端口映射


| 设备           | 端口                       |
| ------------ | ------------------------ |
| master_left  | /dev/ttyDXL_master_left  |
| master_right | /dev/ttyDXL_master_right |
| puppet_left  | /dev/ttyDXL_puppet_left  |
| puppet_right | /dev/ttyDXL_puppet_right |


---

## 关键脚本与用法

### 1. 录制数据 (4 相机 + 4 触觉 FULL 17ch)

```bash
# 终端 1: ROS
roslaunch aloha aloha_test.launch \
  master_left/xs_sdk/load_configs:=false \
  master_right/xs_sdk/load_configs:=false \
  puppet_left/xs_sdk/load_configs:=false \
  puppet_right/xs_sdk/load_configs:=false

# 杀 usb_cam 节点（录制用 RealSense 直连）
rosnode kill /usb_cam_high /usb_cam_low /usb_cam_left_wrist /usb_cam_right_wrist
rosnode list | grep usb_cam # 确认相机设备已经释放
fuser -k /dev/video* 2>/dev/null

# 终端 2: 正式任务试采（先 5 episodes，不要直接录满 50）
conda activate lerobot
python examples/aloha_ros/aloha_ros_record.py \
  --task_name aloha_flip_switch \
  --episodes 5

# 试采后立刻质检：相机视频、parquet、触觉 shape=(17,120,160)、episode 数和时长
python examples/aloha2/check_collection_batch.py \
  --task aloha_flip_switch \
  --min_episodes 5

# 质检通过后，接着已有数据补录到 20 episodes
python examples/aloha_ros/aloha_ros_record.py \
  --task_name aloha_flip_switch \
  --episode_idx 5 \
  --episodes 15

# 后续核心任务同理替换 task_name:
#   aloha_hidden_property_grasp
#   aloha_peg_insertion
#   aloha_slip_hold_or_pull

# 结束后归位
python examples/aloha_ros/aloha_scripts/sleep.py --sleep        # 所有臂归位，保持扭矩
python examples/aloha_ros/aloha_scripts/sleep.py --sleep --arms=master  # 仅主臂
python examples/aloha_ros/aloha_scripts/sleep.py --shut_down --arms=puppet  # 仅从臂卸力
python examples/aloha_ros/aloha_scripts/sleep.py --shut_down     # 所有臂归位 + 卸力
```

### 2. 训练 ACT 模型 (灵活触觉通道消融)

```bash
# visual_only baseline
python examples/aloha2/train_aloha2_tactile.py \
  --dataset_repo_id theodoreliu/aloha_grasp_apple \
  --tactile_channels visual_only \
  --steps 100000 --batch_size 8

# GelSight-equivalent (仅 rectify)
python examples/aloha2/train_aloha2_tactile.py \
  --dataset_repo_id theodoreliu/aloha_grasp_apple \
  --tactile_channels rectify \
  --steps 100000 --batch_size 8

# 全维度
python examples/aloha2/train_aloha2_tactile.py \
  --dataset_repo_id theodoreliu/aloha_grasp_apple \
  --tactile_channels all \
  --steps 100000 --batch_size 8

# 自定义组合
python examples/aloha2/train_aloha2_tactile.py \
  --dataset_repo_id theodoreliu/aloha_grasp_apple \
  --tactile_channels force,depth,marker2d \
  --steps 100000 --batch_size 8
```

### 3. 触觉数据可视化

```bash
python examples/aloha2/extract_tactile_viz.py \
  --task aloha_flip_switch \
  --frames 5
# 输出: /mnt/server/Programs_server/metalerobot/outputs/tactile_viz/
```

### 4. 遥操作 (无录制)

```bash
python examples/aloha_ros/aloha_ros_teleop.py
```

### 5. 夹爪测试

```bash
python examples/aloha2/test_gripper_range_simple.py master_left
python examples/aloha2/test_gripper_range_simple.py puppet_left --release
```

---

## 数据路径


| 用途          | 路径                                                                            |
| ----------- | ----------------------------------------------------------------------------- |
| 数据集         | `/home/robot/Dataset_and_Checkpoint/lerobot-dataset/theodoreliu/{task_name}/` |
| Checkpoints | `/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint/{task_name}/`          |
| 触觉可视化       | `/mnt/server/Programs_server/metalerobot/outputs/tactile_viz/`                |


### 数据集内部结构

```
{task_name}/
├── data/chunk-000/*.parquet    # 动作 + 状态 + 触觉 float32 (17,120,160)
├── videos/observation.images.*/chunk-000/*.mp4  # 4 路相机
├── meta/info.json, stats.json, tasks.parquet, episodes/
└── images/                     # 临时帧缓存
```

---

## 任务配置

在 `examples/aloha_ros/aloha_scripts/constants.py` → `TASK_CONFIGS` 中：


| 任务                          | 层级        | Episodes | 核心触觉                             | 状态   |
| --------------------------- | --------- | -------- | -------------------------------- | ---- |
| aloha_flip_switch           | L1 视觉主导   | 50       | Rectify                          | ⬜    |
| aloha_hidden_property_grasp | L2 力控关键   | 50       | ForceResultant, Marker2D         | ⬜    |
| aloha_slip_hold_or_pull     | L2/3 可控滑移 | 50       | Marker2D, Force                  | ⬜    |
| aloha_peg_insertion         | L3 精细触觉   | 50       | Marker2D, Force                  | ⬜    |
| aloha_pressure_wipe         | L2/3 连续力控 | 50       | ForceNorm, ForceResultant, Depth | 第二轮  |
| aloha_towel_unfold          | L4 形变操作   | 50       | Marker2D, Depth, Force           | 扩展展示 |
| aloha_grasp_apple           | 测试        | 85       | —                                | 测试用  |


加新任务：在 `TASK_CONFIGS` 加一条，`--task_name` 即可使用。

### 正式采集顺序

第一轮只做 4 个核心任务，先小批量试采：

1. `aloha_flip_switch`：负对照，预期 `visual_only ≈ full`
2. `aloha_hidden_property_grasp`：核心触觉必要性，预期 `force_all/full` 明显优于 `visual_only/rectify`
3. `aloha_slip_hold_or_pull`：可控滑移/夹持稳定，预期 `marker/full` 优于 `rectify`
4. `aloha_peg_insertion`：精细插入，预期 `marker+force/full` 有机制优势

第二轮时间允许再补充 `aloha_pressure_wipe`；`aloha_towel_unfold` 保留为 journal scenario 或展示任务。

每个任务先录 5 episodes，运行 `examples/aloha2/check_collection_batch.py` 通过后再补到 20 episodes，最后补到 50 episodes。

### 试采验收标准

- 4 路相机视频存在：`cam_high` / `cam_low` / `cam_left_wrist` / `cam_right_wrist`
- 4 个 XENSE 触觉列存在：`left_fingertip` / `left_knuckle` / `right_fingertip` / `right_knuckle`
- 每个采样触觉帧 shape 为 `(17, 120, 160)`
- parquet 能正常读取，episode 数达到试采数量
- 任务时长符合配置：flip 300 frames，hidden/slip/pressure 450 frames，peg/towel 600 frames
- 可视化 montage 能看到接触变化

---

## 完整改动文件清单 (19 个文件)

### 核心库改动


| 文件                                                    | 改动                                                                                                                          |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `src/lerobot/datasets/utils.py`                       | 3 处: `hw_to_dataset_features` 触觉→float32; `build_dataset_frame` 支持 `.tactiles.` 前缀; `dataset_to_policy_features` 识别 TACTILE |
| `src/lerobot/robots/aloha_ros/config_aloha_ros.py`    | 新增 `tactile_sensors` 字段                                                                                                     |
| `src/lerobot/robots/aloha_ros/aloha_ros.py`           | 触觉 connect/read/disconnect; 移除 transpose; `_tactile_ft` (C,H,W); 删 numpy import                                             |
| `src/lerobot/robots/aloha/aloha.py`                   | 同上                                                                                                                          |
| `src/lerobot/tactile/xense_g1ws/config_xense_g1ws.py` | output_types 新增 difference+force_resultant; 修正 force_resultant 通道数 1→6                                                      |
| `src/lerobot/tactile/xense_g1ws/xense_g1ws.py`        | SIMPLE→3ch; `_mock_frame` →3ch; `_real_simple_frame`→3ch                                                                    |
| `src/lerobot/policies/act/configuration_act.py`       | 修复 import; 新增 `tactile_channel_indices` 与 `use_tactile` 字段                                                                  |
| `src/lerobot/policies/act/modeling_act.py`            | 通道切片 forward; init 适配 channel_indices                                                                                       |
| `src/lerobot/optim/__init__.py`                       | 补 LRSchedulerConfig 导出                                                                                                      |
| `src/lerobot/configs/policies.py`                     | 修复 import 路径                                                                                                                |
| `src/lerobot/configs/__init__.py`                     | 新建，导出 FeatureType/NormalizationMode/PolicyFeature                                                                           |


### 脚本改动


| 文件                                              | 改动                                                                           |
| ----------------------------------------------- | ---------------------------------------------------------------------------- |
| `examples/aloha_ros/aloha_ros_record.py`        | FULL 模式; `--task_name`/`--episodes`/`--duration`/`--episode_idx` CLI; 实测夹爪校准 |
| `examples/aloha_ros/aloha_ros_teleop.py`        | 实测夹爪校准; 移除相机依赖; 修复 close_thresh                                              |
| `examples/aloha_ros/aloha_scripts/constants.py` | 新增 aloha_wear_shoe + aloha_grasp_apple 任务                                    |
| `examples/aloha2/train_aloha2_tactile.py`       | 灵活触觉通道消融 CLI                                                                 |
| `examples/aloha2/extract_tactile_viz.py`        | 新建，触觉帧提取可视化                                                                  |
| `examples/aloha2/test_gripper_range_simple.py`  | 新建，纯 ROS 夹爪测试                                                                |
| `examples/aloha2/record_aloha2_ros.py`          | 修正项目路径 lerobot→metalerobot                                                   |
| `examples/aloha_ros/aloha_scripts/sleep.py`     | 新增 `--arms` 支持 (master/puppet/all)；删除 `sleep.py`（直接 Dynamixel 与 ROS 冲突）      |


---

## 已知问题

1. **Ctrl+Z 中断会损坏 parquet** — 用 `→` (右箭头) 或 ESC 正常结束 episode
2. **cam_right_wrist** — librealsense v4l2 bug，已用 OpenCV 后端
3. **左主臂 wrist_angle** — 校准偏移，单独配置为 -0.4
4. **pynput / rerun** — 无 GUI 环境报错，不影响录制和训练
5. `**listener.stop()` AttributeError** — 录制正常退出时的无害副作用

---

## 研究计划 (已批准)

详见 `RESEARCH_PLAN_CN.md` 及 `/home/ubuntu20/.claude/plans/nice-and-it-is-mellow-boot.md`

### Phase 1: 触觉传感器对比

- 5 任务 × L1→L4 梯度 + 最多 7 variants 消融实验
- 量化 GelSight-style image tactile baseline vs XENSE SIMPLE vs XENSE FULL 差异
- 逐通道贡献归因（哪些物理状态对哪些任务关键）

### Phase 2 (第二贡献原型): 类脑动态触觉分辨率

核心思想（导师提出）：**人类看 4K 电视时并非全分辨率处理**——注意力集中时用小卷积核，随意浏览时用大卷积核。类比触觉：

- 简单任务 → 低触觉分辨率，大 kernel → 计算高效
- 精细任务 → 高触觉分辨率，小 kernel → 精准感知
- 机器人自主评估所需触觉分辨率 → 动态调整通道数/卷积核大小/空间分辨率

技术方案：Gated Tactile Encoder + Resolution Controller，三路径选择 low/simple、medium/task-relevant、full，训练目标 = L_task + λ·FLOPs正则化

### Phase 3 (后续扩展): 触觉元学习 (Tactile Meta-Learning)

- MAML / few-shot tactile adaptation 保留为期刊或博士阶段扩展
- 当前硕士主线先验证任务自评估和触觉分辨率选择，不把完整元学习作为主风险

