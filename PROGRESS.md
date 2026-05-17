# Data Collection Progress

> 更新于: 2026-05-11

## 当前执行策略

先做 4 个核心任务的小批量试采，试采通过后再补录：

1. `aloha_flip_switch` — 负对照，验证触觉不总是必要
2. `aloha_hidden_property_grasp` — 核心触觉必要性任务
3. `aloha_slip_hold_or_pull` — 可控滑移/夹持稳定，直接验证 Marker2D
4. `aloha_peg_insertion` — 精细插入，验证滑移 + 接触力机制

每个核心任务先录 `5 episodes`，通过 `examples/aloha2/check_collection_batch.py` 质检后再补到 20，最后补到 50。

## 正式实验任务

| 任务 | 层级 | 目标 | 已完成 | 核心触觉 | 状态 |
|------|------|------|--------|---------|------|
| aloha_flip_switch | L1 视觉主导 | 50 | 0 | Rectify | ⬜ |
| aloha_hidden_property_grasp | L2 力控关键 | 50 | 0 | ForceResultant, ForceNorm, Marker2D | ⬜ |
| aloha_slip_hold_or_pull | L2/3 可控滑移 | 50 | 0 | Marker2D, Force | ⬜ |
| aloha_peg_insertion | L3 精细触觉 | 50 | 0 | Marker2D, Force | ⬜ |
| aloha_pressure_wipe | L2/3 连续力控 | 50 | 0 | ForceNorm, ForceResultant, Depth | 第二轮 |
| aloha_towel_unfold | L4 形变操作 | 50 | 0 | Marker2D, Depth, Force | 扩展展示 |

> aloha_grasp_apple (85 eps, 测试任务), aloha_wear_shoe (50 eps) 为历史任务，非正式实验

## 训练进度

| 模型 | 触觉通道 | Steps | 状态 |
|------|---------|------|------|
| ACT | all (17ch) | 100 | Smoke test 通过 ✅ |

## 计划消融矩阵

每任务最多 7 个变体 × 4 核心任务 = 28 个第一轮模型:
```
visual_only / rectify / force / force_all / depth / marker / all
```

## 试采命令模板

```bash
conda activate lerobot
python examples/aloha_ros/aloha_ros_record.py \
  --task_name aloha_flip_switch \
  --episodes 5

python examples/aloha2/check_collection_batch.py \
  --task aloha_flip_switch \
  --min_episodes 5
```

质检通过后补录：

```bash
python examples/aloha_ros/aloha_ros_record.py \
  --task_name aloha_flip_switch \
  --episode_idx 5 \
  --episodes 15
```
