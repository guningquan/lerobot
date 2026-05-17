# 导师邮件分析：研究计划逐段对应

> 邮件原文 | 分析：Tao Liu | 2026-05-11

---

## 邮件段落 1：传感器对比

> Nice, and it is good to analyze sensor information differences. in terms of tactile sensing, sensitivity, density of touch sense, and physical states differences between sensors. if new sensor has more detailed information on tactile, than gel sight.


| 要求              | 对应方案                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------ |
| 分析传感器信息差异       | `RESEARCH_PLAN_CN.md` Section 3 — GelSight-style image tactile vs XENSE SIMPLE vs XENSE FULL 对比表 |
| 灵敏度、触觉密度、物理状态差异 | Section 3 — 17 通道逐通道详解（Force/ForceNorm/ForceResultant/Marker2D/Depth/Difference/Rectify）         |
| 新传感器是否有更多显式物理状态 | Section 3 对比表：XENSE FULL 显式输出 Depth/Force/ForceResultant/Marker2D 等物理状态                          |


**实现**：FULL 模式为 17ch/传感器（4 传感器 × 17 通道 = 68 维触觉特征），SIMPLE/Rectify 模式作为 image-only tactile baseline。注意：不把 GelSight 简化为“无 3D/力信息”，这里比较的是策略输入层面的 image-only baseline vs explicit physical-state tactile。

---

## 邮件段落 2：消融对比 + 机制解释

> also nice to have comparison like gelsight vs new sensor simple (equivalent to gelsight) vs new sensor full

> and nice to discuss how new additional states affect on the performance, not just performance comparison, technical understanding the mechanism of improvement.


| 要求                              | 对应方案                                                                                                                                        |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| GelSight vs SIMPLE vs FULL 三者对比 | `RESEARCH_PLAN_CN.md` Section 5 — 消融实验矩阵：`visual_only` / `rectify`（image-only tactile）/ `force` / `force_all` / `depth` / `marker` / `full` |
| 不只报性能，要理解机制                     | Section 4 每个任务有 "Scientific role" 说明为什么特定通道对特定任务有效                                                                                          |
| 机制解释示例                          | peg_insertion: Marker2D 直接测量微滑移（GelSight 需光流推算且不准）；Force 检测接触力不对称判断卡住                                                                       |


**实现**：每个任务最多训练 7 个变体，通过 channel masking 实现维度消融，分析各通道的边际贡献。其中 `force`=3 通道 Force，`force_all`=12 通道 Force+ForceNorm+ForceResultant。

---

## 邮件段落 3：哪些任务有差异、哪些相同

> //////////////////(and journal senario after the first part)
> and for some tasks, performance can be equivalent, then which task has difference, and which task has same.
> if this aspect can be revealed, there is good scientific value.


| 要求           | 对应方案                                                    |
| ------------ | ------------------------------------------------------- |
| 哪些任务有差异、哪些相同 | `RESEARCH_PLAN_CN.md` Section 4 任务梯度 + Section 6 假设结果矩阵 |
| 这个发现本身有科学价值  | Section 6 核心叙事："触觉感知的价值不是一刀切的——它的必要性取决于任务性质"            |


**实现**：五个任务按 L1→L4 梯度设计：

```
L1 flip_switch:  visual_only ≈ tactile_full  (触觉不必要)
L2 hidden_property_grasp:  visual_only ≪ tactile_full  (触觉关键)
L2/L3 slip_hold_or_pull:  marker/full > rectify  (可控滑移需要 Marker2D)
L3 peg_insertion:  marker >> others  (Marker2D 独特价值)
L2/L3 pressure_wipe:  force_all/depth > rectify  (第二轮连续力控补充)
L4 towel_unfold:  journal scenario / 展示任务，不作为第一阶段核心
```

---

## 邮件段落 4：动态卷积（核心创新思想）

> and in my opinion, human uses dynamic convolution for changing sensing resolution.
> I mean even we watch 4k television, our eye doent process all full resolution.
> if you want to watch carefully convolution size is small,
> and casual look with large convolution to reduce computational burden.


| 要求              | 对应方案                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------ |
| 人眼动态分辨率的类比      | `RESEARCH_PLAN_CN.md` Section 7 — 核心思想第一段                                                  |
| 小核 = 精细，大核 = 粗略 | Section 7 形式化定义：`kernel_size ∈ {3, 5, 7, 9}`，`channel_group ∈ {SIMPLE, SIMPLE+force, ALL}` |
| 降低计算负担          | Section 7 训练目标：`L_total = L_task + λ × FLOPs(resolution_choice) / FLOPs_max`               |


**实现**：Gated Tactile Encoder + Gumbel-Softmax Resolution Controller。控制器根据任务上下文输出离散选择（核大小/通道组/空间下采样率），通过 FLOPs 正则化激励低分辨率路径。

---

## 邮件段落 5：自评估 + 元学习

> then task is enough with low resolution of tactile, we simply processes.
> then there would be self-evaluation depending on task.
> it can be computationally efficient if we need low or high resolution,
> dynamically by task self-evaluation.


| 要求                | 对应方案                                                                             |
| ----------------- | -------------------------------------------------------------------------------- |
| 简单任务低分辨率、复杂任务高分辨率 | `RESEARCH_PLAN_CN.md` Section 7 — 分辨率按任务动态调整                                     |
| 自评估               | Section 7 — Task Embedding → Difficulty Predictor (输出 0~1) → Resolution Selector |
| 计算高效              | Section 7 — FLOPs 正则化激励模型在简单任务选择低成本路径                                            |


**实现**：三个组件——

1. **Task Embedding Network**：从少量交互帧提取任务表征（contrastive learning）
2. **Difficulty Predictor**：预测任务的触觉依赖度
3. **Resolution Selector**：根据依赖度动态调整触觉编码器的卷积核/通道/下采样率

---

## 邮件段落 6：硕士论文方向 + 触觉元学习

> if we can judge resolution requirement by robot itself, it can be good master thesis.

> Tactille meta-learning (meta-learning is self evaluation tuning how to learn) not sure, but it might be new way of thinking


| 要求                    | 对应方案                                                              |
| --------------------- | ----------------------------------------------------------------- |
| 机器人自己判断分辨率需求          | Part 2 的 Self-Evaluation 机制 — 这是硕士论文核心创新点                         |
| 可作为硕士论文               | `RESEARCH_PLAN_CN.md` Section 7 标题："第二部分（硕士论文核心创新）：类脑动态触觉分辨率与元学习" |
| Tactile Meta-Learning | Section 7 — 保留为后续扩展，当前硕士先做轻量自评估调优                                 |
| 可能是新的研究方向             | Section 7 预期贡献："触觉元学习（Tactile Meta-Learning）可能是一个新的研究范式"          |


**实现策略**：

- 当前硕士主线：low/simple、medium/task-relevant、full 三路径触觉输入选择
- 后续扩展：MAML / few-shot tactile adaptation

---

## 总结：邮件 → 计划映射表


| 邮件关键词                                     | 研究计划中的实现                                          | 文档位置                         |
| ----------------------------------------- | ------------------------------------------------- | ---------------------------- |
| sensor information differences            | GelSight-style image tactile vs SIMPLE vs FULL 对比 | `RESEARCH_PLAN_CN.md` §3     |
| sensitivity, density, physical states     | 17 通道逐类详解                                         | `RESEARCH_PLAN_CN.md` §3     |
| gelsight vs simple vs full                | 最多 7-variant 消融矩阵                                 | `RESEARCH_PLAN_CN.md` §5     |
| mechanism of improvement                  | 每任务 scientific role + 逐通道贡献分析                     | `RESEARCH_PLAN_CN.md` §4     |
| which task has difference, which has same | L1→L4 梯度 + 假设结果矩阵                                 | `RESEARCH_PLAN_CN.md` §4, §6 |
| dynamic convolution / 4k television       | Gated Tactile Encoder + Resolution Controller     | `RESEARCH_PLAN_CN.md` §7     |
| self-evaluation depending on task         | Difficulty Predictor + Resolution Selector        | `RESEARCH_PLAN_CN.md` §7     |
| computationally efficient                 | FLOPs 正则化                                         | `RESEARCH_PLAN_CN.md` §7     |
| master thesis                             | 第二部分标题标注                                          | `RESEARCH_PLAN_CN.md` §7     |
| Tactile Meta-Learning                     | 后续扩展；当前先做 Adaptive Tactile Resolution             | `RESEARCH_PLAN_CN.md` §7     |

