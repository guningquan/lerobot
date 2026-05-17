# Research Plan: Multi-Dimensional Tactile Sensing for Bimanual Manipulation

> 2026-05-10 | Advisor: Prof. Hayashibe | Student: Tao Liu

---

## 1. Research Motivation

GelSight-style tactile sensing commonly provides image/geometric tactile observations as policy inputs. XENSE G1-WS photonic sensors additionally expose **17 channels of explicit physical states** — including 3D force distribution, depth maps, slip detection, and 6D resultant force. This raises the question:

> Which tactile dimensions matter for which tasks? Can we build a systematic understanding of when and why additional physical states improve manipulation?

The advisor's key insight: humans use **dynamic "convolution" for sensing resolution** — coarse when the task is simple, fine when it's demanding. Similarly, a robot should self-evaluate tactile resolution requirements per task.

---

## 2. Research Questions

1. **Sensor comparison**: GelSight-style image tactile vs XENSE SIMPLE vs XENSE FULL — what does each additional explicit physical state (force, depth, marker2d) contribute?
2. **Task-tactile mapping**: Which tasks are tactile-indifferent (visual suffices), and which tasks critically depend on specific tactile dimensions?
3. **Mechanism understanding**: Not just performance numbers, but *why* specific channels help specific tasks (e.g., Marker2D detects micro-slip during insertion).
4. **Dynamic tactile resolution** (future): Can a robot learn to self-evaluate how much tactile resolution it needs?

---

## 3. Hardware


| Component       | Details                                                            |
| --------------- | ------------------------------------------------------------------ |
| Robot           | ALOHA2 quad-arm: 2× WidowX 250 (masters) + 2× ViperX-300 (puppets) |
| Cameras         | 4× Intel RealSense (640×480, 30 FPS)                               |
| Tactile Sensors | 4× XENSE G1-WS photonic sensors on puppet grippers                 |


### Tactile Channels (FULL mode, 17 channels per sensor)


| Channel        | Dims | Physical Meaning                                        |
| -------------- | ---- | ------------------------------------------------------- |
| Rectify        | 1    | Corrected grayscale contact image (GelSight-equivalent) |
| Difference     | 1    | Frame-to-frame contact change (transient events)        |
| Depth          | 1    | Contact surface depth map (3D geometry)                 |
| Force          | 3    | 3D force distribution (Fx, Fy, Fz) at 35×20 points      |
| ForceNorm      | 3    | Normal force component                                  |
| ForceResultant | 6    | 6D resultant force/moment (Fx, Fy, Fz, Mx, My, Mz)      |
| Marker2D       | 2    | Tangential displacement field (slip detection)          |


---

## 4. Task Design

The task set is redesigned to decouple tasks from tactile dimensions: four core tasks for the first thesis pass, then optional extensions for force-control and deformable manipulation.

### L1: `aloha_flip_switch` — Visual Dominant (Negative Control)


| Parameter   | Value                                   |
| ----------- | --------------------------------------- |
| Goal        | Flip a large toggle switch up/down      |
| Episodes    | 50 × 10s                                |
| Key Tactile | Rectify (minimal expected contribution) |


**Scientific role**: Prove that tactile is NOT always necessary. Expected result: `visual_only ≈ tactile_full`.

### L2: `aloha_hidden_property_grasp` — Hidden Property (Star Task)


| Parameter   | Value                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------- |
| Goal        | Pick up visually identical objects with different weight/softness/friction and place in bowl without dropping |
| Episodes    | 50 × 15s                                                                                                      |
| Key Tactile | ForceResultant, ForceNorm, Marker2D                                                                           |


**Scientific role**: Visual sensors cannot distinguish weight, softness, or surface friction of otherwise identical-looking objects. **Tactile is the only modality that can detect these hidden properties.** Expected result: `visual_only` fails dramatically; `force_all` channels provide the largest force-related gain. This is the strongest evidence for tactile necessity.

### L2/L3: `aloha_slip_hold_or_pull` — Controlled Slip / Grasp Stability


| Parameter   | Value                                                                    |
| ----------- | ------------------------------------------------------------------------ |
| Goal        | Hold a strip/object under controlled slip, or pull it by a target amount |
| Episodes    | 50 × 15s                                                                 |
| Key Tactile | Marker2D, Force                                                          |


**Scientific role**: Directly tests slip detection. Compared with full towel unfolding, this removes high-variance cloth initialization, bimanual coordination, and unfolding-score ambiguity, making Marker2D attribution cleaner.

### L3: `aloha_peg_insertion` — Fine Tactile (Classic Benchmark)


| Parameter   | Value                                         |
| ----------- | --------------------------------------------- |
| Goal        | Insert USB/peg into hole with tight clearance |
| Episodes    | 50 × 20s                                      |
| Key Tactile | Marker2D, Force                               |


**Scientific role**: Classic manipulation benchmark. Marker2D directly outputs micro-slip during insertion; compared with an image-only tactile baseline, the policy does not need to infer this physical state indirectly from contact images. Force detects contact asymmetry. Expected result: `tactile_full` >> `visual_only`.

### L2/L3 (Second Round): `aloha_pressure_wipe` — Simplified Contact Force Control


| Parameter   | Value                                                        |
| ----------- | ------------------------------------------------------------ |
| Goal        | Keep contact in a fixed area and perform a short wipe stroke |
| Episodes    | 50 × 15s                                                     |
| Key Tactile | ForceNorm, ForceResultant, Depth                             |


**Scientific role**: Adds continuous pressure-control evidence while keeping the evaluation local and stable. It is easier to attribute gains to ForceNorm / ForceResultant / Depth than the original full-board erasing task.

### L4 (Extension): `aloha_towel_unfold` — High-Variance Deformable Manipulation


| Parameter   | Value                                                                            |
| ----------- | -------------------------------------------------------------------------------- |
| Goal        | One gripper lifts cloth corner, other gripper slides along edge to unfold in air |
| Episodes    | Optional 50 × 20s                                                                |
| Key Tactile | Marker2D, Depth, Force                                                           |


**Scientific role**: Keep as a journal scenario or demo task. It is valuable but high variance because cloth state, grasp point, bimanual coordination, and unfolding metrics are entangled.

---

## 5. Ablation Experiment Matrix

For each task, train up to 7 policy variants when time permits. The first thesis pass prioritizes 4 core tasks × 4 core variants; the full matrix can expand to 6 candidate tasks × 7 variants.


| Variant       | Tactile Channels                   | Count | Meaning                                  |
| ------------- | ---------------------------------- | ----- | ---------------------------------------- |
| `visual_only` | None                               | 0     | Vision-only baseline                     |
| `rectify`     | Rectify                            | 1     | Image-only tactile baseline              |
| `force`       | Force                              | 3     | Local 3D force distribution              |
| `force_all`   | Force + ForceNorm + ForceResultant | 12    | All force-related channels               |
| `depth`       | Depth                              | 1     | Depth map only                           |
| `marker`      | Marker2D                           | 2     | Slip detection only                      |
| `full`        | All                                | 17    | Complete tactile information             |


For the thesis-critical first pass, prioritize `visual_only`, `rectify`, one task-relevant group, and `full`.

### Analysis Dimensions

1. **Per-task delta**: How much does each tactile variant improve over `visual_only`?
2. **Per-channel contribution**: Which channels drive improvement in which tasks?
3. **L1→L4 gradient**: Does tactile importance increase from Level 1 to Level 4?
4. **Independent vs. synergistic**: Do multiple channels produce > sum of individual gains?

---

## 6. Expected Findings & Contributions

### Scientific Contributions

1. **First systematic ablation** of multi-dimensional photonic tactile sensing across a task gradient
2. **Evidence that tactile necessity is task-dependent**: some tasks don't need it (flip_switch), others critically depend on it (hidden_property_grasp, peg_insertion)
3. **Per-channel attribution**: quantifying which physical states matter for which task types
4. **Tactile gradient taxonomy** (L1→L4) that can guide future sensor and policy design

### Expected Results Table (Hypothesis)


| Task                       | visual_only | rectify | force | force_all | depth | marker | full |
| -------------------------- | ----------- | ------- | ----- | --------- | ----- | ------ | ---- |
| flip_switch (L1)           | ★★★         | ★★★     | ★★★   | ★★★       | ★★★   | ★★★    | ★★★  |
| hidden_property (L2)       | ★           | ★       | ★★    | ★★★       | ★★    | ★★     | ★★★  |
| slip_hold_or_pull (L2/L3)  | ★★          | ★★      | ★★    | ★★        | ★     | ★★★    | ★★★  |
| peg_insertion (L3)         | ★           | ★★      | ★★    | ★★        | ★     | ★★★    | ★★★  |
| pressure_wipe (optional)   | ★★          | ★★      | ★★    | ★★★       | ★★★   | ★★     | ★★★  |


★ = poor, ★★ = moderate, ★★★ = good

### Key Narrative

> "Tactile sensing is not uniformly necessary — its value is a function of task properties. Hidden physical properties, controlled slip, fine insertion, and pressure control depend on different tactile dimensions. Our ablation reveals *which* dimensions matter *where*, enabling task-driven tactile sensor design and dynamic tactile resolution allocation."

---

## 7. Part 2 (Second Thesis Contribution): Lightweight Dynamic Tactile Resolution

This should be a minimal, testable **Adaptive Tactile Resolution** prototype, building on the ablation findings from Part 1. Full MAML-style tactile meta-learning remains a future extension rather than the main MS thesis risk.

### Core Idea: Dynamic Convolution for Tactile Sensing

The human visual system does not process an entire 4K image at full resolution. Instead, it uses **foveated attention** — small convolutional kernels (high resolution) at the point of focus, large kernels (low resolution) in the periphery. This is computationally efficient because *processing resolution adapts to task demands*.

The same principle applies to tactile sensing:

- **Fine manipulation** (peg insertion, threading) → small kernel, high tactile resolution, all physical states
- **Coarse manipulation** (flip switch, push cube) → large kernel, low tactile resolution, single channel
- **Intermediate tasks** → adaptive resolution with partial channel activation

### Formalization

```
Tactile Processing = f(tactile_input, σ(task_context))

where σ is the Resolution Controller:
  - kernel_size ∈ {3, 5, 7, 9}       (small = fine, large = coarse)
  - channel_group ∈ {SIMPLE, task-relevant, ALL}
  - spatial_downsample ∈ {1×, 2×, 4×}
```

### Proposed Architecture: Gated Tactile Encoder with Resolution Controller

```
┌─────────────────┐
│  Task Context    │  ← visual features + robot proprioception
│  (visual+state)  │
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Resolution      │  ← Gumbel-Softmax gating (differentiable discrete selection)
│  Controller      │     learns to output (kernel_size, channel_mask, downsample_ratio)
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Dynamic Tactile │  ← adapts convolution kernel size and channel count
│  Encoder         │     based on controller output
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Policy Head     │  ← ACT / Diffusion action prediction
│  (ACT/Diffusion) │
└──────────────────┘
```

### Training Objective

```
L_total = L_task + λ * FLOPs(resolution_choice) / FLOPs_max
```

The model is incentivized to use low resolution when possible and high resolution only when necessary — achieving **computational efficiency through self-evaluation**.

### Self-Evaluation Mechanism

The key innovation: **the robot judges by itself what tactile resolution the current task requires**.

Implementation:
1. **Low-cost path**: `rectify` / SIMPLE for low tactile-dependency tasks such as `flip_switch`
2. **Medium path**: task-relevant channels such as `force_all` or `marker2d`
3. **High-cost path**: `full` 17-channel tactile input for hidden-property, insertion, and deformable tasks
4. **Gating selector**: choose among the paths using visual features, robot state, and early contact frames, with a compute/latency penalty that encourages low-cost choices when performance is unaffected

### Expected Contributions

1. **Task-dependent tactile physical-state analysis**: identify which tasks need which tactile states
2. **Computational efficiency**: robot allocates tactile computation proportional to task need
3. **Self-awareness**: robot learns to evaluate its own perceptual requirements
4. **Path toward meta-learning**: MAML/few-shot tactile adaptation remains a later journal or PhD extension

---

## 8. Implementation Roadmap

### Phase 1: Data Collection (Current)

- First round: record 50 episodes × 4 core tasks = 200 episodes
- Second round: add `aloha_pressure_wipe` and `aloha_towel_unfold` if time allows
- FULL mode 17ch tactile + 4 cameras
- Target: ~8 hours of teleoperation

### Phase 2: Ablation Training

- Prioritize 4 core tasks: `flip_switch`, `hidden_property_grasp`, `slip_hold_or_pull`, and `peg_insertion`
- First train 4 core variants per task: `visual_only`, `rectify`, one task-relevant channel group, and `full`
- Expand to the full 6-task × 7-variant matrix if time allows
- Evaluate success rate per variant per task
- Visualize tactile attention maps

### Phase 3: Analysis

- Per-channel contribution analysis
- Mechanism analysis: slip/contact onset, ForceResultant grasp stability, Depth contact geometry, and task boundaries where `rectify ≈ full` versus `rectify << full`
- Qualitative failure mode analysis
- Write paper

### Phase 4 (Second Contribution Prototype): Dynamic Tactile Resolution

- Gated Tactile Encoder with learnable resolution controller
- Task self-evaluation module
- Three-path choice: low/simple, medium/task-relevant, full
- MAML-based tactile adaptation is reserved for later work

---

## 8. Codebase


| Component      | Path                                            |
| -------------- | ----------------------------------------------- |
| Recording      | `examples/aloha_ros/aloha_ros_record.py`        |
| Training       | `examples/aloha2/train_aloha2_tactile.py`       |
| Tactile Viz    | `examples/aloha2/extract_tactile_viz.py`        |
| Task Configs   | `examples/aloha_ros/aloha_scripts/constants.py` |
| Tactile Driver | `src/lerobot/tactile/xense_g1ws/`               |
| ACT Model      | `src/lerobot/policies/act/`                     |


Full setup documentation: `SETUP_SUMMARY.md`
Progress tracking: `PROGRESS.md`
