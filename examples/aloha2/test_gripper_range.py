"""
ALOHA2 夹爪开合范围测试脚本。

用户手动开合主臂夹爪 → 读取主臂位置 → 直接发送给从臂夹爪
→ 读取从臂实际位置 → 实时显示并记录范围。

只操作夹爪电机，其他关节保持不动。

Usage:
  conda activate lerobot
  python examples/aloha2/test_gripper_range.py
"""

import time
from collections import defaultdict
from pathlib import Path

from lerobot.robots.aloha import Aloha, AlohaConfig
from lerobot.teleoperators.aloha_teleop import AlohaTeleop, AlohaTeleopConfig
from lerobot.utils.utils import log_say

# ── 硬件配置 ────────────────────────────────────────────────────────
CALIB_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"

robot_config = AlohaConfig(
    id="aloha",
    left_arm_port="/dev/ttyDXL_puppet_left",
    right_arm_port="/dev/ttyDXL_puppet_right",
    left_arm_max_relative_target=200.0,   # 足够大，对夹爪不构成限制
    right_arm_max_relative_target=200.0,
    left_arm_use_degrees=True,
    right_arm_use_degrees=True,
    calibration_dir=CALIB_DIR,
    cameras={},
    tactile_sensors={},
)

teleop_config = AlohaTeleopConfig(
    id="aloha_teleop",
    left_arm_port="/dev/ttyDXL_master_left",
    right_arm_port="/dev/ttyDXL_master_right",
    left_arm_gripper_motor="xc430-w150",
    right_arm_gripper_motor="xc430-w150",
    left_arm_use_degrees=True,
    right_arm_use_degrees=True,
    calibration_dir=CALIB_DIR,
)

# ── 只操作的关节 ────────────────────────────────────────────────────
GRIPPER_KEYS = ["left_gripper.pos", "right_gripper.pos"]

# ── 初始化 ──────────────────────────────────────────────────────────
robot = Aloha(robot_config)
teleop = AlohaTeleop(teleop_config)

log_say("连接主臂 (WidowX) 和从臂 (ViperX)...")
robot.connect()
teleop.connect()
log_say("已连接。请手动缓慢开合主臂夹爪，Ctrl+C 结束测试。\n")

history: dict[str, list[float]] = defaultdict(list)

try:
    print(f"{'时刻':>8s} | {'左主臂':>8s} → {'左从臂':>8s} | {'右主臂':>8s} → {'右从臂':>8s}")
    print("-" * 65)

    t0 = time.perf_counter()

    while True:
        # 1. 读取主臂动作
        master_action = teleop.get_action()

        # 2. 提取夹爪位置
        gripper_action = {k: master_action[k] for k in GRIPPER_KEYS}

        # 3. 直接写入 Goal_Position，绕过 send_action 的 max_relative_target 钳制
        for gripper_key, target_pos in gripper_action.items():
            side = "left" if gripper_key.startswith("left") else "right"
            arm = robot.left_arm if side == "left" else robot.right_arm
            arm.bus.write("Goal_Position", "gripper", target_pos)

        # 4. 读取从臂实际位置
        obs = robot.get_observation()

        # 5. 记录 + 显示
        elapsed = time.perf_counter() - t0
        row = f"{elapsed:7.1f}s |"

        for gripper_key in GRIPPER_KEYS:
            master_val = master_action[gripper_key]
            follower_val = obs[gripper_key]
            history[gripper_key].append(follower_val)

            row += f" {master_val:7.1f} → {follower_val:7.1f} |"

        print(f"\r{row}", end="", flush=True)

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    log_say("\n\n断开连接...")
    robot.disconnect()
    teleop.disconnect()

# ── 统计 ────────────────────────────────────────────────────────────
log_say("\n夹爪开合范围统计：")
print(f"{'关节':<25s} {'最小':>8s} {'最大':>8s} {'范围':>8s}")
print("-" * 55)

for gripper_key in GRIPPER_KEYS:
    if history[gripper_key]:
        vals = history[gripper_key]
        min_v, max_v = min(vals), max(vals)
        print(f"{gripper_key:<25s} {min_v:7.1f}% {max_v:7.1f}% {max_v - min_v:7.1f}%")
    else:
        print(f"{gripper_key:<25s} {'无数据':>8s}")

log_say("\n测试完成。")
