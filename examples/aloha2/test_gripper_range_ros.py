#!/usr/bin/env python3
"""
ALOHA2 夹爪开合范围测试 — 纯 ROS/Interbotix 版本。

直接使用 InterbotixManipulatorXS，不依赖 lerobot。
运行在系统 Python 3.8 + ROS Noetic 环境。

用法:
    python3 test_gripper_range_ros.py
"""

import sys
import time
import numpy as np

# 导入 Interbotix ROS 模块
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

# ── 主臂→从臂夹爪位置映射（实测校准值） ──
MASTER_LEFT_MIN, MASTER_LEFT_MAX = -0.2102, 0.6335
MASTER_RIGHT_MIN, MASTER_RIGHT_MAX = -0.0506, 0.8330
PUPPET_LEFT_MIN, PUPPET_LEFT_MAX = -1.7871, -0.2961
PUPPET_RIGHT_MIN, PUPPET_RIGHT_MAX = -1.5294, -0.0966


def master2puppet_left(master_pos: float) -> float:
    """实测左臂映射。"""
    norm = (master_pos - MASTER_LEFT_MIN) / (MASTER_LEFT_MAX - MASTER_LEFT_MIN)
    return norm * (PUPPET_LEFT_MAX - PUPPET_LEFT_MIN) + PUPPET_LEFT_MIN


def master2puppet_right(master_pos: float) -> float:
    """实测右臂映射。"""
    norm = (master_pos - MASTER_RIGHT_MIN) / (MASTER_RIGHT_MAX - MASTER_RIGHT_MIN)
    return norm * (PUPPET_RIGHT_MAX - PUPPET_RIGHT_MIN) + PUPPET_RIGHT_MIN

# ── 配置 ───────────────────────────────────────────────────────────
# 机器人名称（与 ROS launch 文件中一致）
MASTER_LEFT_NAME = "master_left"
MASTER_RIGHT_NAME = "master_right"
PUPPET_LEFT_NAME = "puppet_left"
PUPPET_RIGHT_NAME = "puppet_right"

# 机器人型号
MASTER_MODEL = "wx250s"
PUPPET_MODEL = "vx300s"

# 测试参数
FPS = 30
DT = 1.0 / FPS


def create_robot(robot_name: str, model: str, init_node: bool = False):
    """创建并初始化 Interbotix 机械臂实例。"""
    bot = InterbotixManipulatorXS(
        robot_model=model,
        robot_name=robot_name,
        moving_time=0.3,
        accel_time=0.3,
        init_node=init_node,
    )
    return bot


def setup_robots(master_left_bot, master_right_bot, puppet_left_bot, puppet_right_bot):
    """设置所有机械臂的工作模式和扭矩。"""
    # ── 从臂：重启夹爪 + 电流位置模式 ──
    for name, bot in [("puppet_left", puppet_left_bot), ("puppet_right", puppet_right_bot)]:
        print(f"  [{name}] 重启夹爪电机...")
        bot.dxl.robot_reboot_motors("single", "gripper", True)
        time.sleep(0.3)

    # ── 主臂：position 模式初始化 + 扭矩使能 ──
    for name, bot in [("master_left", master_left_bot), ("master_right", master_right_bot)]:
        print(f"  [{name}] 设置 position 模式（臂+夹爪）...")
        bot.dxl.robot_set_operating_modes("group", "arm", "position")
        bot.dxl.robot_set_operating_modes("single", "gripper", "position")
        bot.dxl.robot_torque_enable("group", "arm", True)
        bot.dxl.robot_torque_enable("single", "gripper", True)

    # ── 从臂：position 模式 + 电流夹爪 + 扭矩使能 ──
    for name, bot in [("puppet_left", puppet_left_bot), ("puppet_right", puppet_right_bot)]:
        print(f"  [{name}] 设置 position 模式（臂）+ current_based_position（夹爪）...")
        bot.dxl.robot_set_operating_modes("group", "arm", "position")
        bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        bot.dxl.robot_torque_enable("group", "arm", True)
        bot.dxl.robot_torque_enable("single", "gripper", True)

    # ── 主臂夹爪切换到 PWM 模式 → 扭矩关闭 → 人手可自由操作 ──
    for name, bot in [("master_left", master_left_bot), ("master_right", master_right_bot)]:
        print(f"  [{name}] 夹爪切换 PWM + 关闭扭矩（手动操作）...")
        bot.dxl.robot_set_operating_modes("single", "gripper", "pwm")
        bot.dxl.robot_torque_enable("single", "gripper", False)

    print("  所有机械臂初始化完成")


def get_gripper_position(bot) -> float:
    """读取夹爪当前位置。"""
    joint_states = bot.dxl.joint_states
    if joint_states and len(joint_states.position) > 6:
        return float(joint_states.position[6])
    return 0.0


def set_gripper_position(bot, target: float):
    """设置从臂夹爪目标位置。"""
    cmd = JointSingleCommand(name="gripper")
    cmd.cmd = target
    bot.gripper.core.pub_single.publish(cmd)


def main():
    print("=" * 60)
    print("ALOHA2 夹爪开合范围测试")
    print("=" * 60)

    # ── 创建四只机械臂 ──
    print("\n[1/3] 创建机械臂实例...")
    master_left  = create_robot(MASTER_LEFT_NAME,  MASTER_MODEL, init_node=True)
    master_right = create_robot(MASTER_RIGHT_NAME, MASTER_MODEL, init_node=False)
    puppet_left  = create_robot(PUPPET_LEFT_NAME,  PUPPET_MODEL, init_node=False)
    puppet_right = create_robot(PUPPET_RIGHT_NAME, PUPPET_MODEL, init_node=False)

    # ── 初始化（重启夹爪 + 设置模式 + 使能扭矩） ──
    print("\n[2/3] 初始化电机...")
    setup_robots(master_left, master_right, puppet_left, puppet_right)

    # ── 读取当前所有夹爪位置 ──
    print("\n[3/3] 当前夹爪位置：")
    for name, bot in [
        ("左主臂 (master_left)", master_left),
        ("右主臂 (master_right)", master_right),
        ("左从臂 (puppet_left)", puppet_left),
        ("右从臂 (puppet_right)", puppet_right),
    ]:
        pos = get_gripper_position(bot)
        print(f"  {name}: {pos:.4f}")

    print("\n" + "=" * 60)
    print("测试模式：请手动缓慢开合主臂夹爪")
    print("从臂夹爪将实时跟随主臂位置")
    print("Ctrl+C 退出并显示统计")
    print("=" * 60)

    # ── 历史记录 ──
    history: dict[str, list[float]] = {
        "master_left": [],
        "master_right": [],
        "puppet_left": [],
        "puppet_right": [],
    }

    print(f"\n{'时刻':>8s} | {'左主臂':>10s} → {'左从臂':>10s} | {'右主臂':>10s} → {'右从臂':>10s}")
    print("-" * 70)

    t0 = time.perf_counter()

    import signal
    exit_flag = False

    def signal_handler(sig, frame):
        nonlocal exit_flag
        exit_flag = True

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while not exit_flag:
            t_start = time.perf_counter()

            # 读取主臂夹爪
            ml_pos = get_gripper_position(master_left)
            mr_pos = get_gripper_position(master_right)

            # 发送给从臂夹爪（实测校准映射）
            set_gripper_position(puppet_left, master2puppet_left(ml_pos))
            set_gripper_position(puppet_right, master2puppet_right(mr_pos))

            # 给一点时间让命令生效
            time.sleep(0.02)

            # 读取从臂实际位置
            pl_pos = get_gripper_position(puppet_left)
            pr_pos = get_gripper_position(puppet_right)

            # 记录
            history["master_left"].append(ml_pos)
            history["master_right"].append(mr_pos)
            history["puppet_left"].append(pl_pos)
            history["puppet_right"].append(pr_pos)

            # 显示
            elapsed = time.perf_counter() - t0
            print(
                f"\r{elapsed:7.1f}s | {ml_pos:9.4f} → {pl_pos:9.4f} | {mr_pos:9.4f} → {pr_pos:9.4f}",
                end="", flush=True,
            )

            # 频率控制
            sleep_time = DT - (time.perf_counter() - t_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    # ── 统计 ──
    print("\n\n夹爪开合范围统计：")
    print(f"{'夹爪':<20s} {'最小':>12s} {'最大':>12s} {'范围':>12s}")
    print("-" * 58)

    for label, key in [
        ("左主臂 (master_left)", "master_left"),
        ("右主臂 (master_right)", "master_right"),
        ("左从臂 (puppet_left)", "puppet_left"),
        ("右从臂 (puppet_right)", "puppet_right"),
    ]:
        vals = history[key]
        if vals:
            print(f"{label:<20s} {min(vals):11.4f} {max(vals):11.4f} {max(vals)-min(vals):11.4f}")
        else:
            print(f"{label:<20s} {'无数据':>12s}")

    print("\n测试完成。")


if __name__ == "__main__":
    main()
