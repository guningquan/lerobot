#!/usr/bin/env python3
"""
从臂夹爪开合范围测试 — 手动掰动从臂夹爪。

禁用从臂夹爪扭矩，手动扳动夹爪，实时读取并记录开合范围。

运行前确保: roslaunch aloha aloha_test.launch
用法: /usr/bin/python3 test_puppet_gripper_range.py [left|right|both]
"""

import sys
import time
import signal
import numpy as np
from interbotix_xs_modules.arm import InterbotixManipulatorXS

PUPPET_MODEL = "vx300s"
FPS = 30
DT = 1.0 / FPS


def create_puppet(robot_name: str, init_node: bool = False):
    return InterbotixManipulatorXS(
        robot_model=PUPPET_MODEL,
        robot_name=robot_name,
        moving_time=0.3,
        accel_time=0.3,
        init_node=init_node,
    )


def get_gripper_pos(bot) -> float:
    js = bot.dxl.joint_states
    if js and len(js.position) > 6:
        return float(js.position[6])
    return 0.0


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "both"

    print("=" * 55)
    print(f"从臂夹爪开合范围测试 [{side}]")
    print("=" * 55)

    # 创建从臂
    print("\n创建从臂实例...")
    if side in ("left", "both"):
        puppet_left = create_puppet("puppet_left", init_node=True)
    if side in ("right", "both"):
        puppet_right = create_puppet("puppet_right", init_node=(side == "right"))

    # 重启夹爪 + 关闭扭矩（手动掰动）
    for name, bot in [
        ("left", puppet_left if side in ("left", "both") else None),
        ("right", puppet_right if side in ("right", "both") else None),
    ]:
        if bot is None:
            continue
        print(f"  [{name}] 重启夹爪电机 + 关闭扭矩...")
        bot.dxl.robot_reboot_motors("single", "gripper", True)
        time.sleep(0.3)
        bot.dxl.robot_torque_enable("single", "gripper", False)
        print(f"  [{name}] 夹爪已释放，可以手动掰动")

    print("\n" + "=" * 55)
    print("请手动扳动从臂夹爪，实时显示位置")
    print("Ctrl+C 退出并显示统计")
    print("=" * 55)

    history: dict[str, list[float]] = {}
    if side in ("left", "both"):
        history["left"] = []
    if side in ("right", "both"):
        history["right"] = []

    exit_flag = False
    signal.signal(signal.SIGINT, lambda *_: setattr(sys.modules[__name__], "exit_flag", True))

    t0 = time.perf_counter()

    try:
        while not exit_flag:
            t_start = time.perf_counter()
            parts = [f"{time.perf_counter() - t0:6.1f}s"]

            if side in ("left", "both"):
                lp = get_gripper_pos(puppet_left)
                history["left"].append(lp)
                parts.append(f"  左从臂: {lp:8.4f}")

            if side in ("right", "both"):
                rp = get_gripper_pos(puppet_right)
                history["right"].append(rp)
                parts.append(f"  右从臂: {rp:8.4f}")

            print("\r" + " |".join(parts), end="", flush=True)
            time.sleep(max(DT - (time.perf_counter() - t_start), 0.0))

    except Exception:
        pass

    print("\n\n开合范围统计：")
    print(f"{'从臂':<10s} {'最小':>10s} {'最大':>10s} {'范围':>10s}")
    print("-" * 42)
    for label, key in [("左", "left"), ("右", "right")]:
        if key in history and history[key]:
            vals = history[key]
            print(f"{label:<10s} {min(vals):9.4f} {max(vals):9.4f} {max(vals)-min(vals):9.4f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
