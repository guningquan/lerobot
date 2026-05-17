#!/usr/bin/env python3
"""主臂夹爪开合范围测试。 运行前: roslaunch aloha aloha_test.launch"""
import sys, time, signal
from interbotix_xs_modules.arm import InterbotixManipulatorXS

def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "both"
    bot_left = InterbotixManipulatorXS(robot_model="wx250s", robot_name="master_left",
                                        moving_time=0.3, accel_time=0.3, init_node=True)
    if side in ("right", "both"):
        bot_right = InterbotixManipulatorXS(robot_model="wx250s", robot_name="master_right",
                                             moving_time=0.3, accel_time=0.3, init_node=False)

    for name, bot in [("左主臂", bot_left), ("右主臂", bot_right)]:
        if side not in (name[0], "both"): continue
        bot.dxl.robot_set_operating_modes("single", "gripper", "pwm")
        bot.dxl.robot_torque_enable("single", "gripper", False)
        print(f"  [{name}] 夹爪已释放")

    print("\n手动扳动主臂夹爪，Ctrl+C 显示统计\n")
    history = {k: [] for k in (["left"] if side=="left" else ["right"] if side=="right" else ["left","right"])}

    exit_flag = [False]
    signal.signal(signal.SIGINT, lambda *_: exit_flag.__setitem__(0, True))
    t0 = time.perf_counter()

    while not exit_flag[0]:
        parts = [f"{time.perf_counter()-t0:6.1f}s"]
        for key, bot in [("left",bot_left), ("right",bot_right)]:
            if key in history:
                js = bot.dxl.joint_states
                p = float(js.position[6]) if js and len(js.position)>6 else 0.0
                history[key].append(p)
                parts.append(f"  {key}: {p:8.4f}")
        print("\r" + " |".join(parts), end="", flush=True)
        time.sleep(0.033)

    print("\n")
    for label, key in [("左主臂","left"), ("右主臂","right")]:
        if key in history and history[key]:
            v = history[key]
            print(f"{label}: min={min(v):.4f}  max={max(v):.4f}  range={max(v)-min(v):.4f}")
    print("完成。")

if __name__ == "__main__":
    main()
