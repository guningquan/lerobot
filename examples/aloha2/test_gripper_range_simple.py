#!/usr/bin/env python3
"""夹爪开合范围测试 - 纯 ROS 订阅读数 + 可选夹爪释放。
用法:
  python test_gripper_range_simple.py master_left          # 主臂（默认已释放）
  python test_gripper_range_simple.py master_right
  python test_gripper_range_simple.py puppet_left --release # 从臂需释放扭矩
  python test_gripper_range_simple.py puppet_right --release
"""
import sys
import time
import signal
import rospy
from sensor_msgs.msg import JointState


def release_gripper(robot_name: str, robot_model: str):
    """用 Interbotix 释放夹爪扭矩，使其可手动拨动。"""
    from interbotix_xs_modules.arm import InterbotixManipulatorXS
    print(f"  释放 {robot_name} 夹爪扭矩 ...")
    bot = InterbotixManipulatorXS(
        robot_model=robot_model, robot_name=robot_name,
        moving_time=0.3, accel_time=0.3, init_node=False,
    )
    if "puppet" in robot_name:
        bot.dxl.robot_reboot_motors("single", "gripper", True)
        time.sleep(0.3)
    bot.dxl.robot_set_operating_modes("single", "gripper", "pwm")
    bot.dxl.robot_torque_enable("single", "gripper", False)
    print("  夹爪已释放，可以手动拨动")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if len(args) < 1:
        print("用法: python test_gripper_range_simple.py <robot_name> [--release]")
        print("示例: python test_gripper_range_simple.py master_left")
        print("      python test_gripper_range_simple.py puppet_left --release")
        sys.exit(1)

    robot = args[0]
    topic = f"/{robot}/joint_states"

    # 先初始化 ROS 节点（Interbotix 需要）
    rospy.init_node(f"gripper_range_{robot}", anonymous=True)

    # 判断臂类型
    if "master" in robot:
        robot_model = "wx250s"
    else:
        robot_model = "vx300s"

    # 释放夹爪（必须在 rospy.init_node 之后）
    if "--release" in flags:
        release_gripper(robot, robot_model)

    # 读数
    latest_gripper = [0.0]
    history = []

    def cb(msg: JointState):
        for i, name in enumerate(msg.name):
            if name == "gripper":
                latest_gripper[0] = msg.position[i]
                history.append(msg.position[i])
                break

    rospy.Subscriber(topic, JointState, cb)

    print(f"等待 {topic} ...")
    for _ in range(50):
        if history:
            break
        time.sleep(0.1)
    if not history:
        print(f"错误: 未收到 {topic} 消息，请确认 roslaunch 是否已启动")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"夹爪开合范围测试 [{robot}]")
    print(f"{'='*55}")
    print("手动拨动夹爪到最大和最小位置")
    print("Ctrl+C 退出并显示统计\n")

    exit_flag = [False]
    signal.signal(signal.SIGINT, lambda *_: exit_flag.__setitem__(0, True))

    while not exit_flag[0]:
        val = latest_gripper[0]
        if abs(val) * 30 < 50:
            bar = "#" * int(abs(val) * 30)
        else:
            bar = "#" * 50
        print(f"\r  夹爪位置: {val:+.4f}  {bar}", end="", flush=True)
        time.sleep(0.05)

    print("\n\n统计:")
    if len(history) > 10:
        print(f"  最小值 (闭合): {min(history):.4f}")
        print(f"  最大值 (张开): {max(history):.4f}")
        print(f"  开合范围:      {max(history) - min(history):.4f}")
    else:
        print("  数据不足，请手动拨动夹爪")
    print("完成。")


if __name__ == "__main__":
    main()
