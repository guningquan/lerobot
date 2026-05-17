#!/usr/bin/env python3
"""读取 4 个臂当前关节位置，用于设置独立起始位姿。"""
import rospy
from sensor_msgs.msg import JointState

JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
latest = {}

def make_cb(arm):
    def cb(msg):
        vals = []
        for j in JOINTS:
            if j in msg.name:
                vals.append(msg.position[msg.name.index(j)])
            else:
                vals.append(0.0)
        latest[arm] = vals
    return cb

rospy.init_node("read_poses", anonymous=True)
for arm in ["master_left", "master_right", "puppet_left", "puppet_right"]:
    rospy.Subscriber(f"/{arm}/joint_states", JointState, make_cb(arm))

print("等待关节数据...")
rospy.sleep(1)

print("\n各臂当前关节位置 [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]:")
for arm in ["master_left", "master_right", "puppet_left", "puppet_right"]:
    if arm in latest:
        vals = ", ".join(f"{v:.3f}" for v in latest[arm])
        print(f"  {arm}: [{vals}]")
    else:
        print(f"  {arm}: 无数据")
