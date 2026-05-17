#!/usr/bin/env python3
"""
ALOHA2 ROS 录制脚本 — 适配 theodoreliu 硬件。

4 臂遥操作 + 4 RealSense 相机同步录制。
需要在系统 Python 3.8 + ROS Noetic 环境运行。

用法:
  终端1: roslaunch aloha aloha_test.launch
  终端2: python record_aloha2_ros.py
"""

import sys, os, time, numpy as np
from pathlib import Path

# ── 使用服务器端的 lerobot（含 ROS 模块） ──
SERVER_LEROBOT = "/mnt/server/Programs_server/metalerobot/src"
if SERVER_LEROBOT not in sys.path:
    sys.path.insert(0, SERVER_LEROBOT)
# aloha_scripts constants
sys.path.insert(0, "/mnt/server/Programs_server/metalerobot/examples/aloha_ros/aloha_scripts")

from constants import (
    START_ARM_POSE, FPS,
    MASTER_GRIPPER_JOINT_MID, MASTER_GRIPPER_JOINT_CLOSE,
    PUPPET_GRIPPER_JOINT_CLOSE,
)

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.processor.pipeline import RobotActionProcessorStep, RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.robots.aloha_ros import AlohaRos, AlohaRosConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.aloha_teleop_ros import AlohaTeleopRos, AlohaTeleopRosConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.robot_utils import precise_sleep

# ── 实测校准值 (2026-05-13) ───────────────────────────────────────
MASTER_LEFT_MIN, MASTER_LEFT_MAX   = -0.0752, 0.7517
MASTER_RIGHT_MIN, MASTER_RIGHT_MAX = -0.0552, 0.8191
PUPPET_LEFT_MIN, PUPPET_LEFT_MAX   = -1.2090, 0.3068
PUPPET_RIGHT_MIN, PUPPET_RIGHT_MAX = -1.3514, 0.1212


def master2puppet_left(x):
    norm = (x - MASTER_LEFT_MIN) / (MASTER_LEFT_MAX - MASTER_LEFT_MIN)
    return norm * (PUPPET_LEFT_MAX - PUPPET_LEFT_MIN) + PUPPET_LEFT_MIN

def master2puppet_right(x):
    norm = (x - MASTER_RIGHT_MIN) / (MASTER_RIGHT_MAX - MASTER_RIGHT_MIN)
    return norm * (PUPPET_RIGHT_MAX - PUPPET_RIGHT_MIN) + PUPPET_RIGHT_MIN


# ── 录制配置 ────────────────────────────────────────────────────────
HF_REPO_ID = "theodoreliu/aloha2_test"
TASK_DESCRIPTION = "test teleoperation"
NUM_EPISODES = 3
EPISODE_TIME_SEC = 30
DATASET_ROOT = "/home/robot/Dataset_and_Checkpoint/lerobot-dataset"

# ── 相机配置（与你的硬件一致） ─────────────────────────────────────
camera_config = {
    "cam_high":        RealSenseCameraConfig(serial_number_or_name="109422062625", fps=30, width=640, height=480),
    "cam_low":         RealSenseCameraConfig(serial_number_or_name="936322072119", fps=30, width=640, height=480),
    "cam_left_wrist":  RealSenseCameraConfig(serial_number_or_name="134222077139", fps=30, width=640, height=480),
    "cam_right_wrist": RealSenseCameraConfig(serial_number_or_name="943222070893", fps=30, width=640, height=480),
}


# ── 辅助函数 ────────────────────────────────────────────────────────
def get_arm_joint_positions(bot):
    js = bot.bot.dxl.joint_states
    return list(js.position[:6]) if js and len(js.position) >= 6 else [0.0] * 6

def get_arm_gripper_position(bot):
    js = bot.bot.dxl.joint_states
    return float(js.position[6]) if js and len(js.position) > 6 else 0.0

def move_arms_smoothly(bot_list, target_list, move_time=1.0):
    DT = 1.0 / FPS
    num = int(move_time / DT)
    cur = [get_arm_joint_positions(b) for b in bot_list]
    traj = [np.linspace(c, t, num) for c, t in zip(cur, target_list)]
    for step in range(num):
        t0 = time.perf_counter()
        for i, b in enumerate(bot_list):
            b.bot.arm.set_joint_positions(traj[i][step], blocking=False)
        precise_sleep(max(DT - (time.perf_counter() - t0), 0.0))

def prep_robots(ml, mr, pl, pr):
    """初始化四臂：重启夹爪 + 设置模式 + 扭矩使能 + 回到起始位姿。"""
    # 重启从臂夹爪
    pl.bot.dxl.robot_reboot_motors("single", "gripper", True)
    pr.bot.dxl.robot_reboot_motors("single", "gripper", True)

    # 主臂设置
    for bot in [ml, mr]:
        bot.bot.dxl.robot_set_operating_modes("group", "arm", "position")
        bot.bot.dxl.robot_set_operating_modes("single", "gripper", "position")
        bot.bot.dxl.robot_torque_enable("group", "arm", True)
        bot.bot.dxl.robot_torque_enable("single", "gripper", True)

    # 从臂设置
    for bot in [pl, pr]:
        bot.bot.dxl.robot_set_operating_modes("group", "arm", "position")
        bot.bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        bot.bot.dxl.robot_torque_enable("group", "arm", True)
        bot.bot.dxl.robot_torque_enable("single", "gripper", True)

    # 所有臂移到起始位姿
    start = START_ARM_POSE[:6]
    print("Moving all arms to starting position...")
    move_arms_smoothly([ml, mr, pl, pr], [start] * 4, move_time=1.5)

    # 夹爪到中间位置
    from interbotix_xs_msgs.msg import JointSingleCommand
    targets = [MASTER_GRIPPER_JOINT_MID, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_CLOSE]
    print("Moving grippers to starting position...")
    DT = 1.0/FPS
    cur_g = [get_arm_gripper_position(b) for b in [ml, mr, pl, pr]]
    traj_g = [np.linspace(c, t, int(0.5/DT)) for c, t in zip(cur_g, targets)]
    for step in range(int(0.5/DT)):
        for i, b in enumerate([ml, mr, pl, pr]):
            cmd = JointSingleCommand(name="gripper")
            cmd.cmd = traj_g[i][step]
            b.bot.gripper.core.pub_single.publish(cmd)
        time.sleep(DT)
    print("Robots ready.")


def press_to_start(ml, mr):
    """合拢主臂夹爪开始录制。"""
    ml.bot.dxl.robot_torque_enable("single", "gripper", False)
    mr.bot.dxl.robot_torque_enable("single", "gripper", False)
    print("\n合拢两侧主臂夹爪开始录制...")
    threshold = 0.15
    done_l, done_r = False, False
    while not (done_l and done_r):
        if not done_l:
            pos = get_arm_gripper_position(ml)
            print(f"\r  左主臂夹爪: {pos:.4f}  {'' if pos > threshold else '(已闭合!)'}", end="")
            if pos > threshold:
                done_l = True
        if not done_r:
            pos = get_arm_gripper_position(mr)
            if pos > threshold:
                done_r = True
        time.sleep(0.05)
    print("\n开始录制!")


def main():
    print("=" * 55)
    print("ALOHA2 ROS 录制")
    print(f"Dataset: {HF_REPO_ID}")
    print(f"Episodes: {NUM_EPISODES}")
    print("=" * 55)

    # 创建主臂（遥操作器）和从臂（机器人）
    master = AlohaTeleopRos(AlohaTeleopRosConfig(
        id="aloha_master",
        left_arm_robot_name="master_left", right_arm_robot_name="master_right",
        robot_model="wx250s", init_ros_node=True,
    ))
    puppet = AlohaRos(AlohaRosConfig(
        id="aloha_puppet",
        puppet_left_robot_name="puppet_left", puppet_right_robot_name="puppet_right",
        robot_model="vx300s", init_ros_node=False, cameras=camera_config,
    ))

    # ── 夹爪映射处理器（实测校准） ──
    class AlohaGripperMapperStep(RobotActionProcessorStep):
        """将主臂夹爪值映射到从臂夹爪范围。"""
        def action(self, action):
            if "left_gripper.pos" in action:
                action["left_gripper.pos"] = master2puppet_left(action["left_gripper.pos"])
            if "right_gripper.pos" in action:
                action["right_gripper.pos"] = master2puppet_right(action["right_gripper.pos"])
            return action
        def transform_features(self, features):
            return features

    # 数据处理器
    teleop_ap, robot_ap_base, robot_op = make_default_processors()
    robot_ap = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[AlohaGripperMapperStep()] + list(robot_ap_base.steps),
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 数据集
    action_ft = hw_to_dataset_features(puppet.action_features, ACTION)
    obs_ft = hw_to_dataset_features(puppet.observation_features, OBS_STR)
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID, fps=FPS, features={**action_ft, **obs_ft},
        robot_type=puppet.name, use_videos=True,
        image_writer_threads=16, root=DATASET_ROOT,
    )

    # 连接
    master.connect()
    puppet.connect()
    listener, events = init_keyboard_listener()
    init_rerun(session_name="aloha2_record")

    for ep in range(NUM_EPISODES):
        prep_robots(master.left_arm, master.right_arm, puppet.left_arm, puppet.right_arm)
        press_to_start(master.left_arm, master.right_arm)

        # 主臂切 PWM 模式 + 关扭矩（遥操作）
        for side in ["left_arm", "right_arm"]:
            arm = getattr(master, side)
            arm.bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
            arm.bot.dxl.robot_torque_enable("group", "arm", False)
            arm.bot.dxl.robot_torque_enable("single", "gripper", False)

        log_say(f"Recording episode {ep+1}/{NUM_EPISODES}")
        record_loop(
            robot=puppet, events=events, fps=FPS, dataset=dataset,
            teleop=master, control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION, display_data=True,
            teleop_action_processor=teleop_ap,
            robot_action_processor=robot_ap,
            robot_observation_processor=robot_op,
        )

        if events["rerecord_episode"]:
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        if events["stop_recording"]:
            break

        dataset.save_episode()

    puppet.disconnect()
    master.disconnect()
    listener.stop()
    dataset.finalize()
    log_say(f"Dataset saved to {dataset.root}")


if __name__ == "__main__":
    main()
