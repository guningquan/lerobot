#!/usr/bin/env python
"""
Example script for using AlohaRos robot with lerobot camera system.
This uses ROS InterbotixManipulatorXS for robot control and lerobot cameras for image capture.

This example demonstrates:
- Using AlohaTeleopRos (master arms) to get actions
- Using AlohaRos (puppet arms) to send actions and get observations
"""

import time
import sys
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.aloha_ros import AlohaRos, AlohaRosConfig
from lerobot.teleoperators.aloha_teleop_ros import AlohaTeleopRos, AlohaTeleopRosConfig
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.robot_utils import precise_sleep

# Import constants from aloha_scripts/constants.py
import sys
import os
# Add aloha_scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
aloha_scripts_dir = os.path.join(script_dir, 'aloha_scripts')
if aloha_scripts_dir not in sys.path:
    sys.path.insert(0, aloha_scripts_dir)

from constants import (
    START_ARM_POSE,
    FPS
)

# ---------- 实测夹爪校准值 (2026-05-08) ----------
# master_left:  open= 0.6796, close=-0.2393
# master_right: open= 0.8468, close=-0.0537
# puppet_left:  open=-0.2562, close=-1.7871
# puppet_right: open=-0.0430, close=-1.6168
# ------------------------------------------------
MASTER_LEFT_GRIPPER_OPEN  = 0.6796
MASTER_LEFT_GRIPPER_CLOSE = -0.2393
PUPPET_LEFT_GRIPPER_OPEN  = -0.2562
PUPPET_LEFT_GRIPPER_CLOSE = -1.7871

MASTER_RIGHT_GRIPPER_OPEN  = 0.8468
MASTER_RIGHT_GRIPPER_CLOSE = -0.0537
PUPPET_RIGHT_GRIPPER_OPEN  = -0.0430
PUPPET_RIGHT_GRIPPER_CLOSE = -1.6168

def _normalize(x, lo, hi): return (x - lo) / (hi - lo)
def _unnormalize(x, lo, hi): return x * (hi - lo) + lo

def master2puppet_left(master_val):
    n = _normalize(master_val, MASTER_LEFT_GRIPPER_CLOSE, MASTER_LEFT_GRIPPER_OPEN)
    return _unnormalize(n, PUPPET_LEFT_GRIPPER_CLOSE, PUPPET_LEFT_GRIPPER_OPEN)

def master2puppet_right(master_val):
    n = _normalize(master_val, MASTER_RIGHT_GRIPPER_CLOSE, MASTER_RIGHT_GRIPPER_OPEN)
    return _unnormalize(n, PUPPET_RIGHT_GRIPPER_CLOSE, PUPPET_RIGHT_GRIPPER_OPEN)

MASTER_LEFT_GRIPPER_MID  = (MASTER_LEFT_GRIPPER_OPEN  + MASTER_LEFT_GRIPPER_CLOSE) / 2
MASTER_RIGHT_GRIPPER_MID = (MASTER_RIGHT_GRIPPER_OPEN + MASTER_RIGHT_GRIPPER_CLOSE) / 2
PUPPET_LEFT_GRIPPER_CLOSE_VAL = PUPPET_LEFT_GRIPPER_CLOSE
PUPPET_RIGHT_GRIPPER_CLOSE_VAL = PUPPET_RIGHT_GRIPPER_CLOSE

# Try to import pynput for keyboard listening
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Press Ctrl+C to exit instead of 'q'.")

# Camera configuration (same as aloha_teleop.py)
camera_config = {
    "cam_high": OpenCVCameraConfig(
        index_or_path="/dev/CAM_HIGH", 
        width=640, 
        height=480, 
        fps=FPS
    ),
    "cam_right_wrist": OpenCVCameraConfig(
        index_or_path="/dev/CAM_RIGHT_WRIST", 
        width=640, 
        height=480, 
        fps=FPS
    ),
    "cam_left_wrist": OpenCVCameraConfig(
        index_or_path="/dev/CAM_LEFT_WRIST", 
        width=640, 
        height=480, 
        fps=FPS
    ),
    "cam_low": OpenCVCameraConfig(
        index_or_path="/dev/CAM_LOW", 
        width=640, 
        height=480, 
        fps=FPS
    ),
}

# Create master arms config (teleoperators)
master_config = AlohaTeleopRosConfig(
    id="aloha_master",
    left_arm_robot_name="master_left",
    right_arm_robot_name="master_right",
    robot_model="wx250s",
    init_ros_node=True,
)

# Create puppet arms config (robots)
# Note: cameras removed because ROS usb_cam nodes already own the devices.
# For teleop-only verification, cameras are not needed.
puppet_config = AlohaRosConfig(
    id="aloha_puppet",
    puppet_left_robot_name="puppet_left",
    puppet_right_robot_name="puppet_right",
    robot_model="vx300s",
    init_ros_node=False,  # Node already initialized by master
)

# Initialize visualization
init_rerun(session_name="aloha_ros_teleop")

from lerobot.utils.errors import DeviceAlreadyConnectedError

# Create and connect master arms (teleoperators)
master = AlohaTeleopRos(master_config)
try:
    master.connect()
except DeviceAlreadyConnectedError:
    print("Master arms already connected from previous run, continuing...")

# Create and connect puppet arms (robots)
puppet = AlohaRos(puppet_config)
try:
    puppet.connect()
except DeviceAlreadyConnectedError:
    print("Puppet arms already connected from previous run, continuing...")


def get_arm_joint_positions(bot):
    """Get current arm joint positions (6 joints)."""
    joint_states = bot.bot.dxl.joint_states
    if joint_states and len(joint_states.position) >= 6:
        return list(joint_states.position[:6])
    return [0.0] * 6


def get_arm_gripper_position(bot):
    """Get current gripper joint position."""
    joint_states = bot.bot.dxl.joint_states
    if joint_states and len(joint_states.position) > 6:
        return joint_states.position[6]
    return 0.0


def move_arms_smoothly(bot_list, target_pose_list, move_time=1.0):
    """Move arms smoothly to target positions using trajectory interpolation."""
    DT = 1/30  # FPS=30
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        t0 = time.perf_counter()
        for bot_id, bot in enumerate(bot_list):
            bot.bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


def move_grippers_smoothly(bot_list, target_pose_list, move_time=0.5):
    """Move grippers smoothly to target positions using trajectory interpolation."""
    from interbotix_xs_msgs.msg import JointSingleCommand
    DT = 1/30  # FPS=30
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_position(bot) for bot in bot_list]
    traj_list = [np.linspace([curr_pose], [target_pose], num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t][0]
            bot.bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)


def prep_robots(master_left, master_right, puppet_left, puppet_right):
    """Prepare robots: set operating modes, enable torque, and move to starting position."""
    # Reboot puppet gripper motors and set operating modes
    puppet_left.bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_right.bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_left.bot.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_left.bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    puppet_right.bot.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_right.bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    
    # Set master operating modes (switch from PWM to position for initialization)
    master_left.bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_left.bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    master_right.bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_right.bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    
    # Enable torque on all robots
    master_left.bot.dxl.robot_torque_enable("group", "arm", True)
    master_left.bot.dxl.robot_torque_enable("single", "gripper", True)
    master_right.bot.dxl.robot_torque_enable("group", "arm", True)
    master_right.bot.dxl.robot_torque_enable("single", "gripper", True)
    puppet_left.bot.dxl.robot_torque_enable("group", "arm", True)
    puppet_left.bot.dxl.robot_torque_enable("single", "gripper", True)
    puppet_right.bot.dxl.robot_torque_enable("group", "arm", True)
    puppet_right.bot.dxl.robot_torque_enable("single", "gripper", True)
    
    # Move arms to starting position smoothly
    start_arm_qpos = START_ARM_POSE[:6]
    print("Moving all arms to starting position...")
    move_arms_smoothly([master_left, master_right, puppet_left, puppet_right], [start_arm_qpos] * 4, move_time=1)
    
    # Move grippers to starting position smoothly
    print("Moving grippers to starting position...")
    move_grippers_smoothly(
        [master_left, master_right, puppet_left, puppet_right],
        [MASTER_LEFT_GRIPPER_MID, MASTER_RIGHT_GRIPPER_MID, PUPPET_LEFT_GRIPPER_CLOSE_VAL, PUPPET_RIGHT_GRIPPER_CLOSE_VAL],
        move_time=0.5
    )
    
    print("Robots prepared and moved to starting position.")


def get_arm_gripper_positions(bot):
    """Get current gripper joint position."""
    if hasattr(bot, 'bot'):
        # For WidowXRos/ViperXRos wrapper
        return bot.bot.dxl.joint_states.position[6] if len(bot.bot.dxl.joint_states.position) > 6 else 0.0
    else:
        # Direct InterbotixManipulatorXS
        return bot.dxl.joint_states.position[6] if len(bot.dxl.joint_states.position) > 6 else 0.0


def press_to_start(master_left, master_right):
    """Wait for user to close both grippers to start teleoperation."""
    # Disable torque for only gripper joints of master robots to allow user movement
    master_left.bot.dxl.robot_torque_enable("single", "gripper", False)
    master_right.bot.dxl.robot_torque_enable("single", "gripper", False)
    print('Close both master grippers to start...')
    # Per-side close thresholds: left close=-0.2393, right close=-0.0537. Use 0.0 as conservative.
    close_thresh_left = 0.0
    close_thresh_right = 0.0
    pressed_left = False
    pressed_right = False
    while not (pressed_left and pressed_right):
        t1 = time.perf_counter()
        if not pressed_left:
            gripper_pos_left = get_arm_gripper_positions(master_left)
            if gripper_pos_left < close_thresh_left:
                pressed_left = True
                print("Left gripper closed!")
        if not pressed_right:
            gripper_pos_right = get_arm_gripper_positions(master_right)
            if gripper_pos_right < close_thresh_right:
                pressed_right = True
                print("Right gripper closed!")
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t1), 0.0))
    
    # Turn off master arm torque (master should be in PWM mode for teleoperation)
    master_left.bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    master_left.bot.dxl.robot_torque_enable("group", "arm", False)
    master_left.bot.dxl.robot_torque_enable("single", "gripper", False)
    master_right.bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    master_right.bot.dxl.robot_torque_enable("group", "arm", False)
    master_right.bot.dxl.robot_torque_enable("single", "gripper", False)
    print('Started! Begin teleoperation...')


# Prepare robots: move to starting position
prep_robots(master.left_arm, master.right_arm, puppet.left_arm, puppet.right_arm)

# Wait for user to press both grippers to start
press_to_start(master.left_arm, master.right_arm)

# Setup keyboard listener for 'q' key to exit
exit_flag = False

def on_press(key):
    global exit_flag
    try:
        if hasattr(key, 'char') and key.char == 'q':
            print("\n'q' key pressed. Exiting...")
            exit_flag = True
            return False  # Stop listener
    except AttributeError:
        pass

listener = None
if PYNPUT_AVAILABLE:
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Press 'q' to quit")
else:
    print("Press Ctrl+C to quit")

# Gripper commands for puppet arms
from interbotix_xs_msgs.msg import JointSingleCommand
gripper_command_left = JointSingleCommand(name="gripper")
gripper_command_right = JointSingleCommand(name="gripper")

try:
    while not exit_flag:
        t0 = time.perf_counter()
        
        # Get joint states from master arms (directly from ROS)
        master_left_joint_states = master.left_arm.bot.dxl.joint_states
        master_right_joint_states = master.right_arm.bot.dxl.joint_states
        
        # Sync joint positions from master to puppet (left arm)
        if master_left_joint_states and len(master_left_joint_states.position) >= 6:
            master_left_state_joints = list(master_left_joint_states.position[:6])
            puppet.left_arm.bot.arm.set_joint_positions(master_left_state_joints, blocking=False)
        
        # Sync joint positions from master to puppet (right arm)
        if master_right_joint_states and len(master_right_joint_states.position) >= 6:
            master_right_state_joints = list(master_right_joint_states.position[:6])
            puppet.right_arm.bot.arm.set_joint_positions(master_right_state_joints, blocking=False)
        
        # Sync gripper positions (with mapping function) - left arm
        if master_left_joint_states and len(master_left_joint_states.position) > 6:
            master_left_gripper_joint = master_left_joint_states.position[6]
            puppet_left_gripper_joint_target = master2puppet_left(master_left_gripper_joint)
            gripper_command_left.cmd = puppet_left_gripper_joint_target
            puppet.left_arm.bot.gripper.core.pub_single.publish(gripper_command_left)
        
        # Sync gripper positions (with mapping function) - right arm
        if master_right_joint_states and len(master_right_joint_states.position) > 6:
            master_right_gripper_joint = master_right_joint_states.position[6]
            puppet_right_gripper_joint_target = master2puppet_right(master_right_gripper_joint)
            gripper_command_right.cmd = puppet_right_gripper_joint_target
            puppet.right_arm.bot.gripper.core.pub_single.publish(gripper_command_right)
        
        # Get observations from puppet arms (robots)
        obs = puppet.get_observation()
        
        # Create action dict for logging (from master joint states)
        action = {}
        if master_left_joint_states and len(master_left_joint_states.position) >= 6:
            action["left_waist.pos"] = master_left_joint_states.position[0]
            action["left_shoulder.pos"] = master_left_joint_states.position[1]
            action["left_elbow.pos"] = master_left_joint_states.position[2]
            action["left_forearm_roll.pos"] = master_left_joint_states.position[3]
            action["left_wrist_angle.pos"] = master_left_joint_states.position[4]
            action["left_wrist_rotate.pos"] = master_left_joint_states.position[5]
            if len(master_left_joint_states.position) > 6:
                action["left_gripper.pos"] = master_left_joint_states.position[6]
        
        if master_right_joint_states and len(master_right_joint_states.position) >= 6:
            action["right_waist.pos"] = master_right_joint_states.position[0]
            action["right_shoulder.pos"] = master_right_joint_states.position[1]
            action["right_elbow.pos"] = master_right_joint_states.position[2]
            action["right_forearm_roll.pos"] = master_right_joint_states.position[3]
            action["right_wrist_angle.pos"] = master_right_joint_states.position[4]
            action["right_wrist_rotate.pos"] = master_right_joint_states.position[5]
            if len(master_right_joint_states.position) > 6:
                action["right_gripper.pos"] = master_right_joint_states.position[6]
        
        # Print joint positions (less frequently to reduce output)
        # Uncomment if you want to see detailed output
        # print("=" * 60)
        # print("MASTER LEFT ARM (Action):")
        # for key, value in action.items():
        #     if key.startswith("left_") and key.endswith(".pos"):
        #         print(f"  {key:25}: {value:8.3f}")
        # 
        # print("\nMASTER RIGHT ARM (Action):")
        # for key, value in action.items():
        #     if key.startswith("right_") and key.endswith(".pos"):
        #         print(f"  {key:25}: {value:8.3f}")
        # 
        # print("\nPUPPET LEFT ARM (Observation):")
        # for key, value in obs.items():
        #     if key.startswith("left_") and key.endswith(".pos"):
        #         print(f"  {key:25}: {value:8.3f}")
        # 
        # print("\nPUPPET RIGHT ARM (Observation):")
        # for key, value in obs.items():
        #     if key.startswith("right_") and key.endswith(".pos"):
        #         print(f"  {key:25}: {value:8.3f}")
        # print("=" * 60)
        
        # Log to rerun
        log_rerun_data(obs, action)
        
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
        
except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    # Disconnect robots
    print("Disconnecting robots...")
    if master.is_connected:
        master.disconnect()
    if puppet.is_connected:
        puppet.disconnect()
    if PYNPUT_AVAILABLE and listener is not None and listener.is_alive():
        listener.stop()
    print("All devices disconnected. Exiting.")

