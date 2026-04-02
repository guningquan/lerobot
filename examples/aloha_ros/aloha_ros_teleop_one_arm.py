#!/usr/bin/env python
"""
Example script for using single arm AlohaRos robot with lerobot camera system.
This uses ROS InterbotixManipulatorXS for robot control and lerobot cameras for image capture.

This example demonstrates:
- Using WidowXRos (single master arm) to get actions
- Using ViperXRos (single puppet arm) to send actions and get observations
"""

import time
import sys
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.viperx_ros import ViperXRos, ViperXRosConfig
from lerobot.teleoperators.widowx_ros import WidowXRos, WidowXRosConfig
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
    MASTER_GRIPPER_JOINT_OPEN,
    MASTER_GRIPPER_JOINT_CLOSE,
    MASTER_GRIPPER_JOINT_MID,
    PUPPET_GRIPPER_JOINT_OPEN,
    PUPPET_GRIPPER_JOINT_CLOSE,
    MASTER2PUPPET_JOINT_FN,
)

# Try to import pynput for keyboard listening
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Press Ctrl+C to exit instead of 'q'.")

# Camera configuration (same as aloha_teleop.py)
FPS = 30
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

# Choose which side to use (left or right)
ARM_SIDE = "right"  # Change to "left" if you want to test left arm

# Create master arm config (teleoperator)
master_config = WidowXRosConfig(
    id=f"aloha_master_{ARM_SIDE}",
    robot_name=f"master_{ARM_SIDE}",
    robot_model="wx250s",
    init_ros_node=True,
)

# Create puppet arm config (robot)
puppet_config = ViperXRosConfig(
    id=f"aloha_puppet_{ARM_SIDE}",
    robot_name=f"puppet_{ARM_SIDE}",
    robot_model="vx300s",
    init_ros_node=False,  # Node already initialized by master
    cameras=camera_config,
)

# Initialize visualization
init_rerun(session_name=f"aloha_ros_teleop_{ARM_SIDE}")

# Create and connect master arm (teleoperator)
master = WidowXRos(master_config)
master.connect()

# Create and connect puppet arm (robot)
puppet = ViperXRos(puppet_config)
puppet.connect()


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
        for bot_id, bot in enumerate(bot_list):
            bot.bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)


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


def prep_robots(master_bot, puppet_bot):
    """Prepare robots: set operating modes, enable torque, and move to starting position."""
    # Reboot puppet gripper motors and set operating modes
    puppet_bot.bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot.bot.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot.bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    
    # Set master operating modes (switch from PWM to position for initialization)
    master_bot.bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot.bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    
    # Enable torque on both robots
    master_bot.bot.dxl.robot_torque_enable("group", "arm", True)
    master_bot.bot.dxl.robot_torque_enable("single", "gripper", True)
    puppet_bot.bot.dxl.robot_torque_enable("group", "arm", True)
    puppet_bot.bot.dxl.robot_torque_enable("single", "gripper", True)
    
    # Move arms to starting position smoothly
    start_arm_qpos = START_ARM_POSE[:6]
    print(f"Moving {ARM_SIDE} arm to starting position...")
    move_arms_smoothly([master_bot, puppet_bot], [start_arm_qpos] * 2, move_time=1.0)
    
    # Move grippers to starting position smoothly
    print(f"Moving grippers to starting position...")
    move_grippers_smoothly([master_bot, puppet_bot], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)
    
    print("Robots prepared and moved to starting position.")


def get_arm_gripper_positions(bot):
    """Get current gripper joint position."""
    if hasattr(bot, 'bot'):
        # For WidowXRos/ViperXRos wrapper
        return bot.bot.dxl.joint_states.position[6] if len(bot.bot.dxl.joint_states.position) > 6 else 0.0
    else:
        # Direct InterbotixManipulatorXS
        return bot.dxl.joint_states.position[6] if len(bot.dxl.joint_states.position) > 6 else 0.0


def press_to_start(master_bot):
    """Wait for user to close gripper to start teleoperation."""
    # Disable torque for only gripper joint of master robot to allow user movement
    master_bot.bot.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the {ARM_SIDE} master gripper to start...')
    close_thresh = -0.3
    pressed = False
    while not pressed:
        t1 = time.perf_counter()
        gripper_pos = get_arm_gripper_positions(master_bot)
        if gripper_pos < close_thresh:
            pressed = True
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t1), 0.0))
    
    # Turn off master arm torque (master should be in PWM mode for teleoperation)
    master_bot.bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    master_bot.bot.dxl.robot_torque_enable("group", "arm", False)
    master_bot.bot.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Started! Begin teleoperation...')


# Prepare robots: move to starting position
prep_robots(master, puppet)

# Wait for user to press gripper to start
press_to_start(master)

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

print(f"Starting single arm teleoperation for {ARM_SIDE} arm...")
print("Move the master arm to control the puppet arm.")

# Gripper command for puppet
from interbotix_xs_msgs.msg import JointSingleCommand
gripper_command = JointSingleCommand(name="gripper")

try:
    while not exit_flag:
        t0 = time.perf_counter()
        # Get joint states from master arm (directly from ROS)
        master_joint_states = master.bot.dxl.joint_states
        
        # Sync joint positions from master to puppet
        if master_joint_states and len(master_joint_states.position) >= 6:
            master_state_joints = list(master_joint_states.position[:6])
            puppet.bot.arm.set_joint_positions(master_state_joints, blocking=False)
        
        # Sync gripper positions (with mapping function)
        if master_joint_states and len(master_joint_states.position) > 6:
            master_gripper_joint = master_joint_states.position[6]
            puppet_gripper_joint_target = MASTER2PUPPET_JOINT_FN(master_gripper_joint)
            gripper_command.cmd = puppet_gripper_joint_target
            puppet.bot.gripper.core.pub_single.publish(gripper_command)
        
        # Get observations from puppet arm (robot)
        obs = puppet.get_observation()
        
        # Create action dict for logging (from master joint states)
        action = {}
        if master_joint_states and len(master_joint_states.position) >= 6:
            action["waist.pos"] = master_joint_states.position[0]
            action["shoulder.pos"] = master_joint_states.position[1]
            action["elbow.pos"] = master_joint_states.position[2]
            action["forearm_roll.pos"] = master_joint_states.position[3]
            action["wrist_angle.pos"] = master_joint_states.position[4]
            action["wrist_rotate.pos"] = master_joint_states.position[5]
            if len(master_joint_states.position) > 6:
                action["gripper.pos"] = master_joint_states.position[6]
        
        # Print joint positions (less frequently to reduce output)
        # Uncomment if you want to see detailed output
        # print("=" * 60)
        # print(f"MASTER {ARM_SIDE.upper()} ARM (Action):")
        # for key, value in action.items():
        #     if key.endswith(".pos"):
        #         print(f"  {key:25}: {value:8.3f}")
        # 
        # print(f"\nPUPPET {ARM_SIDE.upper()} ARM (Observation):")
        # for key, value in obs.items():
        #     if key.endswith(".pos"):
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