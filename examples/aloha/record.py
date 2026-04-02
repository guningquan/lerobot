#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ALOHA ROS Bimanual Recording Script

This script records episodes using ALOHA ROS dual-arm system (ViperX followers + WidowX leaders via ROS).

Usage:
1. Configure the robot and teleoperator settings below
2. Set your HuggingFace repository ID
3. Run the script to start recording
"""

import sys
import os
import time
import threading
from pathlib import Path
from typing import Any
import numpy as np
from tqdm import tqdm

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.processor.pipeline import RobotActionProcessorStep, RobotProcessorPipeline
from lerobot.robots.aloha_ros import AlohaRos, AlohaRosConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.aloha_teleop_ros import AlohaTeleopRos, AlohaTeleopRosConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.robot_utils import precise_sleep
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action

# Import constants from aloha_scripts/constants.py
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
    FPS
)


# Recording configuration
NUM_EPISODES = 2
EPISODE_TIME_SEC = 20
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "guningquan2025/aloha_ros_record"

# Dataset root directory (optional)
# If None, dataset will be saved to ~/.cache/huggingface/lerobot/{HF_REPO_ID}
# You can specify a custom path like: DATASET_ROOT = "/path/to/your/dataset"

DATASET_ROOT = '/home/robot/Dataset_and_Checkpoint/lerobot-dataset/testlerobot'  # Set to None for default location, or specify a Path/str

# Camera configuration (same as aloha_ros_teleop.py)
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


class AlohaGripperMapperStep(RobotActionProcessorStep):
    """Maps master gripper joint values to puppet gripper joint values using MASTER2PUPPET_JOINT_FN."""
    
    def __init__(self):
        super().__init__()
        self.master2puppet_fn = MASTER2PUPPET_JOINT_FN
    
    def action(self, action: dict[str, float]) -> dict[str, float]:
        """Map master gripper values to puppet gripper values."""
        # Map left gripper
        if "left_gripper.pos" in action:
            action["left_gripper.pos"] = self.master2puppet_fn(action["left_gripper.pos"])
        
        # Map right gripper
        if "right_gripper.pos" in action:
            action["right_gripper.pos"] = self.master2puppet_fn(action["right_gripper.pos"])
        
        return action
    
    def transform_features(
        self, features: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Features remain unchanged, only values are mapped."""
        return features

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


def get_arm_gripper_positions(bot):
    """Get current gripper joint position (alias for compatibility)."""
    return get_arm_gripper_position(bot)


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
        t0 = time.perf_counter()
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t][0]
            bot.bot.gripper.core.pub_single.publish(gripper_command)
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


def prep_robots(master_left, master_right, puppet_left, puppet_right):
    """Prepare robots: set operating modes, enable torque, and move to starting position."""
    # Reboot puppet gripper motors and set operating modes
    puppet_left.bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_right.bot.dxl.robot_reboot_motors("single", "gripper", True)

    # puppet_left.bot.dxl.robot_set_operating_modes("group", "arm", "position") # drop position mode to position mode, comment by @gnq
    # puppet_left.bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    # puppet_right.bot.dxl.robot_set_operating_modes("group", "arm", "position")
    # puppet_right.bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    
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
        [MASTER_GRIPPER_JOINT_MID, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_CLOSE],
        move_time=0.5
    )
    
    print("Robots prepared and moved to starting position.")



def press_to_start(master_left, master_right):
    """Wait for user to close both grippers to start teleoperation."""
    # Disable torque for only gripper joints of master robots to allow user movement
    master_left.bot.dxl.robot_torque_enable("single", "gripper", False)
    master_right.bot.dxl.robot_torque_enable("single", "gripper", False)
    print('Close both master grippers to start...')
    close_thresh = -0.3
    pressed_left = False
    pressed_right = False
    while not (pressed_left and pressed_right):
        t1 = time.perf_counter()
        if not pressed_left:
            gripper_pos_left = get_arm_gripper_positions(master_left)
            if gripper_pos_left < close_thresh:
                pressed_left = True
                print("Left gripper closed!")
        if not pressed_right:
            gripper_pos_right = get_arm_gripper_positions(master_right)
            if gripper_pos_right < close_thresh:
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


def main():
    # Create master arms config (teleoperators) - same naming as aloha_ros_teleop.py
    master_config = AlohaTeleopRosConfig(
        id="aloha_master",
        left_arm_robot_name="master_left",
        right_arm_robot_name="master_right",
        robot_model="wx250s",
        init_ros_node=True,
    )

    # Create puppet arms config (robots) - same naming as aloha_ros_teleop.py
    puppet_config = AlohaRosConfig(
        id="aloha_puppet",
        puppet_left_robot_name="puppet_left",
        puppet_right_robot_name="puppet_right",
        robot_model="vx300s",
        init_ros_node=False,  # Node already initialized by master
        cameras=camera_config,
    )

    # Initialize the robot and teleoperator - same naming as aloha_ros_teleop.py
    master = AlohaTeleopRos(master_config)
    puppet = AlohaRos(puppet_config)

    # Create processors
    teleop_action_processor, robot_action_processor_base, robot_observation_processor = make_default_processors()
    
    # Add gripper mapper to robot_action_processor
    # This maps master gripper joint values to puppet gripper joint values

    gripper_mapper = AlohaGripperMapperStep()
    robot_action_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[gripper_mapper] + list(robot_action_processor_base.steps),
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Configure the dataset features
    action_features = hw_to_dataset_features(puppet.action_features, ACTION)
    obs_features = hw_to_dataset_features(puppet.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # Create or load existing dataset
    # Dataset will be saved to: DATASET_ROOT/{HF_REPO_ID} if DATASET_ROOT is specified
    # Otherwise: ~/.cache/huggingface/lerobot/{HF_REPO_ID}
    # Videos will be saved in: {dataset_root}/videos/observation.images.{camera_name}/chunk-{N}/file-{M}.mp4
    
    # Check if dataset already exists
    # dataset_path = Path(DATASET_ROOT) / HF_REPO_ID if DATASET_ROOT else None
    dataset_path = Path(DATASET_ROOT) if DATASET_ROOT else None
    dataset_exists = False
    existing_episodes = 0
    
    if dataset_path and dataset_path.exists():
        try:
            # Try to load existing dataset to check episode count
            existing_dataset = LeRobotDataset(
                repo_id=HF_REPO_ID,
                root=DATASET_ROOT,
                download_videos=False,  # Don't download videos when checking
            )
            existing_episodes = existing_dataset.num_episodes
            dataset_exists = True
            print(f"\n{'='*60}")
            print(f"EXISTING DATASET FOUND!")
            print(f"{'='*60}")
            print(f"Dataset location: {existing_dataset.root}")
            print(f"Already recorded episodes: {existing_episodes}")
            print(f"Total frames: {existing_dataset.num_frames}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Could not load existing dataset: {e}")
            print("Creating new dataset...")
            dataset_exists = False
    
    # Create or continue dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=puppet.name,
        use_videos=True,
        image_writer_threads=4 * len(puppet.cameras) if puppet.cameras else 4,
        root=DATASET_ROOT,  # Specify custom root directory if needed
    )
    
    # If dataset existed, update episode count
    if dataset_exists:
        # Load metadata to get current episode count
        dataset.meta.load_metadata()
        existing_episodes = dataset.meta.total_episodes
        print(f"Continuing recording from episode {existing_episodes + 1}")
    
    print(f"Dataset will be saved to: {dataset.root}")
    print(f"Videos will be saved in: {dataset.root / 'videos'}")

    # Connect the robot and teleoperator
    master.connect()
    puppet.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="aloha_ros_record")

    if not puppet.is_connected or not master.is_connected:
        raise ValueError("Robot or teleoperator is not connected!")

    print("\n" + "="*60)
    print("RECORDING SETUP COMPLETE")
    print("="*60)
    print(f"Episodes to record in this session: {NUM_EPISODES}")
    if dataset_exists:
        print(f"Already recorded episodes: {existing_episodes}")
        print(f"Total episodes after this session: {existing_episodes + NUM_EPISODES}")
    print(f"Episode duration: {EPISODE_TIME_SEC} seconds")
    # print(f"Reset duration: {RESET_TIME_SEC} seconds")
    print(f"Task: {TASK_DESCRIPTION}")
    print("\nKeyboard Controls:")
    print("  → (Right Arrow): End current episode early (if task completed)")
    print("  ← (Left Arrow): End current episode and re-record it")
    print("  ESC: Stop recording completely")
    print("="*60 + "\n")

    recorded_episodes = 0
    while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
        # Prepare robots: move to starting position (every time at the beginning)
        print("\n" + "="*60)
        print("PREPARING ROBOTS - Moving to initial position...")
        print("="*60)
        prep_robots(master.left_arm, master.right_arm, puppet.left_arm, puppet.right_arm)
        print("="*60 + "\n")
        current_episode_num = existing_episodes + recorded_episodes + 1
        print(f"\n{'='*60}")
        print(f"EPISODE {recorded_episodes + 1} of {NUM_EPISODES} (Total: {current_episode_num})")
        print(f"{'='*60}")
        log_say(f"Recording episode {current_episode_num}")
        
        # Wait for user to press grippers to start recording this episode
        press_to_start(master.left_arm, master.right_arm)

        
        # Main record loop
        record_loop(
            robot=puppet,
            events=events,
            fps=FPS,
            dataset=dataset,
            teleop=master,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # # Reset the environment if not stopping or re-recording
        # if not events["stop_recording"] and (
        #     (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
        # ):  
        #     print("Reset the environment")
        #     log_say("Reset the environment")
        #     record_loop(
        #         robot=puppet,
        #         events=events,
        #         fps=FPS,
        #         teleop=master,
        #         control_time_s=RESET_TIME_SEC,
        #         single_task=TASK_DESCRIPTION,
        #         display_data=True,
        #         teleop_action_processor=teleop_action_processor,
        #         robot_action_processor=robot_action_processor,
        #         robot_observation_processor=robot_observation_processor,
        #     )

        if events["rerecord_episode"]:
            log_say("Re-record episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save episode only if not stopping recording
        if not events["stop_recording"]:
            dataset.save_episode()
            recorded_episodes += 1
        else:
            # Clear buffer if stopping recording (ESC pressed)
            dataset.clear_episode_buffer()
            print("Recording stopped. Current episode discarded.")

    # Clean up
    print("Stop recording")
    log_say("Stop recording")
    puppet.disconnect()
    master.disconnect()
    listener.stop()

    dataset.finalize()
    print(f"\nDataset saved to: {dataset.root}")
    print(f"Videos location: {dataset.root / 'videos'}")
    
    # Uncomment to push to HuggingFace Hub
    # dataset.push_to_hub()
    # log_say(f"Dataset '{HF_REPO_ID}' pushed to HuggingFace Hub")


if __name__ == "__main__":
    main()

