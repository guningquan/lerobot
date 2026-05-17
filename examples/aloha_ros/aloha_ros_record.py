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
import argparse
import threading
from pathlib import Path
from typing import Any
import numpy as np
from tqdm import tqdm

# from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.tactile.config import TactileSensorMode
from lerobot.tactile.xense_g1ws.config_xense_g1ws import XENSEG1WSConfig
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
from aloha_scripts.constants import (
    START_ARM_POSE,
    FPS,
    TASK_CONFIGS,
)

# Default task (overridden by --task CLI argument)
DEFAULT_TASK = "aloha_wear_shoe"

# Per-arm start poses: left master wrist_angle tuned down due to calibration offset
MASTER_LEFT_START  = START_ARM_POSE[:6]
MASTER_RIGHT_START = START_ARM_POSE[:6]
PUPPET_LEFT_START  = START_ARM_POSE[:6]
PUPPET_RIGHT_START = START_ARM_POSE[:6]

# ---------- 实测夹爪校准值 (2026-05-13) ----------
# master_left:  open= 0.7517, close=-0.0752
# master_right: open= 0.8191, close=-0.0552
# puppet_left:  open= 0.3068, close=-1.2717
# puppet_right: open= 0.1212, close=-1.3514
# ------------------------------------------------
MASTER_LEFT_GRIPPER_OPEN  = 0.7517
MASTER_LEFT_GRIPPER_CLOSE = -0.0752
PUPPET_LEFT_GRIPPER_OPEN  = 0.3068
PUPPET_LEFT_GRIPPER_CLOSE = -1.2717

MASTER_RIGHT_GRIPPER_OPEN  = 0.8191
MASTER_RIGHT_GRIPPER_CLOSE = -0.0552
PUPPET_RIGHT_GRIPPER_OPEN  = 0.1212
PUPPET_RIGHT_GRIPPER_CLOSE = -1.3514

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


# (dataset_root configured in main() from --task arg)

# Camera configuration (same as aloha_ros_teleop.py)
camera_config = {
    "cam_high": RealSenseCameraConfig(
        serial_number_or_name="109422062625", 
        fps=30,
        width=640, 
        height=480
    ),
    "cam_right_wrist": OpenCVCameraConfig(  # librealsense v4l2 bug on this camera, use OpenCV
        index_or_path="/dev/CAM_RIGHT_WRIST",
        fps=30,
        width=640,
        height=480,
    ),
    "cam_left_wrist": RealSenseCameraConfig(
        serial_number_or_name="134222077139", 
        fps=30,
        width=640, 
        height=480
    ),
    "cam_low": RealSenseCameraConfig(
        serial_number_or_name="936322072119",
        fps=30,
        width=640,
        height=480
    ),
}

# ── Tactile sensor configuration (4 XENSE G1-WS sensors) ────────────
TACTILE_MODE = TactileSensorMode.FULL  # 10-ch: rectify+depth+force+force_norm+marker2d
# TACTILE_MODE = TactileSensorMode.SIMPLE  # 3-ch grayscale (GelSight-equivalent)

tactile_config = {
    "left_fingertip": XENSEG1WSConfig(
        serial_number="OG000635",
        mode=TACTILE_MODE,
        fps=FPS,
        width=160, height=120,
        mock=False,
    ),
    "left_knuckle": XENSEG1WSConfig(
        serial_number="OG000707",
        mode=TACTILE_MODE,
        fps=FPS,
        width=160, height=120,
        mock=False,
    ),
    "right_fingertip": XENSEG1WSConfig(
        serial_number="OG000614",
        mode=TACTILE_MODE,
        fps=FPS,
        width=160, height=120,
        mock=False,
    ),
    "right_knuckle": XENSEG1WSConfig(
        serial_number="OG000706",
        mode=TACTILE_MODE,
        fps=FPS,
        width=160, height=120,
        mock=False,
    ),
}


class AlohaGripperMapperStep(RobotActionProcessorStep):
    """Maps master gripper joint values to puppet gripper joint values using per-side calibration."""

    def __init__(self):
        super().__init__()

    def action(self, action: dict[str, float]) -> dict[str, float]:
        """Map master gripper values to puppet gripper values."""
        if "left_gripper.pos" in action:
            action["left_gripper.pos"] = master2puppet_left(action["left_gripper.pos"])
        if "right_gripper.pos" in action:
            action["right_gripper.pos"] = master2puppet_right(action["right_gripper.pos"])
        return action

    def transform_features(
        self, features: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
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
    
    # Move arms to starting position
    print("Moving all arms to starting position...")
    move_arms_smoothly(
        [master_left, master_right, puppet_left, puppet_right],
        [MASTER_LEFT_START, MASTER_RIGHT_START, PUPPET_LEFT_START, PUPPET_RIGHT_START],
        move_time=1,
    )
    
    # Move grippers to starting position smoothly
    print("Moving grippers to starting position...")
    move_grippers_smoothly(
        [master_left, master_right, puppet_left, puppet_right],
        [MASTER_LEFT_GRIPPER_MID, MASTER_RIGHT_GRIPPER_MID, PUPPET_LEFT_GRIPPER_CLOSE_VAL, PUPPET_RIGHT_GRIPPER_CLOSE_VAL],
        move_time=0.5
    )
    
    print("Robots prepared and moved to starting position.")


def reset_after_episode(master_left, master_right, puppet_left, puppet_right):
    """Release the object, then return all arms and grippers to the next-episode ready pose."""
    print("\nResetting after episode...")
    print("Enabling position control for reset...")

    master_left.bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_left.bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    master_right.bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_right.bot.dxl.robot_set_operating_modes("single", "gripper", "position")

    master_left.bot.dxl.robot_torque_enable("group", "arm", True)
    master_left.bot.dxl.robot_torque_enable("single", "gripper", True)
    master_right.bot.dxl.robot_torque_enable("group", "arm", True)
    master_right.bot.dxl.robot_torque_enable("single", "gripper", True)
    puppet_left.bot.dxl.robot_torque_enable("group", "arm", True)
    puppet_left.bot.dxl.robot_torque_enable("single", "gripper", True)
    puppet_right.bot.dxl.robot_torque_enable("group", "arm", True)
    puppet_right.bot.dxl.robot_torque_enable("single", "gripper", True)

    print("Holding final pose for 2 seconds before releasing object...")
    time.sleep(2.0)

    print("Opening puppet grippers to release object...")
    move_grippers_smoothly(
        [puppet_left, puppet_right],
        [PUPPET_LEFT_GRIPPER_OPEN, PUPPET_RIGHT_GRIPPER_OPEN],
        move_time=0.5,
    )

    print("Moving all arms to starting position...")
    move_arms_smoothly(
        [master_left, master_right, puppet_left, puppet_right],
        [MASTER_LEFT_START, MASTER_RIGHT_START, PUPPET_LEFT_START, PUPPET_RIGHT_START],
        move_time=1,
    )

    print("Setting grippers to next-episode ready state...")
    move_grippers_smoothly(
        [master_left, master_right, puppet_left, puppet_right],
        [
            MASTER_LEFT_GRIPPER_MID,
            MASTER_RIGHT_GRIPPER_MID,
            PUPPET_LEFT_GRIPPER_CLOSE,
            PUPPET_RIGHT_GRIPPER_CLOSE,
        ],
        move_time=0.5,
    )

    print("Reset complete. Robots are ready for the next episode.")



def press_to_start(master_left, master_right):
    """Wait for user to close both grippers to start teleoperation."""
    # Disable torque for only gripper joints of master robots to allow user movement
    master_left.bot.dxl.robot_torque_enable("single", "gripper", False)
    master_right.bot.dxl.robot_torque_enable("single", "gripper", False)
    print('Close both master grippers to start...')
    close_thresh_left = 0.0
    close_thresh_right = 0.0
    pressed_left = False
    pressed_right = False
    while not (pressed_left and pressed_right):
        t1 = time.perf_counter()

        gripper_pos_left = get_arm_gripper_positions(master_left)
        gripper_pos_right = get_arm_gripper_positions(master_right)
        print(f"实时夹爪位置 - 左: {gripper_pos_left:.3f}, 右: {gripper_pos_right:.3f}", end='\r')

        if not pressed_left:
            if gripper_pos_left < close_thresh_left:
                pressed_left = True
                print("\nLeft gripper closed!")
        if not pressed_right:
            if gripper_pos_right < close_thresh_right:
                pressed_right = True
                print("\nRight gripper closed!")
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
    parser = argparse.ArgumentParser(description="ALOHA ROS bimanual recording with tactile")
    parser.add_argument("--task_name", type=str, default=DEFAULT_TASK,
                        choices=list(TASK_CONFIGS.keys()),
                        help=f"Task name from TASK_CONFIGS (default: {DEFAULT_TASK})")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override num_episodes from TASK_CONFIGS")
    parser.add_argument("--duration", type=int, default=None,
                        help="Episode duration in seconds (override episode_len/FPS)")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Starting episode index for resuming data collection")
    args = parser.parse_args()

    task_cfg = TASK_CONFIGS[args.task_name]
    task_name = args.task_name
    num_episodes = args.episodes if args.episodes is not None else task_cfg["num_episodes"]
    episode_time_sec = args.duration if args.duration is not None else int(task_cfg["episode_len"] / FPS)
    hf_repo_id = f"theodoreliu/{task_name}"
    dataset_root = f"/home/robot/Dataset_and_Checkpoint/lerobot-dataset/{hf_repo_id}"

    print(f"Task: {task_name} | Episodes: {num_episodes} | Duration: {episode_time_sec}s | Dataset: {dataset_root}")

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
        tactile_sensors=tactile_config,
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
    # Dataset will be saved to: dataset_root/{hf_repo_id} if dataset_root is specified
    # Otherwise: ~/.cache/huggingface/lerobot/{hf_repo_id}
    # Videos will be saved in: {dataset_root}/videos/observation.images.{camera_name}/chunk-{N}/file-{M}.mp4
    
    # Check if dataset already exists
    # dataset_path = Path(dataset_root) / hf_repo_id if dataset_root else None
    dataset_path = Path(dataset_root) if dataset_root else None
    existing_episodes = args.episode_idx
    
    if dataset_path and dataset_path.exists() and args.episode_idx == 0:
        try:
            existing_dataset = LeRobotDataset(
                repo_id=hf_repo_id, root=dataset_root, download_videos=False)
            existing_episodes = existing_dataset.num_episodes
            print(f"\n{'='*60}")
            print(f"EXISTING DATASET FOUND! ({existing_episodes} episodes)")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Could not load existing dataset: {e}")
            print("Creating new dataset...")

    dataset = LeRobotDataset.create(
        repo_id=hf_repo_id, fps=FPS, features=dataset_features,
        robot_type=puppet.name, use_videos=True,
        image_writer_threads=4 * len(puppet.cameras) + 4 * len(puppet.tactile_sensors),
        root=dataset_root,
    )

    if existing_episodes > 0:
        print(f"Recording from episode {existing_episodes + 1}")
    
    print(f"Dataset will be saved to: {dataset.root}")
    print(f"Videos will be saved in: {dataset.root / 'videos'}")

    # Connect the robot and teleoperator
    master.connect()
    puppet.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    # init_rerun(session_name="aloha_ros_record")  # disabled: no rerun viewer in headless mode

    if not puppet.is_connected or not master.is_connected:
        raise ValueError("Robot or teleoperator is not connected!")

    print("\n" + "="*60)
    print("RECORDING SETUP COMPLETE")
    print("="*60)
    print(f"Episodes to record in this session: {num_episodes}")
    if existing_episodes > 0:
        print(f"Already recorded episodes: {existing_episodes}")
        print(f"Total episodes after this session: {existing_episodes + num_episodes}")
    print(f"Episode duration: {episode_time_sec} seconds")
    # print(f"Reset duration: {RESET_TIME_SEC} seconds")
    print(f"Task: {task_name}")
    print("\nKeyboard Controls:")
    print("  → (Right Arrow): End current episode early (if task completed)")
    print("  ← (Left Arrow): End current episode and re-record it")
    print("  ESC: Stop recording completely")
    print("="*60 + "\n")

    print("\n" + "="*60)
    print("PREPARING ROBOTS - Moving to initial position...")
    print("="*60)
    prep_robots(master.left_arm, master.right_arm, puppet.left_arm, puppet.right_arm)
    print("="*60 + "\n")

    recorded_episodes = 0
    needs_final_reset = False
    while recorded_episodes < num_episodes and not events["stop_recording"]:
        current_episode_num = existing_episodes + recorded_episodes + 1
        print(f"\n{'='*60}")
        print(f"EPISODE {recorded_episodes + 1} of {num_episodes} (Total: {current_episode_num})")
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
            control_time_s=episode_time_sec,
            single_task=task_name,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        needs_final_reset = True

        # # Reset the environment if not stopping or re-recording
        # if not events["stop_recording"] and (
        #     (recorded_episodes < num_episodes - 1) or events["rerecord_episode"]
        # ):  
        #     print("Reset the environment")
        #     log_say("Reset the environment")
        #     record_loop(
        #         robot=puppet,
        #         events=events,
        #         fps=FPS,
        #         teleop=master,
        #         control_time_s=RESET_TIME_SEC,
        #         single_task=task_name,
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
            reset_after_episode(master.left_arm, master.right_arm, puppet.left_arm, puppet.right_arm)
            needs_final_reset = False
            continue

        # Save episode only if not stopping recording
        if not events["stop_recording"]:
            reset_after_episode(master.left_arm, master.right_arm, puppet.left_arm, puppet.right_arm)
            needs_final_reset = False
            dataset.save_episode()
            recorded_episodes += 1
        else:
            # Clear buffer if stopping recording (ESC pressed)
            dataset.clear_episode_buffer()
            print("Recording stopped. Current episode discarded.")

    # Return all arms to start pose before cleanup
    if needs_final_reset:
        print("\nReturning arms to start position...")
        reset_after_episode(master.left_arm, master.right_arm, puppet.left_arm, puppet.right_arm)
        print("Arms at start position. You can now run sleep.py.")
    else:
        print("\nArms already at start position. You can now run sleep.py.")

    # Clean up
    print("Stop recording")
    log_say("Stop recording")
    puppet.disconnect()
    master.disconnect()
    if listener is not None:
        listener.stop()

    dataset.finalize()
    print(f"\nDataset saved to: {dataset.root}")
    print(f"Videos location: {dataset.root / 'videos'}")
    
    # Uncomment to push to HuggingFace Hub
    # dataset.push_to_hub()
    # log_say(f"Dataset '{hf_repo_id}' pushed to HuggingFace Hub")


if __name__ == "__main__":
    main()
