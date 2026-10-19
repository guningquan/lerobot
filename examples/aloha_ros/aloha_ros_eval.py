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
ALOHA ROS Bimanual Evaluation Script

This script evaluates a trained policy on ALOHA ROS dual-arm system (ViperX followers via ROS).

Usage:
1. Configure the robot settings and policy path below
2. Run the script to start evaluation
"""

import sys
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.robots.aloha_ros import AlohaRos, AlohaRosConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.utils import get_safe_torch_device, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.robot_utils import precise_sleep
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action

# Import constants from aloha_scripts/constants.py
script_dir = os.path.dirname(os.path.abspath(__file__))
aloha_scripts_dir = os.path.join(script_dir, 'aloha_scripts')
if aloha_scripts_dir not in sys.path:
    sys.path.insert(0, aloha_scripts_dir)

from constants import (
    START_ARM_POSE,
    PUPPET_GRIPPER_JOINT_CLOSE,
    FPS
)


# Evaluation configuration
NUM_EPISODES = 5
EPISODE_TIME_SEC = 30
POLICY_PATH = "/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint/pi05_apple_grasping_theodore/pretrained_model"
DATASET_REPO_ID = "/home/robot/Dataset_and_Checkpoint/lerobot-dataset/apple_grasping_pi05"  # For loading dataset stats

# Policy configuration overrides (optional)
# Set to None to use values from config.json, or set specific values to override
POLICY_TEMPORAL_ENSEMBLE_COEFF = None  # e.g., 0.01 to enable temporal aggregation
POLICY_N_ACTION_STEPS = None  # e.g., 1 (required if temporal_ensemble_coeff is set)

# Video saving configuration
SAVE_VIDEOS = True  # Set to True to save videos of evaluation episodes
VIDEO_OUTPUT_DIR = POLICY_PATH  # Directory to save videos

# Camera configuration
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
    DT = 1.0 / FPS
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):
        t0 = time.perf_counter()
        for bot_id, bot in enumerate(bot_list):
            bot.bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


def move_grippers_smoothly(bot_list, target_pose_list, move_time=0.5):
    """Move grippers smoothly to target positions using trajectory interpolation."""
    from interbotix_xs_msgs.msg import JointSingleCommand
    DT = 1.0 / FPS
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_position(bot) for bot in bot_list]
    traj_list = [
        np.linspace([curr_pose], [target_pose], num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):
        t0 = time.perf_counter()
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t][0]
            bot.bot.gripper.core.pub_single.publish(gripper_command)
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


def prep_robots(puppet_left, puppet_right):
    """Prepare robots: set operating modes, enable torque, and move to starting position."""
    # Reboot puppet gripper motors
    puppet_left.bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_right.bot.dxl.robot_reboot_motors("single", "gripper", True)
    
    # Enable torque on all robots
    puppet_left.bot.dxl.robot_torque_enable("group", "arm", True)
    puppet_left.bot.dxl.robot_torque_enable("single", "gripper", True)
    puppet_right.bot.dxl.robot_torque_enable("group", "arm", True)
    puppet_right.bot.dxl.robot_torque_enable("single", "gripper", True)
    
    # Move arms to starting position smoothly
    start_arm_qpos = START_ARM_POSE[:6]
    print("Moving all arms to starting position...")
    move_arms_smoothly([puppet_left, puppet_right], [start_arm_qpos] * 2, move_time=1.0)
    
    # Move grippers to starting position smoothly
    print("Moving grippers to starting position...")
    move_grippers_smoothly(
        [puppet_left, puppet_right],
        [PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_CLOSE],
        move_time=0.5
    )
    
    print("Robots prepared and moved to starting position.")


def wait_for_enter_key():
    """Wait for user to press Enter key to start evaluation."""
    print("\n" + "="*60)
    print("Press Enter to start evaluation...")
    print("="*60)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        # Handle case where stdin is not available
        print("Could not read from stdin, proceeding automatically...")
        time.sleep(2)


def eval_loop(
    robot: AlohaRos,
    policy: PreTrainedPolicy,
    preprocessor: Any,
    postprocessor: Any,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    control_time_s: float,
    fps: int,
    events: dict,
    display_data: bool = True,
    save_video: bool = False,
    video_writers: dict[str, cv2.VideoWriter] | None = None,
    episode_index: int = 0,
):
    """Main evaluation loop that runs policy on robot."""
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    
    timestamp = 0
    start_episode_t = time.perf_counter()
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        
        if events["exit_early"]:
            events["exit_early"] = False
            break
        
        # Get robot observation
        obs = robot.get_observation()
        
        # Process observation
        obs_processed = robot_observation_processor(obs)
        
        # Build observation frame for policy
        from lerobot.datasets.utils import build_dataset_frame
        observation_frame = build_dataset_frame(
            {**hw_to_dataset_features(robot.observation_features, OBS_STR)},
            obs_processed,
            prefix=OBS_STR
        )
        
        # Get action from policy
        device = get_safe_torch_device(policy.config.device)
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=None,
            robot_type=robot.robot_type,
        )
        
        # Convert to robot action format
        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        act_processed = make_robot_action(action_values, action_features)
        
        # Process action through robot action processor
        act_final = robot_action_processor((act_processed, obs_processed))
        
        
        print(f"Action: {act_final}")
        # === 安全的夹爪二值化 ===
        THRESHOLD = 0.65 
        
        # 假设真实夹住苹果时的位置大约是 0.25
        SAFE_CLOSE_POS = 0.22  # 留一点余量产生抓力，但不要设为 -0.05 这种极限值
        SAFE_OPEN_POS = 0.85

        if act_final['right_gripper.pos'] < THRESHOLD:
            act_final['right_gripper.pos'] = SAFE_CLOSE_POS 
        else:
            act_final['right_gripper.pos'] = SAFE_OPEN_POS

        # 如果用左手同理
        if act_final['left_gripper.pos'] < THRESHOLD:
            act_final['left_gripper.pos'] = SAFE_CLOSE_POS
        else:
            act_final['left_gripper.pos'] = SAFE_OPEN_POS
        # =================================
        # input("Press Enter to continue...")
        
        # Send action to robot
        robot.send_action(act_final)
        
        # # Log to rerun if enabled
        # if display_data:
        #     log_rerun_data(obs_processed, act_final, timestamp)
        
        # Save video frames if enabled
        if save_video and video_writers is not None:
            # Save frames from all cameras
            for cam_name in robot.cameras.keys():
                if cam_name in obs_processed:
                    frame = obs_processed[cam_name]
                    # Convert to BGR if needed (OpenCV uses BGR)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # Assume RGB, convert to BGR
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    if cam_name in video_writers:
                        video_writers[cam_name].write(frame_bgr)
        
        # Sleep to maintain fps
        elapsed = time.perf_counter() - start_loop_t
        sleep_time = max(1.0 / fps - elapsed, 0.0)
        precise_sleep(sleep_time)
        
        timestamp = time.perf_counter() - start_episode_t


def main():
    print("\n" + "="*60)
    print("ALOHA ROS EVALUATION SCRIPT")
    print("="*60)
    print(f"Policy path: {POLICY_PATH}")
    print(f"Number of episodes: {NUM_EPISODES}")
    print(f"Episode duration: {EPISODE_TIME_SEC} seconds")
    print("="*60 + "\n")
    
    # Create puppet arms config (robots)
    puppet_config = AlohaRosConfig(
        id="aloha_puppet",
        puppet_left_robot_name="puppet_left",
        puppet_right_robot_name="puppet_right",
        robot_model="vx300s",
        init_ros_node=True,
        cameras=camera_config,
    )
    
    # Initialize the robot
    puppet = AlohaRos(puppet_config)
    
    # Load dataset to get metadata for policy initialization
    print("Loading dataset for metadata...")
    try:
        dataset = LeRobotDataset(
            repo_id=DATASET_REPO_ID,
            download_videos=False,
        )
        dataset_meta = dataset.meta
        print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        print("Policy will be initialized without dataset metadata (may cause issues)")
        dataset_meta = None
    
    # Load policy
    print(f"\nLoading policy from {POLICY_PATH}...")
    from lerobot.configs.policies import PreTrainedConfig
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.pretrained_path = Path(POLICY_PATH)
    
    # Apply policy configuration overrides if specified
    if POLICY_TEMPORAL_ENSEMBLE_COEFF is not None:
        policy_cfg.temporal_ensemble_coeff = POLICY_TEMPORAL_ENSEMBLE_COEFF
        print(f"Overriding temporal_ensemble_coeff: {POLICY_TEMPORAL_ENSEMBLE_COEFF}")
    
    if POLICY_N_ACTION_STEPS is not None:
        policy_cfg.n_action_steps = POLICY_N_ACTION_STEPS
        print(f"Overriding n_action_steps: {POLICY_N_ACTION_STEPS}")
    
    # # Validate configuration after overrides
    # if policy_cfg.temporal_ensemble_coeff is not None and policy_cfg.n_action_steps > 1:
    #     raise ValueError(
    #         "`n_action_steps` must be 1 when using temporal ensembling. "
    #         f"Current values: temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff}, "
    #         f"n_action_steps={policy_cfg.n_action_steps}. "
    #         "Please set POLICY_N_ACTION_STEPS=1 if enabling temporal aggregation."
    #     )
    
    # Make policy
    from lerobot.policies.factory import make_policy
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=dataset_meta,
    )
    policy.eval()
    print(f"Policy loaded: {policy_cfg.type}")
    print(f"Device: {policy.config.device}")
    
    # Create processors
    _, robot_action_processor_base, robot_observation_processor = make_default_processors()
    robot_action_processor = robot_action_processor_base
    
    # Load preprocessor and postprocessor from policy
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=dataset_meta.stats if dataset_meta else None,
        preprocessor_overrides={
            "device_processor": {"device": policy.config.device},
        },
    )
    
    # Connect the robot
    print("\nConnecting to robot...")
    puppet.connect()
    
    if not puppet.is_connected:
        raise ValueError("Robot is not connected!")
    
    # Initialize keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="aloha_ros_eval")
    
    print("\n" + "="*60)
    print("EVALUATION SETUP COMPLETE")
    print("="*60)
    print(f"Episodes to evaluate: {NUM_EPISODES}")
    print(f"Episode duration: {EPISODE_TIME_SEC} seconds")
    print("\nKeyboard Controls:")
    print("  → (Right Arrow): End current episode early")
    print("  ESC: Stop evaluation completely")
    print("="*60 + "\n")
    
    # Prepare robots: move to initial position
    print("\n" + "="*60)
    print("PREPARING ROBOTS - Moving to initial position...")
    print("="*60)
    prep_robots(puppet.left_arm, puppet.right_arm)
    print("="*60 + "\n")
    
    # Wait for user to press Enter
    wait_for_enter_key()
    
    # Create video output directory if saving videos
    video_writers = None
    if SAVE_VIDEOS:
        video_output_path = Path(VIDEO_OUTPUT_DIR)
        video_output_path.mkdir(parents=True, exist_ok=True)
        print(f"Videos will be saved to: {video_output_path}")
    
    # Run evaluation episodes
    evaluated_episodes = 0
    while evaluated_episodes < NUM_EPISODES and not events["stop_recording"]:
        current_episode_num = evaluated_episodes + 1
        print(f"\n{'='*60}")
        print(f"EPISODE {current_episode_num} of {NUM_EPISODES}")
        print(f"{'='*60}")
        log_say(f"Evaluating episode {current_episode_num}")
        
        # Initialize video writers for this episode if saving videos
        if SAVE_VIDEOS:
            video_writers = {}
            for cam_name in puppet.cameras.keys():
                video_path = video_output_path / f"episode_{current_episode_num:03d}_{cam_name}.mp4"
                # Get camera dimensions
                cam_config = puppet.config.cameras[cam_name]
                width = cam_config.width
                height = cam_config.height
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    float(FPS),
                    (width, height)
                )
                if not video_writer.isOpened():
                    print(f"Warning: Could not open video writer for {cam_name}")
                else:
                    video_writers[cam_name] = video_writer
                    print(f"Recording video for {cam_name}: {video_path}")
        
        # Run evaluation loop
        eval_loop(
            robot=puppet,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            control_time_s=EPISODE_TIME_SEC,
            fps=FPS,
            events=events,
            display_data=True,
            save_video=SAVE_VIDEOS,
            video_writers=video_writers,
            episode_index=current_episode_num,
        )
        
        # Close video writers for this episode
        if SAVE_VIDEOS and video_writers is not None:
            for cam_name, writer in video_writers.items():
                writer.release()
                print(f"Video saved for {cam_name}")
            video_writers = None
        
        evaluated_episodes += 1
        
        # Brief pause between episodes
        if evaluated_episodes < NUM_EPISODES and not events["stop_recording"]:
            print(f"\nEpisode {current_episode_num} completed.")
            print("Preparing for next episode...")
            time.sleep(2)
            
            # Move back to initial position for next episode
            print("Moving robots back to initial position...")
            prep_robots(puppet.left_arm, puppet.right_arm)
    
    # Clean up
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Evaluated {evaluated_episodes} episodes")
    print("="*60 + "\n")
    
    log_say("Evaluation complete")
    puppet.disconnect()
    if listener:
        listener.stop()


if __name__ == "__main__":
    main()

