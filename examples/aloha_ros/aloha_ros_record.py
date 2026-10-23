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
ALOHA ROS Bimanual Recording Script (ALOHA2)

Records episodes with the ALOHA ROS dual-arm system (ViperX followers + WidowX leaders via ROS).
Arm control follows ALOHA2's act-triple-plus-aloha2/aloha_scripts (see AlohaRos / AlohaTeleopRos).

Before running, start the arms and cameras and keep that terminal open:
    roslaunch aloha 4arms_tactile_teleop.launch

Usage:
1. Set the task and dataset settings below
2. python examples/aloha_ros/aloha_ros_record.py --fps 30     (or --fps 50)

Every episode starts with all four arms at the start pose; press → (Right arrow) to begin.
While recording:
    → (Right arrow)  end the episode early and save it
    ← (Left arrow)   discard the episode and record it again
    ESC              stop recording (the current episode is discarded)
Between episodes all arms go to the rest pose; at the end they power off.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from lerobot.cameras.ros_image import RosImageCameraConfig
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.aloha_ros import AlohaRos, AlohaRosConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.aloha_teleop_ros import AlohaTeleopRos, AlohaTeleopRosConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Import constants from aloha_scripts/constants.py
script_dir = os.path.dirname(os.path.abspath(__file__))
aloha_scripts_dir = os.path.join(script_dir, 'aloha_scripts')
if aloha_scripts_dir not in sys.path:
    sys.path.insert(0, aloha_scripts_dir)

from constants import (
    FPS,
    ROS_CAMERA_HEIGHT,
    ROS_CAMERA_TOPICS,
    ROS_CAMERA_WIDTH,
)
from robot_utils import run_parallel

# Recording configuration
NUM_EPISODES = 50
EPISODE_TIME_SEC = 15
TASK_DESCRIPTION = "Grasp an apple"
HF_REPO_ID = "guningquan2025/apple_grasping"
# HF_REPO_ID = "theodoreliu/apple_grasping"

# Dataset root directory (optional)
# If None, dataset will be saved to ~/.cache/huggingface/lerobot/{HF_REPO_ID}
# You can specify a custom path like: DATASET_ROOT = "/path/to/your/dataset"
# If it already holds a dataset, recording continues it (it must have been recorded at the same --fps).

DATASET_ROOT = '/home/robot/Dataset_and_Checkpoint/lerobot-dataset/apple_grasping'  # Set to None for default location, or specify a Path/str

# Gripper opening both masters and puppets park at before each episode (0.0 = closed, 1.0 = open)
START_OPEN_RATIO = 1.0


def make_camera_config(fps: int) -> dict[str, RosImageCameraConfig]:
    """The four cameras, read from the topics the launch publishes."""
    return {
        name: RosImageCameraConfig(topic=topic, width=ROS_CAMERA_WIDTH, height=ROS_CAMERA_HEIGHT, fps=fps)
        for name, topic in ROS_CAMERA_TOPICS.items()
    }


def wait_for_start(events: dict) -> bool:
    """Block until → (start) or ESC (quit). Returns False on ESC."""
    print("Press → (Right arrow) to start this episode, ESC to quit.")
    events["exit_early"] = False
    events["rerecord_episode"] = False
    while True:
        if events["stop_recording"]:
            return False
        if events["exit_early"]:
            started = not events["rerecord_episode"]  # ← only sets exit_early together with rerecord
            events["exit_early"] = False
            events["rerecord_episode"] = False
            if started:
                return True
        time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fps", type=int, default=FPS, help="Control and recording rate in Hz, e.g. 30 or 50 (default: %(default)s)."
    )
    args = parser.parse_args()
    fps = args.fps

    # Create master arms config (teleoperators). The script moves the arms and releases the
    # masters itself (after the start key), so connect() only sets them up.
    master_config = AlohaTeleopRosConfig(
        id="aloha_master",
        left_arm_robot_name="master_left",
        right_arm_robot_name="master_right",
        robot_model="wx250s",
        init_ros_node=True,
        fps=fps,
        move_to_start_on_connect=False,
        start_open_ratio=START_OPEN_RATIO,
        release_on_connect=False,
    )

    # Create puppet arms config (robots)
    puppet_config = AlohaRosConfig(
        id="aloha_puppet",
        puppet_left_robot_name="puppet_left",
        puppet_right_robot_name="puppet_right",
        robot_model="vx300s",
        init_ros_node=False,  # Node already initialized by master
        fps=fps,
        move_to_start_on_connect=False,
        start_open_ratio=START_OPEN_RATIO,
        cameras=make_camera_config(fps),
    )

    # Initialize the robot and teleoperator
    master = AlohaTeleopRos(master_config)
    puppet = AlohaRos(puppet_config)

    # Default (identity) processors: the master -> puppet gripper mapping happens inside AlohaRos,
    # so the dataset stores the master's normalised gripper opening as the action.
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Configure the dataset features
    action_features = hw_to_dataset_features(puppet.action_features, ACTION)
    obs_features = hw_to_dataset_features(puppet.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}
    image_writer_threads = 4 * len(puppet.cameras) if puppet.cameras else 4

    # Create or continue the dataset. Videos will be saved in:
    # {dataset_root}/videos/observation.images.{camera_name}/chunk-{N}/file-{M}.mp4
    dataset_path = Path(DATASET_ROOT) if DATASET_ROOT else None
    if dataset_path is not None and (dataset_path / "meta" / "info.json").exists():
        # LeRobotDataset.create() on an existing root would overwrite its metadata; resume() appends.
        dataset = LeRobotDataset.resume(
            HF_REPO_ID,
            root=DATASET_ROOT,
            image_writer_threads=image_writer_threads,
        )
        sanity_check_dataset_robot_compatibility(dataset, puppet, fps, dataset_features)
        existing_episodes = dataset.meta.total_episodes
        print(f"\n{'='*60}")
        print("EXISTING DATASET FOUND!")
        print(f"{'='*60}")
        print(f"Dataset location: {dataset.root}")
        print(f"Already recorded episodes: {existing_episodes}")
        print(f"Total frames: {dataset.meta.total_frames}")
        print(f"Continuing recording from episode {existing_episodes + 1}")
        print(f"{'='*60}\n")
    else:
        dataset = LeRobotDataset.create(
            repo_id=HF_REPO_ID,
            fps=fps,
            features=dataset_features,
            robot_type=puppet.name,
            use_videos=True,
            image_writer_threads=image_writer_threads,
            root=DATASET_ROOT,  # Specify custom root directory if needed
        )
        existing_episodes = 0

    print(f"Dataset will be saved to: {dataset.root}")
    print(f"Videos will be saved in: {dataset.root / 'videos'}")

    # Connect the robot and teleoperator (motor setup; the arms stay where they are)
    master.connect()
    puppet.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="aloha_ros_record")

    print("\n" + "="*60)
    print("RECORDING SETUP COMPLETE")
    print("="*60)
    print(f"Control / recording rate: {fps} Hz")
    print(f"Episodes to record in this session: {NUM_EPISODES}")
    if existing_episodes:
        print(f"Already recorded episodes: {existing_episodes}")
        print(f"Total episodes after this session: {existing_episodes + NUM_EPISODES}")
    print(f"Episode duration: {EPISODE_TIME_SEC} seconds")
    print(f"Task: {TASK_DESCRIPTION}")
    print("\nKeyboard Controls:")
    print("  → (Right Arrow): Start an episode / end the current episode early (if task completed)")
    print("  ← (Left Arrow): End current episode and re-record it")
    print("  ESC: Stop recording completely")
    print("="*60 + "\n")

    recorded_episodes = 0
    try:
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            # Move all four arms to the start pose (every time at the beginning)
            print("\n" + "="*60)
            print("PREPARING ROBOTS - Moving to the start pose...")
            print("="*60)
            run_parallel(master.move_to_start, puppet.move_to_start)
            current_episode_num = existing_episodes + recorded_episodes + 1
            print(f"\n{'='*60}")
            print(f"EPISODE {recorded_episodes + 1} of {NUM_EPISODES} (Total: {current_episode_num})")
            print(f"{'='*60}")
            log_say(f"Recording episode {current_episode_num}")

            if listener is None:
                print("No keyboard listener available: starting right away.")
            elif not wait_for_start(events):
                break

            # Hand the masters to the operator; the puppets ease onto them over start_blend_s
            master.unlock()
            puppet.begin_blend()

            # Main record loop
            record_loop(
                robot=puppet,
                events=events,
                fps=fps,
                dataset=dataset,
                teleop=master,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            # Hold the masters where they are while the episode is saved or discarded
            master.lock()

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
                break

            if recorded_episodes < NUM_EPISODES:
                print("Moving all arms to the rest pose...")
                run_parallel(master.sleep, puppet.sleep)
    finally:
        # Clean up: all four arms to the rest pose and power off
        print("Stop recording")
        log_say("Stop recording")
        run_parallel(puppet.disconnect, master.disconnect)
        if listener is not None:
            listener.stop()

        dataset.finalize()
        print(f"\nDataset saved to: {dataset.root}")
        print(f"Videos location: {dataset.root / 'videos'}")

    # Uncomment to push to HuggingFace Hub
    # dataset.push_to_hub()


if __name__ == "__main__":
    main()
