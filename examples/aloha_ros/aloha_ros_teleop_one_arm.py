#!/usr/bin/env python
"""
Single-arm ALOHA2 teleoperation (no recording) with WidowXRos (one master arm) and ViperXRos (one
puppet arm). Arm control follows ALOHA2's act-triple-plus-aloha2/aloha_scripts/one_side_teleop.py.

Before running, start the arms and cameras and keep that terminal open:
    roslaunch aloha 4arms_tactile_teleop.launch

    python examples/aloha_ros/aloha_ros_teleop_one_arm.py --side right --fps 30

Keys:
    → (Right arrow)  start, and resume after a pause
    ← (Left arrow)   pause and lock: the puppet stops following, the master is held rigid
    ESC (or q)       stop: both arms go to the rest pose and power off
"""

import argparse
import os
import sys
import time

from lerobot.cameras.ros_image import RosImageCameraConfig
from lerobot.robots.viperx_ros import ViperXRos, ViperXRosConfig
from lerobot.teleoperators.widowx_ros import WidowXRos, WidowXRosConfig
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

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

# Gripper opening master and puppet park at before teleoperation (0.0 = closed, 1.0 = open)
START_OPEN_RATIO = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=("left", "right"), default="right", help="Which arm pair to use.")
    parser.add_argument(
        "--fps", type=int, default=FPS, help="Control rate in Hz, e.g. 30 or 50 (default: %(default)s)."
    )
    args = parser.parse_args()
    side = args.side
    fps = args.fps

    # Cameras, read from the topics the launch publishes
    camera_config = {
        name: RosImageCameraConfig(topic=topic, width=ROS_CAMERA_WIDTH, height=ROS_CAMERA_HEIGHT, fps=fps)
        for name, topic in ROS_CAMERA_TOPICS.items()
    }

    # Create master arm config (teleoperator); released after the start key, not on connect
    master_config = WidowXRosConfig(
        id=f"aloha_master_{side}",
        robot_name=f"master_{side}",
        robot_model="wx250s",
        init_ros_node=True,
        fps=fps,
        move_to_start_on_connect=False,
        start_open_ratio=START_OPEN_RATIO,
        release_on_connect=False,
    )

    # Create puppet arm config (robot)
    puppet_config = ViperXRosConfig(
        id=f"aloha_puppet_{side}",
        robot_name=f"puppet_{side}",
        robot_model="vx300s",
        init_ros_node=False,  # Node already initialized by master
        fps=fps,
        move_to_start_on_connect=False,
        start_open_ratio=START_OPEN_RATIO,
        cameras=camera_config,
    )

    # Initialize visualization
    init_rerun(session_name=f"aloha_ros_teleop_{side}")

    # Create and connect master arm (teleoperator) and puppet arm (robot)
    master = WidowXRos(master_config)
    puppet = ViperXRos(puppet_config)
    master.connect()
    puppet.connect()

    listener, events = init_keyboard_listener()
    try:
        # Prepare robots: both arms to the start pose at once
        print(f"Moving the {side} arms to the start pose...")
        run_parallel(master.move_to_start, puppet.move_to_start)

        paused = True
        if listener is None:
            print("No keyboard listener available: starting right away, Ctrl+C to stop.")
            master.unlock()
            puppet.begin_blend()
            paused = False
        else:
            print(f"Press → (Right arrow) to start {side} arm teleoperation, ESC to quit.")

        while not events["stop_recording"]:
            t0 = time.perf_counter()

            if events["exit_early"]:
                if events["rerecord_episode"]:
                    # ← pause and lock
                    if not paused:
                        master.lock()
                        paused = True
                        print("Paused and locked (master rigid). → to resume, ESC to stop and power off.")
                elif paused:
                    # → start / resume: hand the master back, ease the puppet onto it
                    master.unlock()
                    puppet.begin_blend()
                    paused = False
                    print("Teleoperating. ← = pause & lock, ESC = stop and power off.")
                events["exit_early"] = False
                events["rerecord_episode"] = False

            action = None
            if not paused:
                action = master.get_action()
                puppet.send_action(action)

            # Get observations from puppet arm (robot) and log to rerun
            obs = puppet.get_observation()
            log_rerun_data(observation=obs, action=action)

            precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Hold the master, then walk both arms to the rest pose and power them off
        print("Moving both arms to the rest pose and powering off...")
        master.lock()
        run_parallel(puppet.disconnect, master.disconnect)
        if listener is not None:
            listener.stop()
        print("All devices disconnected. Exiting.")


if __name__ == "__main__":
    main()
