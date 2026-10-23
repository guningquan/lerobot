#!/usr/bin/env python
"""
Bimanual ALOHA2 teleoperation (no recording) with AlohaTeleopRos (master arms) and AlohaRos (puppet
arms). Arm control follows ALOHA2's act-triple-plus-aloha2/aloha_scripts/one_side_teleop.py.

Before running, start the arms and cameras and keep that terminal open:
    roslaunch aloha 4arms_tactile_teleop.launch

    python examples/aloha_ros/aloha_ros_teleop.py --fps 30     (or --fps 50)

Keys:
    → (Right arrow)  start, and resume after a pause
    ← (Left arrow)   pause and lock: the puppets stop following, the masters are held rigid
    ESC (or q)       stop: all four arms go to the rest pose and power off
"""

import argparse
import os
import sys
import time

from lerobot.cameras.ros_image import RosImageCameraConfig
from lerobot.robots.aloha_ros import AlohaRos, AlohaRosConfig
from lerobot.teleoperators.aloha_teleop_ros import AlohaTeleopRos, AlohaTeleopRosConfig
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

# Gripper opening masters and puppets park at before teleoperation (0.0 = closed, 1.0 = open)
START_OPEN_RATIO = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fps", type=int, default=FPS, help="Control rate in Hz, e.g. 30 or 50 (default: %(default)s)."
    )
    args = parser.parse_args()
    fps = args.fps

    # Cameras, read from the topics the launch publishes
    camera_config = {
        name: RosImageCameraConfig(topic=topic, width=ROS_CAMERA_WIDTH, height=ROS_CAMERA_HEIGHT, fps=fps)
        for name, topic in ROS_CAMERA_TOPICS.items()
    }

    # Create master arms config (teleoperators); released after the start key, not on connect
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
        cameras=camera_config,
    )

    # Initialize visualization
    init_rerun(session_name="aloha_ros_teleop")

    # Create and connect master arms (teleoperators) and puppet arms (robots)
    master = AlohaTeleopRos(master_config)
    puppet = AlohaRos(puppet_config)
    master.connect()
    puppet.connect()

    listener, events = init_keyboard_listener()
    try:
        # Prepare robots: all four arms to the start pose at once
        print("Moving all arms to the start pose...")
        run_parallel(master.move_to_start, puppet.move_to_start)

        paused = True
        if listener is None:
            print("No keyboard listener available: starting right away, Ctrl+C to stop.")
            master.unlock()
            puppet.begin_blend()
            paused = False
        else:
            print("Press → (Right arrow) to start teleoperation, ESC to quit.")

        while not events["stop_recording"]:
            t0 = time.perf_counter()

            if events["exit_early"]:
                if events["rerecord_episode"]:
                    # ← pause and lock
                    if not paused:
                        master.lock()
                        paused = True
                        print("Paused and locked (masters rigid). → to resume, ESC to stop and power off.")
                elif paused:
                    # → start / resume: hand the masters back, ease the puppets onto them
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

            # Get observations from puppet arms (robots) and log to rerun
            obs = puppet.get_observation()
            log_rerun_data(observation=obs, action=action)

            precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Hold the masters, then walk all four arms to the rest pose and power them off
        print("Moving all arms to the rest pose and powering off...")
        master.lock()
        run_parallel(puppet.disconnect, master.disconnect)
        if listener is not None:
            listener.stop()
        print("All devices disconnected. Exiting.")


if __name__ == "__main__":
    main()
