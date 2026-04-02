#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
import numpy as np
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_viperx_ros import ViperXRosConfig

logger = logging.getLogger(__name__)

# Import constants from aloha_scripts/constants.py
import sys
import os

# Add aloha_scripts directory to path
# __file__ is at: src/lerobot/robots/viperx_ros/viperx_ros.py
# Need to go up 5 levels to reach lerobot root
try:
    current_file = os.path.abspath(__file__)
    lerobot_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
    aloha_scripts_dir = os.path.join(lerobot_root, 'examples', 'aloha_ros', 'aloha_scripts')
    if os.path.exists(aloha_scripts_dir) and aloha_scripts_dir not in sys.path:
        sys.path.insert(0, aloha_scripts_dir)
except Exception:
    # Fallback: try to find lerobot root from current working directory
    cwd = os.getcwd()
    if 'lerobot' in cwd:
        lerobot_root = cwd[:cwd.rindex('lerobot') + len('lerobot')]
        aloha_scripts_dir = os.path.join(lerobot_root, 'examples', 'aloha_ros', 'aloha_scripts')
        if os.path.exists(aloha_scripts_dir) and aloha_scripts_dir not in sys.path:
            sys.path.insert(0, aloha_scripts_dir)

from constants import (
    DT,
    PUPPET_SLEEP_POSITION,
    PUPPET_GRIPPER_JOINT_OPEN,
)

# Try to import ROS dependencies
try:
    from interbotix_xs_modules.arm import InterbotixManipulatorXS
    from interbotix_xs_msgs.msg import JointSingleCommand
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS dependencies not available. Install interbotix_xs_modules to use ViperXRos.")


def get_arm_joint_positions(bot):
    """Get current arm joint positions (6 joints)."""
    joint_states = bot.dxl.joint_states
    if joint_states and len(joint_states.position) >= 6:
        return list(joint_states.position[:6])
    return [0.0] * 6


def get_arm_gripper_position(bot):
    """Get current gripper joint position."""
    joint_states = bot.dxl.joint_states
    if joint_states and len(joint_states.position) > 6:
        return joint_states.position[6]
    return 0.0


def move_arm_smoothly(bot, target_pose, move_time=1.0):
    """Move arm smoothly to target position using trajectory interpolation."""
    num_steps = int(move_time / DT)
    curr_pose = get_arm_joint_positions(bot)
    traj = np.linspace(curr_pose, target_pose, num_steps)
    for t in range(num_steps):
        bot.arm.set_joint_positions(traj[t], blocking=False)
        time.sleep(DT)


def move_gripper_smoothly(bot, target_position, move_time=0.5):
    """Move gripper smoothly to target position using trajectory interpolation."""
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_position = get_arm_gripper_position(bot)
    traj = np.linspace([curr_position], [target_position], num_steps)
    for t in range(num_steps):
        gripper_command.cmd = traj[t][0]
        bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)


class ViperXRos(Robot):
    """
    ViperX Robot using ROS InterbotixManipulatorXS for control.
    Uses lerobot camera system for image capture.
    This is the puppet/follower arm.
    """

    config_class = ViperXRosConfig
    name = "viperx_ros"

    def __init__(self, config: ViperXRosConfig):
        if not ROS_AVAILABLE:
            raise ImportError(
                "ROS dependencies not available. Please install: "
                "pip install interbotix-xseries-modules"
            )
        
        super().__init__(config)
        self.config = config
        
        # Initialize ROS robot
        self.bot = InterbotixManipulatorXS(
            robot_model=config.robot_model,
            group_name="arm",
            gripper_name="gripper",
            robot_name=config.robot_name,
            init_node=config.init_ros_node
        )
        
        # Setup cameras if provided
        if config.cameras:
            self.cameras = make_cameras_from_configs(config.cameras)
        else:
            self.cameras = {}
        
        # Gripper command message
        self.gripper_command = JointSingleCommand(name="gripper") if ROS_AVAILABLE else None
        
        # Setup robot (puppet bot setup: position mode, torque on)
        self._setup_robot()

    def _setup_robot(self):
        """Setup puppet robot with proper operating modes."""
        self.bot.dxl.robot_reboot_motors("single", "gripper", True)
        self.bot.dxl.robot_set_operating_modes("group", "arm", "position")
        self.bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        self.bot.dxl.robot_torque_enable("group", "arm", True)
        self.bot.dxl.robot_torque_enable("single", "gripper", True)
        logger.info(f"ViperXRos {self.config.robot_name} setup complete")

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features."""
        motor_names = [
            "waist", "shoulder", "elbow", "forearm_roll", 
            "wrist_angle", "wrist_rotate", "gripper"
        ]
        return {f"{motor}.pos": float for motor in motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features (same as motors)."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot and cameras are connected."""
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        return cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot and cameras."""
        if self.is_connected:
            # raise DeviceAlreadyConnectedError(f"{self} already connected")
            pass

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """ROS robots don't use lerobot calibration system."""
        return True

    def calibrate(self) -> None:
        """Calibration is handled by ROS."""
        logger.info("Calibration is handled by ROS system. Skipping lerobot calibration.")

    def configure(self) -> None:
        """Apply configuration to robot."""
        # Configuration is done in _setup_robot
        pass

    def get_observation(self) -> dict[str, Any]:
        """Get observations from robot and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}
        
        # Read arm positions from ROS
        start = time.perf_counter()
        joint_states = self.bot.dxl.joint_states
        
        if joint_states and len(joint_states.position) >= 6:
            obs_dict["waist.pos"] = joint_states.position[0]
            obs_dict["shoulder.pos"] = joint_states.position[1]
            obs_dict["elbow.pos"] = joint_states.position[2]
            obs_dict["forearm_roll.pos"] = joint_states.position[3]
            obs_dict["wrist_angle.pos"] = joint_states.position[4]
            obs_dict["wrist_rotate.pos"] = joint_states.position[5]
            obs_dict["gripper.pos"] = joint_states.position[6] if len(joint_states.position) > 6 else 0.0
        else:
            logger.warning("Joint states not available")
            for motor in ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]:
                obs_dict[f"{motor}.pos"] = 0.0
            raise ValueError("Joint states not available")
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send actions to robot.
        
        Args:
            action: Dictionary with keys like "waist.pos", "gripper.pos", etc.
        
        Returns:
            Dictionary of actions actually sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract arm actions (6 joints)
        arm_joints = [
            action.get("waist.pos", 0.0),
            action.get("shoulder.pos", 0.0),
            action.get("elbow.pos", 0.0),
            action.get("forearm_roll.pos", 0.0),
            action.get("wrist_angle.pos", 0.0),
            action.get("wrist_rotate.pos", 0.0),
        ]
        
        # Send joint positions to robot
        self.bot.arm.set_joint_positions(arm_joints, blocking=False)
        
        # Send gripper command
        gripper_pos = action.get("gripper.pos", 0.0)
        self.gripper_command.cmd = gripper_pos
        self.bot.gripper.core.pub_single.publish(self.gripper_command)
        
        # Return the action sent
        return action

    def disconnect(self) -> None:
        """Disconnect from robot and cameras.
        
        Before disconnecting, moves robot to sleep position and turns off torque.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # Move to sleep position smoothly
            logger.info(f"Moving {self} to sleep position...")
            move_arm_smoothly(self.bot, list(PUPPET_SLEEP_POSITION), move_time=1.0)
            
            # Open gripper smoothly
            logger.info(f"Opening gripper...")
            move_gripper_smoothly(self.bot, PUPPET_GRIPPER_JOINT_OPEN, move_time=0.5)
            
            # Turn off torque
            logger.info(f"Turning off torque...")
            self.bot.dxl.robot_torque_enable("group", "arm", False)
            self.bot.dxl.robot_torque_enable("single", "gripper", False)
            
        except Exception as e:
            logger.warning(f"Error during sleep sequence: {e}. Continuing with disconnect...")

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected and moved to sleep position.")

