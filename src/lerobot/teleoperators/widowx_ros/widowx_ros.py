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

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_widowx_ros import WidowXRosConfig

logger = logging.getLogger(__name__)

# Import constants from aloha_scripts/constants.py
import sys
import os

# Add aloha_scripts directory to path
# __file__ is at: src/lerobot/teleoperators/widowx_ros/widowx_ros.py
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
    MASTER_SLEEP_POSITION,
    MASTER_GRIPPER_JOINT_OPEN,
)


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
    from interbotix_xs_msgs.msg import JointSingleCommand
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_position = get_arm_gripper_position(bot)
    traj = np.linspace([curr_position], [target_position], num_steps)
    for t in range(num_steps):
        gripper_command.cmd = traj[t][0]
        bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)

# Try to import ROS dependencies
try:
    from interbotix_xs_modules.arm import InterbotixManipulatorXS
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS dependencies not available. Install interbotix_xs_modules to use WidowXRos.")


class WidowXRos(Teleoperator):
    """
    WidowX Teleoperator using ROS InterbotixManipulatorXS for control.
    This is the master/leader arm used for teleoperation.
    """

    config_class = WidowXRosConfig
    name = "widowx_ros"

    def __init__(self, config: WidowXRosConfig):
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
        
        # Setup robot (master bot setup: PWM mode, torque off)
        self._setup_robot()

    def _setup_robot(self):
        """Setup master robot with proper operating modes."""
        self.bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
        self.bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        self.bot.dxl.robot_torque_enable("group", "arm", False)
        self.bot.dxl.robot_torque_enable("single", "gripper", False)
        logger.info(f"WidowXRos {self.config.robot_name} setup complete")

    @property
    def action_features(self) -> dict[str, type]:
        """Action features: joint positions."""
        motor_names = [
            "waist", "shoulder", "elbow", "forearm_roll", 
            "wrist_angle", "wrist_rotate", "gripper"
        ]
        return {f"{motor}.pos": float for motor in motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        """No feedback features for master arm."""
        return {}

    @property
    def is_connected(self) -> bool:
        """ROS robots are always 'connected' if initialized."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot."""
        if self.is_connected:
            # raise DeviceAlreadyConnectedError(f"{self} already connected")
            pass
        
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

    def get_action(self) -> dict[str, float]:
        """Get current joint positions from master arm."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        joint_states = self.bot.dxl.joint_states
        
        action_dict = {}
        if joint_states and len(joint_states.position) >= 6:
            action_dict["waist.pos"] = joint_states.position[0]
            action_dict["shoulder.pos"] = joint_states.position[1]
            action_dict["elbow.pos"] = joint_states.position[2]
            action_dict["forearm_roll.pos"] = joint_states.position[3]
            action_dict["wrist_angle.pos"] = joint_states.position[4]
            action_dict["wrist_rotate.pos"] = joint_states.position[5]
            action_dict["gripper.pos"] = joint_states.position[6] if len(joint_states.position) > 6 else 0.0
        else:
            logger.warning("Joint states not available")
            for motor in ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]:
                action_dict[f"{motor}.pos"] = 0.0
            raise ValueError("Joint states not available")
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Master arm doesn't receive feedback."""
        raise NotImplementedError("Master arm does not receive feedback.")

    def disconnect(self) -> None:
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        try:
            # Move to sleep position smoothly
            print(f"Moving {self} to sleep position...")
            move_arm_smoothly(self.bot, list(MASTER_SLEEP_POSITION), move_time=1.0)
            
            # Open gripper smoothly
            logger.info(f"Opening gripper...")
            move_gripper_smoothly(self.bot, MASTER_GRIPPER_JOINT_OPEN, move_time=0.5)
            
            # Turn off torque
            logger.info(f"Turning off torque...")
            self.bot.dxl.robot_torque_enable("group", "arm", False)
            self.bot.dxl.robot_torque_enable("single", "gripper", False)
            
        except Exception as e:
            logger.warning(f"Error during sleep sequence: {e}. Continuing with disconnect...")

        logger.info(f"{self} disconnected.")

