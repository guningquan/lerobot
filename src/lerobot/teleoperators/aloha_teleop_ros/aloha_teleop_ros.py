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
from functools import cached_property

from lerobot.teleoperators.widowx_ros.config_widowx_ros import WidowXRosConfig
from lerobot.teleoperators.widowx_ros.widowx_ros import WidowXRos

from ..teleoperator import Teleoperator
from .config_aloha_teleop_ros import AlohaTeleopRosConfig

logger = logging.getLogger(__name__)


class AlohaTeleopRos(Teleoperator):
    """
    ALOHA Bimanual Teleoperator System using dual WidowX leader arms via ROS.
    Based on the ALOHA (A Low-cost Open-source Hardware System for Bimanual Teleoperation) design.
    """

    config_class = AlohaTeleopRosConfig
    name = "aloha_teleop_ros"

    def __init__(self, config: AlohaTeleopRosConfig):
        super().__init__(config)
        self.config = config

        # Create left arm config
        left_arm_config = WidowXRosConfig(
            id=f"{config.id}_left" if config.id else None,
            robot_name=config.left_arm_robot_name,
            robot_model=config.robot_model,
            init_ros_node=config.init_ros_node,
        )

        # Create right arm config
        right_arm_config = WidowXRosConfig(
            id=f"{config.id}_right" if config.id else None,
            robot_name=config.right_arm_robot_name,
            robot_model=config.robot_model,
            init_ros_node=False,  # Only initialize node once
        )

        self.left_arm = WidowXRos(left_arm_config)
        self.right_arm = WidowXRos(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features for both arms."""
        return {f"left_{motor}.pos": float for motor in self.left_arm.action_features.keys()} | {
            f"right_{motor}.pos": float for motor in self.right_arm.action_features.keys()
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """No feedback features."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if both arms are connected."""
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect both arms."""
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        """Check if both arms are calibrated."""
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """Calibrate both arms."""
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        """Configure both arms."""
        self.left_arm.configure()
        self.right_arm.configure()

    def get_action(self) -> dict[str, float]:
        """Get actions from both master arms."""
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Master arms don't receive feedback."""
        raise NotImplementedError("Master arms do not receive feedback.")

    def disconnect(self) -> None:
        """Disconnect both arms."""
        self.left_arm.disconnect()
        self.right_arm.disconnect()

