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
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.tactile.utils import make_tactile_sensors_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_aloha_ros import AlohaRosConfig
from ..viperx_ros.config_viperx_ros import ViperXRosConfig
from ..viperx_ros.viperx_ros import ViperXRos

logger = logging.getLogger(__name__)


class AlohaRos(Robot):
    """
    ALOHA Robot using ROS InterbotixManipulatorXS for control.
    Uses lerobot camera system for image capture.
    Combines two ViperXRos robots (left and right puppet arms).
    """

    config_class = AlohaRosConfig
    name = "aloha_ros"

    def __init__(self, config: AlohaRosConfig):
        super().__init__(config)
        self.config = config

        # Create left arm config
        left_arm_config = ViperXRosConfig(
            id=f"{config.id}_left" if config.id else None,
            robot_name=config.puppet_left_robot_name,
            robot_model=config.robot_model,
            init_ros_node=config.init_ros_node,
            cameras={},  # Cameras are handled at AlohaRos level
        )

        # Create right arm config
        right_arm_config = ViperXRosConfig(
            id=f"{config.id}_right" if config.id else None,
            robot_name=config.puppet_right_robot_name,
            robot_model=config.robot_model,
            init_ros_node=False,  # Only initialize node once
            cameras={},  # Cameras are handled at AlohaRos level
        )

        self.left_arm = ViperXRos(left_arm_config)
        self.right_arm = ViperXRos(right_arm_config)

        # Setup cameras if provided (shared between both arms)
        if config.cameras:
            self.cameras = make_cameras_from_configs(config.cameras)
        else:
            self.cameras = {}

        # Setup tactile sensors if provided (attached to follower grippers)
        if config.tactile_sensors:
            self.tactile_sensors = make_tactile_sensors_from_configs(config.tactile_sensors)
        else:
            self.tactile_sensors = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for both arms."""
        motor_names = [
            "waist", "shoulder", "elbow", "forearm_roll", 
            "wrist_angle", "wrist_rotate", "gripper"
        ]
        features = {}
        for side in ["left", "right"]:
            for motor in motor_names:
                features[f"{side}_{motor}.pos"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def _tactile_ft(self) -> dict[str, tuple]:
        """Tactile feature shapes: (channels, height, width) — channel-first."""
        return {
            f"tactile_{name}": (sensor.channels, sensor.height, sensor.width)
            for name, sensor in self.tactile_sensors.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features."""
        return {**self._motors_ft, **self._cameras_ft, **self._tactile_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features (same as motors)."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robots, cameras, and tactile sensors are connected."""
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        tactile_connected = all(s.is_connected for s in self.tactile_sensors.values()) if self.tactile_sensors else True
        return self.left_arm.is_connected and self.right_arm.is_connected and cameras_connected and tactile_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robots, cameras, and tactile sensors."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect arms
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        # Connect tactile sensors (staggered to avoid SDK camera resource conflicts)
        for name, sensor in self.tactile_sensors.items():
            import time as _time
            logger.info(f"Connecting tactile sensor: {name} ...")
            sensor.connect()
            _time.sleep(1.0)  # Allow SDK camera resources to stabilise

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if both arms are calibrated."""
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """Calibrate both arms."""
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        """Apply configuration to robots."""
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        """Get observations from both robots and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Get observations from left arm (without cameras)
        left_obs = self.left_arm.get_observation()
        for key, value in left_obs.items():
            if not isinstance(value, (list, tuple)) or (isinstance(value, tuple) and len(value) == 3):
                # Skip camera images, add "left_" prefix to motor observations
                if key.endswith(".pos"):
                    obs_dict[f"left_{key}"] = value

        # Get observations from right arm (without cameras)
        right_obs = self.right_arm.get_observation()
        for key, value in right_obs.items():
            if not isinstance(value, (list, tuple)) or (isinstance(value, tuple) and len(value) == 3):
                # Skip camera images, add "right_" prefix to motor observations
                if key.endswith(".pos"):
                    obs_dict[f"right_{key}"] = value

        # Capture images from cameras (shared cameras)
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        # Capture tactile data — kept in (C,H,W) for policy consumption
        for tac_key, sensor in self.tactile_sensors.items():
            obs_dict[f"tactile_{tac_key}"] = sensor.async_read()

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send actions to both robots.
        
        Args:
            action: Dictionary with keys like "left_waist.pos", "right_gripper.pos", etc.
        
        Returns:
            Dictionary of actions actually sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract left arm actions
        left_action = {
            key.removeprefix("left_"): value 
            for key, value in action.items() 
            if key.startswith("left_")
        }

        # Extract right arm actions
        right_action = {
            key.removeprefix("right_"): value 
            for key, value in action.items() 
            if key.startswith("right_")
        }

        # Send actions to both arms
        if left_action:
            self.left_arm.send_action(left_action)
        if right_action:
            self.right_arm.send_action(right_action)

        return action

    def disconnect(self) -> None:
        """Disconnect from robots, cameras, and tactile sensors."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect arms
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Disconnect tactile sensors
        for sensor in self.tactile_sensors.values():
            sensor.disconnect()

        logger.info(f"{self} disconnected.")
