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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..viperx_ros.config_viperx_ros import ViperXRosConfig
from ..viperx_ros.viperx_ros import ViperXRos
from .config_aloha_ros import AlohaRosConfig

logger = logging.getLogger(__name__)


class AlohaRos(Robot):
    """
    ALOHA Robot using ROS InterbotixManipulatorXS for control.
    Uses lerobot camera system for image capture.
    Combines two ViperXRos robots (left and right puppet arms); see ViperXRos for how each arm is
    driven and what `gripper.pos` means. Moves that involve both arms (start pose, sleep, shutdown)
    run on both arms at the same time.
    """

    config_class = AlohaRosConfig
    name = "aloha_ros"

    def __init__(self, config: AlohaRosConfig):
        super().__init__(config)
        self.config = config

        def arm_config(side: str, robot_name: str, init_ros_node: bool) -> ViperXRosConfig:
            return ViperXRosConfig(
                id=f"{config.id}_{side}" if config.id else None,
                robot_name=robot_name,
                robot_model=config.robot_model,
                init_ros_node=init_ros_node,
                side=side,
                fps=config.fps,
                move_to_start_on_connect=False,  # both arms move together in move_to_start()
                start_open_ratio=config.start_open_ratio,
                start_blend_s=config.start_blend_s,
                gripper_contact_guard=config.gripper_contact_guard,
                gripper_stall_error=config.gripper_stall_error,
                gripper_stall_motion=config.gripper_stall_motion,
                gripper_stall_cycles=config.gripper_stall_cycles,
                gripper_release_margin=config.gripper_release_margin,
                gripper_effort_threshold=config.gripper_effort_threshold,
                gripper_current_limit=config.gripper_current_limit,
                cameras={},  # Cameras are handled at AlohaRos level
            )

        self.left_arm = ViperXRos(arm_config("left", config.puppet_left_robot_name, config.init_ros_node))
        # Only initialize node once
        self.right_arm = ViperXRos(arm_config("right", config.puppet_right_robot_name, False))

        # Setup cameras if provided (shared between both arms)
        if config.cameras:
            self.cameras = make_cameras_from_configs(config.cameras)
        else:
            self.cameras = {}

    def _on_both_arms(self, fn: Callable[[ViperXRos], None]) -> None:
        """Run `fn` on the left and the right arm at the same time; re-raises the first error."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(fn, (self.left_arm, self.right_arm)))

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for both arms."""
        return {
            f"{side}_{key}": float
            for side, arm in (("left", self.left_arm), ("right", self.right_arm))
            for key in arm.action_features
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
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
        """Check if robots and cameras are connected."""
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        return self.left_arm.is_connected and self.right_arm.is_connected and cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robots and cameras."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect arms (motor setup, see ViperXRos.configure)
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        if self.config.move_to_start_on_connect:
            self.move_to_start()
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
        """Re-apply the motor setup of both arms (connect() already does this once)."""
        self.left_arm.configure()
        self.right_arm.configure()

    def move_to_start(self, start_open_ratio: float | None = None) -> None:
        """Move both arms to START_ARM_POSE and park the grippers (see ViperXRos.move_to_start)."""
        self._on_both_arms(lambda arm: arm.move_to_start(start_open_ratio))

    def begin_blend(self, duration_s: float | None = None) -> None:
        """Ease both arms onto the next commands (see ViperXRos.begin_blend)."""
        self.left_arm.begin_blend(duration_s)
        self.right_arm.begin_blend(duration_s)

    def sleep(self) -> None:
        """Both arms to the rest pose, torque kept on (see ViperXRos.sleep)."""
        self._on_both_arms(lambda arm: arm.sleep())

    def shutdown(self) -> None:
        """Both arms to the rest pose, then torque off (see ViperXRos.shutdown)."""
        self._on_both_arms(lambda arm: arm.shutdown())

    def get_observation(self) -> dict[str, Any]:
        """Get observations from both robots and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            # The arms have no cameras of their own, so these are all "<motor>.pos" values
            for key, value in arm.get_observation().items():
                obs_dict[f"{side}_{key}"] = value

        # Capture images from cameras (shared cameras)
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send actions to both robots.

        Args:
            action: Dictionary with keys like "left_waist.pos", "right_gripper.pos", etc. Joints are
                in rad; "*_gripper.pos" is the gripper opening in [0, 1].

        Returns:
            Dictionary of actions actually sent.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent: dict[str, float] = {}
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            arm_action = {
                key.removeprefix(f"{side}_"): value
                for key, value in action.items()
                if key.startswith(f"{side}_")
            }
            if arm_action:
                sent.update({f"{side}_{key}": value for key, value in arm.send_action(arm_action).items()})
        return sent

    def disconnect(self) -> None:
        """Walk both arms to the rest pose and power them off (unless shutdown() already did), then
        disconnect robots and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            self._on_both_arms(lambda arm: None if arm.is_powered_down else arm.shutdown())
        except Exception as e:
            logger.warning(f"Error during shutdown sequence: {e}. Continuing with disconnect...")

        # Disconnect arms (no further motion once powered down)
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
