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

from lerobot.teleoperators.widowx_ros.config_widowx_ros import WidowXRosConfig
from lerobot.teleoperators.widowx_ros.widowx_ros import WidowXRos
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_aloha_teleop_ros import AlohaTeleopRosConfig

logger = logging.getLogger(__name__)


class AlohaTeleopRos(Teleoperator):
    """
    ALOHA Bimanual Teleoperator System using dual WidowX leader arms via ROS.
    Based on the ALOHA (A Low-cost Open-source Hardware System for Bimanual Teleoperation) design.
    See WidowXRos for how each arm is set up; moves that involve both arms run on both at once.
    """

    config_class = AlohaTeleopRosConfig
    name = "aloha_teleop_ros"

    def __init__(self, config: AlohaTeleopRosConfig):
        super().__init__(config)
        self.config = config

        def arm_config(side: str, robot_name: str, init_ros_node: bool) -> WidowXRosConfig:
            return WidowXRosConfig(
                id=f"{config.id}_{side}" if config.id else None,
                robot_name=robot_name,
                robot_model=config.robot_model,
                init_ros_node=init_ros_node,
                side=side,
                fps=config.fps,
                move_to_start_on_connect=False,  # both arms move together in move_to_start()
                start_open_ratio=config.start_open_ratio,
                release_on_connect=False,  # both arms are released together in connect()
                release_ramp_s=config.release_ramp_s,
                relative_on_release=config.relative_on_release,
                settle_s=config.settle_s,
                settle_tolerance=config.settle_tolerance,
                settle_timeout_s=config.settle_timeout_s,
                max_offset_warn=config.max_offset_warn,
            )

        self.left_arm = WidowXRos(arm_config("left", config.left_arm_robot_name, config.init_ros_node))
        # Only initialize node once
        self.right_arm = WidowXRos(arm_config("right", config.right_arm_robot_name, False))

    def _on_both_arms(self, fn: Callable[[WidowXRos], None]) -> None:
        """Run `fn` on the left and the right arm at the same time; re-raises the first error."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(fn, (self.left_arm, self.right_arm)))

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features for both arms."""
        features: dict[str, type] = {f"left_{key}": float for key in self.left_arm.action_features}
        features.update({f"right_{key}": float for key in self.right_arm.action_features})
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """Both arms can be driven to joint positions (see WidowXRos.send_feedback)."""
        return self.action_features

    @property
    def is_connected(self) -> bool:
        """Check if both arms are connected."""
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Set both arms up, optionally move them to the start pose and release them to the operator."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        if self.config.move_to_start_on_connect:
            self.move_to_start()
        if self.config.release_on_connect:
            logger.warning(f"{self}: releasing both master arms now - hold the handles.")
            self.unlock()

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

    def lock(self) -> None:
        """Hold both master arms in place (torque on), e.g. on pause or after an episode."""
        self.left_arm.lock()
        self.right_arm.lock()

    def unlock(self) -> None:
        """Release both master arms (torque off) so the operator can move them; both soften at the
        same time (see WidowXRos.unlock)."""
        self._on_both_arms(lambda arm: arm.unlock())

    # enable_torque / disable_torque: the names lerobot's actuated-teleop handover looks for
    def enable_torque(self) -> None:
        self.lock()

    def disable_torque(self) -> None:
        self.unlock()

    def move_to_start(self, start_open_ratio: float | None = None) -> None:
        """Move both arms to START_ARM_POSE and park the grippers; they stay locked."""
        self._on_both_arms(lambda arm: arm.move_to_start(start_open_ratio))

    def sleep(self) -> None:
        """Both arms to the rest pose, torque kept on."""
        self._on_both_arms(lambda arm: arm.sleep())

    def shutdown(self) -> None:
        """Both arms to the rest pose, then torque off."""
        self._on_both_arms(lambda arm: arm.shutdown())

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
        """Drive both master arms towards "left_*" / "right_*" joint positions (WidowXRos.send_feedback)."""
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            arm_feedback = {
                key.removeprefix(f"{side}_"): value for key, value in feedback.items() if key.startswith(f"{side}_")
            }
            if arm_feedback:
                arm.send_feedback(arm_feedback)

    def disconnect(self) -> None:
        """Walk both arms to the rest pose and power them off (unless shutdown() already did)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            self._on_both_arms(lambda arm: None if arm.is_powered_down else arm.shutdown())
        except Exception as e:
            logger.warning(f"Error during shutdown sequence: {e}. Continuing with disconnect...")

        # No further motion once powered down
        self.left_arm.disconnect()
        self.right_arm.disconnect()
