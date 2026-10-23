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

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("aloha_ros")
@dataclass
class AlohaRosConfig(RobotConfig):
    # ROS robot names for puppet arms
    puppet_left_robot_name: str = "puppet_left"
    puppet_right_robot_name: str = "puppet_right"

    # Robot model type
    robot_model: str = "vx300s"

    # ROS node initialization (only first arm initializes node)
    init_ros_node: bool = True

    # Control loop rate in Hz (e.g. 30 or 50). Must match the loop that calls send_action()
    # (--dataset.fps for lerobot-record, --fps for lerobot-rollout). See ViperXRosConfig.fps.
    fps: int = 30

    # Move both arms to START_ARM_POSE on connect (see ViperXRosConfig).
    move_to_start_on_connect: bool = True
    # Gripper opening in move_to_start(): 0.0 = closed, 1.0 = open.
    start_open_ratio: float = 1.0
    # Seconds to ease onto the first commands after connect / move_to_start() / begin_blend().
    start_blend_s: float = 1.5

    # Puppet gripper contact guard and current limit (see ViperXRosConfig).
    gripper_contact_guard: bool = True
    gripper_stall_error: float = 0.04
    gripper_stall_motion: float = 0.001
    gripper_stall_cycles: int = 8
    gripper_release_margin: float = 0.05
    gripper_effort_threshold: float | None = None
    gripper_current_limit: int | None = 300

    # Camera configuration (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
