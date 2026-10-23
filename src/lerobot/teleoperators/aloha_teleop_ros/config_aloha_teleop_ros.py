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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("aloha_teleop_ros")
@dataclass
class AlohaTeleopRosConfig(TeleoperatorConfig):
    # ROS robot names for master arms
    left_arm_robot_name: str = "master_left"
    right_arm_robot_name: str = "master_right"

    # Robot model type
    robot_model: str = "wx250s"

    # ROS node initialization (only first arm initializes node)
    init_ros_node: bool = True

    # Control loop rate in Hz (e.g. 30 or 50), see WidowXRosConfig.fps.
    fps: int = 30

    # Move both arms to START_ARM_POSE on connect.
    move_to_start_on_connect: bool = True
    # Gripper opening in move_to_start(): 0.0 = closed, 1.0 = open.
    start_open_ratio: float = 1.0
    # Cut the torque at the end of connect() (see WidowXRosConfig.release_on_connect).
    release_on_connect: bool = True
