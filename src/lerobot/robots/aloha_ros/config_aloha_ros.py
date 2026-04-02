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
    
    # Camera configuration (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
