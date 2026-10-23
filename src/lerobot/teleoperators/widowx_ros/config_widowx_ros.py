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


@TeleoperatorConfig.register_subclass("widowx_ros")
@dataclass
class WidowXRosConfig(TeleoperatorConfig):
    # ROS robot name (as defined in ROS launch files)
    robot_name: str  # e.g., "master_left" or "master_right"

    # Robot model type
    robot_model: str = "wx250s"

    # ROS node initialization
    init_ros_node: bool = True

    # "left" or "right": selects this arm's gripper calibration in
    # examples/aloha_ros/aloha_scripts/constants.py. None infers it from robot_name.
    side: str | None = None

    # Control loop rate in Hz (e.g. 30 or 50); sets the gripper interpolation step of the slow moves.
    fps: int = 30

    # Move to START_ARM_POSE (gripper at start_open_ratio) on connect.
    move_to_start_on_connect: bool = True
    # Gripper opening in move_to_start(): 0.0 = closed, 1.0 = open.
    start_open_ratio: float = 1.0
    # Cut the torque at the end of connect() so the operator can move the arm right away (needed
    # by lerobot-record / lerobot-teleoperate, which start reading actions immediately). Scripts
    # that wait for a key press set this to False and call unlock() themselves.
    release_on_connect: bool = True

    def __post_init__(self) -> None:
        if self.side is None:
            for side in ("left", "right"):
                if self.robot_name == side or self.robot_name.endswith(f"_{side}"):
                    self.side = side
            if self.side is None:
                raise ValueError(
                    f"Cannot infer the arm side from robot_name={self.robot_name!r}; set `side` explicitly."
                )
        if self.side not in ("left", "right"):
            raise ValueError(f"`side` must be 'left' or 'right', got {self.side!r}.")
        if self.fps <= 0:
            raise ValueError(f"`fps` must be positive, got {self.fps}.")
        if not 0.0 <= self.start_open_ratio <= 1.0:
            raise ValueError(f"`start_open_ratio` must be in [0, 1], got {self.start_open_ratio}.")
