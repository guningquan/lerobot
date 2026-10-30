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
    # unlock() lowers the arm motors' position P gain to 0 over this many seconds before cutting
    # the torque, so the arm's weight passes to the operator's hand gradually instead of the arm
    # dropping the moment it is released. 0 cuts the torque at once.
    release_ramp_s: float = 0.4
    # Clutch-style takeover. When the arm was last driven onto the follower's pose (send_feedback,
    # e.g. DAgger's pause handover), unlock() waits for the released arm to settle and stores
    # offset = arm pose - that follower pose; get_action() then returns arm pose - offset for the
    # six arm joints (the gripper stays absolute), so the follower starts from where it holds and
    # only follows the operator's motion from there. lock() clears the offset.
    relative_on_release: bool = False
    # The arm counts as settled once every joint stayed within `settle_tolerance` rad for
    # `settle_s` seconds; after `settle_timeout_s` the offset is taken anyway.
    settle_s: float = 0.2
    settle_tolerance: float = 0.01
    settle_timeout_s: float = 1.0
    # Warn when a joint's offset exceeds this (rad): the leader and follower poses then differ
    # visibly, so re-align (pause) before fine work.
    max_offset_warn: float = 0.25

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
        if self.release_ramp_s < 0:
            raise ValueError(f"`release_ramp_s` must be >= 0, got {self.release_ramp_s}.")
        if self.settle_s <= 0 or self.settle_tolerance <= 0 or self.settle_timeout_s < self.settle_s:
            raise ValueError(
                "Need settle_s > 0, settle_tolerance > 0 and settle_timeout_s >= settle_s, got "
                f"{self.settle_s}, {self.settle_tolerance}, {self.settle_timeout_s}."
            )
