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


def infer_arm_side(robot_name: str) -> str:
    """Infer the arm side from a ROS robot name, e.g. "puppet_left" -> "left"."""
    for side in ("left", "right"):
        if robot_name == side or robot_name.endswith(f"_{side}"):
            return side
    raise ValueError(f"Cannot infer the arm side from robot_name={robot_name!r}; set `side` explicitly.")


@RobotConfig.register_subclass("viperx_ros")
@dataclass
class ViperXRosConfig(RobotConfig):
    # ROS robot name (as defined in ROS launch files)
    robot_name: str  # e.g., "puppet_left" or "puppet_right"

    # Robot model type
    robot_model: str = "vx300s"

    # ROS node initialization
    init_ros_node: bool = True

    # "left" or "right": selects this arm's gripper calibration in
    # examples/aloha_ros/aloha_scripts/constants.py. None infers it from robot_name.
    side: str | None = None

    # Control loop rate in Hz (e.g. 30 or 50). Must match the loop that calls send_action()
    # (--dataset.fps for lerobot-record, --fps for lerobot-rollout): every command asks the servos
    # to reach their target in 1 / fps seconds.
    fps: int = 30

    # Move to START_ARM_POSE (gripper at start_open_ratio) on connect, so that the first command is
    # never a jump from the sleep pose.
    move_to_start_on_connect: bool = True
    # Gripper opening in move_to_start(): 0.0 = closed, 1.0 = open.
    start_open_ratio: float = 1.0
    # After connect / move_to_start() / begin_blend(), the arm eases from where it is onto the
    # incoming commands over this many seconds instead of jumping to them.
    start_blend_s: float = 1.5

    # Gripper contact guard (ALOHA2 one_side_teleop.py): once the gripper stops moving while still
    # short of its closing target (it is pressing on an object), hold it where it is.
    gripper_contact_guard: bool = True
    gripper_stall_error: float = 0.04  # rad still short of the target
    gripper_stall_motion: float = 0.001  # rad moved in one tick, below = not moving
    gripper_stall_cycles: int = 8  # consecutive stalled ticks before holding
    gripper_release_margin: float = 0.05  # rad the target must open past the hold point to release
    gripper_effort_threshold: float | None = None  # also hold once |effort| reaches this (None: off)
    # Gripper motor Current_Limit register; None leaves it unchanged.
    gripper_current_limit: int | None = 300

    # Meaning of the observed "gripper.pos":
    #   "joint"  - gripper joint angle normalised with this arm's calibration (1 = open, ~0.07-0.09 =
    #              closed empty); what aloha_ros_record.py / lerobot-record store.
    #   "aloha2" - finger position normalised like ALOHA2's RealEnv (PUPPET_GRIPPER_POSITION_NORMALIZE_FN,
    #              ~0.31 = open, ~-0.11 = closed); for policies trained on converted ALOHA2 HDF5 data
    #              (e.g. packaging_lerobot). Actions are the same either way.
    gripper_state_format: str = "joint"

    # Camera configuration (same as ViperX)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.side is None:
            self.side = infer_arm_side(self.robot_name)
        if self.side not in ("left", "right"):
            raise ValueError(f"`side` must be 'left' or 'right', got {self.side!r}.")
        if self.gripper_state_format not in ("joint", "aloha2"):
            raise ValueError(f"`gripper_state_format` must be 'joint' or 'aloha2', got {self.gripper_state_format!r}.")
        if self.fps <= 0:
            raise ValueError(f"`fps` must be positive, got {self.fps}.")
        if not 0.0 <= self.start_open_ratio <= 1.0:
            raise ValueError(f"`start_open_ratio` must be in [0, 1], got {self.start_open_ratio}.")
