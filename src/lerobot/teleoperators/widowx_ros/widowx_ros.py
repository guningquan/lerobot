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
import time
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_widowx_ros import WidowXRosConfig

logger = logging.getLogger(__name__)

# Import constants from aloha_scripts/constants.py
import sys
import os

# Add aloha_scripts directory to path
# __file__ is at: src/lerobot/teleoperators/widowx_ros/widowx_ros.py
# Need to go up 5 levels to reach lerobot root
try:
    current_file = os.path.abspath(__file__)
    lerobot_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
    aloha_scripts_dir = os.path.join(lerobot_root, 'examples', 'aloha_ros', 'aloha_scripts')
    if os.path.exists(aloha_scripts_dir) and aloha_scripts_dir not in sys.path:
        sys.path.insert(0, aloha_scripts_dir)
except Exception:
    # Fallback: try to find lerobot root from current working directory
    cwd = os.getcwd()
    if 'lerobot' in cwd:
        lerobot_root = cwd[:cwd.rindex('lerobot') + len('lerobot')]
        aloha_scripts_dir = os.path.join(lerobot_root, 'examples', 'aloha_ros', 'aloha_scripts')
        if os.path.exists(aloha_scripts_dir) and aloha_scripts_dir not in sys.path:
            sys.path.insert(0, aloha_scripts_dir)

from constants import (
    MASTER_GRIPPER_JOINT_RANGE,
    MASTER_SHUTDOWN_MOVE_TIMES,
    MASTER_SLEEP_MOVE_TIMES,
    MASTER_SLEEP_POSITION,
    OLD_MASTER_SLEEP_POSITION,
    START_ARM_POSE,
)

# Try to import ROS dependencies
try:
    from interbotix_xs_modules.arm import InterbotixManipulatorXS
    from interbotix_xs_msgs.msg import JointSingleCommand
    from robot_utils import move_arms, move_grippers, set_arm_position_time_profile, torque_off, torque_on
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS dependencies not available. Install interbotix_xs_modules to use WidowXRos.")

ARM_MOTORS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
# Durations of the slow moves to the start pose (ALOHA2 one_side_teleop.py DEFAULT_PREP_ARM_MOVE_TIME).
START_ARM_MOVE_TIME_S = 2.0
START_GRIPPER_MOVE_TIME_S = 0.5


class WidowXRos(Teleoperator):
    """
    WidowX Teleoperator using ROS InterbotixManipulatorXS for control.
    This is the master/leader arm used for teleoperation.

    Follows ALOHA2's act-triple-plus-aloha2/aloha_scripts: the arm is set to position mode with a
    time-based profile so that it can be driven to the start / rest pose, and is held rigid (torque
    on) until unlock() hands it to the operator. `gripper.pos` is the gripper opening normalised to
    [0, 1] with this arm's own calibration (0 = closed, 1 = open; a squeeze past the calibrated
    stop can go slightly outside); the puppet maps it to its own gripper range.
    """

    config_class = WidowXRosConfig
    name = "widowx_ros"

    def __init__(self, config: WidowXRosConfig):
        if not ROS_AVAILABLE:
            raise ImportError(
                "ROS dependencies not available. Please install: "
                "pip install interbotix-xseries-modules"
            )

        super().__init__(config)
        self.config = config
        assert config.side is not None  # filled in by WidowXRosConfig.__post_init__
        self.side: str = config.side
        self.dt = 1.0 / config.fps
        self.gripper_joint_close, self.gripper_joint_open = MASTER_GRIPPER_JOINT_RANGE[self.side]

        # Initialize ROS robot. This only connects to the arm's ROS services; motor modes and
        # torque are set up in connect().
        self.bot = InterbotixManipulatorXS(
            robot_model=config.robot_model,
            group_name="arm",
            gripper_name="gripper",
            robot_name=config.robot_name,
            init_node=config.init_ros_node
        )

        self._connected = False
        # True once shutdown() cut the torque at the rest pose; disconnect() then has nothing to do.
        self._powered_down = False
        # Position P gain of each arm joint, read in configure(); unlock() ramps them down to 0 and
        # puts them back once the torque is off.
        self._arm_p_gains: list[int] = []
        # Clutch-style takeover (relative_on_release): the follower pose the arm was last driven
        # onto, the arm-minus-follower offset applied by get_action(), and the last arm action it
        # returned (the follower's command while the offset applies).
        self._reference: list[float] | None = None
        self._relative = False
        self._offset = [0.0] * len(ARM_MOTORS)
        self._last_arm_action: list[float] | None = None

    def normalize_gripper(self, joint_position: float) -> float:
        """Gripper joint angle (rad) -> opening in [0, 1] (not clamped)."""
        return (joint_position - self.gripper_joint_close) / (self.gripper_joint_open - self.gripper_joint_close)

    def unnormalize_gripper(self, opening: float) -> float:
        """Opening in [0, 1] -> gripper joint angle (rad)."""
        return self.gripper_joint_close + opening * (self.gripper_joint_open - self.gripper_joint_close)

    @property
    def action_features(self) -> dict[str, type]:
        """Action features: joint positions."""
        return {f"{motor}.pos": float for motor in (*ARM_MOTORS, "gripper")}

    @property
    def feedback_features(self) -> dict[str, type]:
        """The arm can be driven to joint positions (same keys as the action), which makes it an
        actuated teleop: lerobot-rollout's DAgger moves it onto the follower's pose before a takeover."""
        return self.action_features

    @property
    def is_connected(self) -> bool:
        """Whether connect() has set the arm up."""
        return self._connected

    @property
    def is_powered_down(self) -> bool:
        """True once shutdown() left the arm at the rest pose with torque off."""
        return self._powered_down

    def connect(self, calibrate: bool = True) -> None:
        """Set the arm up, optionally move it to the start pose and release it to the operator."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.configure()
        self._connected = True
        if self.config.move_to_start_on_connect:
            self.move_to_start()
        if self.config.release_on_connect:
            logger.warning(f"{self}: releasing the arm now - hold the handle.")
            self.unlock()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """ROS robots don't use lerobot calibration system."""
        return True

    def calibrate(self) -> None:
        """Calibration is handled by ROS."""
        logger.info("Calibration is handled by ROS system. Skipping lerobot calibration.")

    def configure(self) -> None:
        """Master setup (ALOHA2 opening_ceremony): arm in position mode with a time-based profile,
        gripper in position mode, torque on (the arm is held until unlock()).

        Writing the operating mode goes to EEPROM and briefly drops the arm's torque, so this runs
        once, while the arm rests.
        """
        set_arm_position_time_profile(self.bot, START_ARM_MOVE_TIME_S, START_ARM_MOVE_TIME_S / 4)
        self.bot.dxl.robot_set_operating_modes("single", "gripper", "position")
        self._arm_p_gains = [
            int(gain)
            for gain in self.bot.dxl.robot_get_motor_registers("group", "arm", "Position_P_Gain").values
        ]
        torque_on(self.bot)
        self._powered_down = False
        logger.info(f"WidowXRos {self.config.robot_name} setup complete")

    def _set_arm_p_gains(self, gains: list[int]) -> None:
        if len(set(gains)) == 1:
            self.bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", gains[0])
            return
        for joint, gain in zip(self.bot.arm.group_info.joint_names, gains, strict=True):
            self.bot.dxl.robot_set_motor_registers("single", joint, "Position_P_Gain", gain)

    def lock(self) -> None:
        """Torque the arm and gripper on: the arm holds its current pose (pause / before moving it).
        Ends a clutch-style takeover: the offset is cleared, and the follower's last command becomes
        the reference for the next release."""
        if self._relative and self._last_arm_action is not None:
            self._reference = list(self._last_arm_action)
        self._relative = False
        self._offset = [0.0] * len(ARM_MOTORS)
        # The goal registers still hold the pose the arm was released at; aim them at the present
        # pose first so torquing on cannot pull the arm back there.
        position = self.bot.dxl.joint_states.position
        move_time = max(2 * self.dt, 0.05)
        self.bot.arm.publish_positions(
            list(position[:6]), moving_time=move_time, accel_time=move_time / 2, blocking=False
        )
        command = JointSingleCommand(name="gripper")
        command.cmd = float(position[6])
        self.bot.gripper.core.pub_single.publish(command)
        torque_on(self.bot)

    def unlock(self) -> None:
        """Torque the arm and gripper off so the operator can move them. With `release_ramp_s` > 0
        the arm's position P gains are first lowered to 0 over that time, so the arm softens into
        the operator's hand instead of dropping."""
        ramp_s = self.config.release_ramp_s
        if ramp_s > 0 and self._arm_p_gains:
            steps = max(int(ramp_s / 0.04), 1)
            for step in range(1, steps + 1):
                scale = 1.0 - step / steps
                self._set_arm_p_gains([round(gain * scale) for gain in self._arm_p_gains])
                time.sleep(ramp_s / steps)
        torque_off(self.bot)
        if ramp_s > 0 and self._arm_p_gains:
            # The torque is off, so restoring the gains moves nothing; every later torque_on()
            # finds the arm stiff again.
            self._set_arm_p_gains(self._arm_p_gains)
        if self.config.relative_on_release and self._reference is not None:
            self._engage_relative(self._reference)

    def _engage_relative(self, reference: list[float]) -> None:
        """Wait for the released arm to settle in the operator's hand, then store the offset
        between its pose and `reference` (the follower's pose), see `relative_on_release`."""
        cfg = self.config
        start = time.perf_counter()
        window: list[tuple[float, list[float]]] = []
        settled = False
        while True:
            now = time.perf_counter()
            arm, _ = self._read_joints()
            window = [(t, q) for t, q in window if now - t <= cfg.settle_s] + [(now, arm)]
            if now - window[0][0] >= cfg.settle_s * 0.9 and all(
                max(q[j] for _, q in window) - min(q[j] for _, q in window) <= cfg.settle_tolerance
                for j in range(len(ARM_MOTORS))
            ):
                settled = True
                break
            if now - start >= cfg.settle_timeout_s:
                break
            time.sleep(0.01)
        self._offset = [q - r for q, r in zip(arm, reference, strict=True)]
        self._last_arm_action = list(reference)
        self._relative = True
        worst = max(range(len(ARM_MOTORS)), key=lambda j: abs(self._offset[j]))
        message = (
            f"{self}: relative control from here ({'settled' if settled else 'settle timeout'} after "
            f"{time.perf_counter() - start:.2f}s, largest offset {self._offset[worst]:+.3f} rad on "
            f"{ARM_MOTORS[worst]})"
        )
        if abs(self._offset[worst]) > cfg.max_offset_warn:
            logger.warning(message + " - leader and follower poses differ visibly; pause to re-align.")
        else:
            logger.info(message)

    # enable_torque / disable_torque: the names lerobot's actuated-teleop handover looks for
    def enable_torque(self) -> None:
        self.lock()

    def disable_torque(self) -> None:
        self.unlock()

    def move_to_start(self, start_open_ratio: float | None = None) -> None:
        """Move the arm to START_ARM_POSE and park the gripper at `start_open_ratio` (0 = closed,
        1 = open; default from the config). The arm stays locked afterwards."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        ratio = self.config.start_open_ratio if start_open_ratio is None else start_open_ratio
        torque_on(self.bot)
        move_arms([self.bot], [START_ARM_POSE[:6]], move_time=START_ARM_MOVE_TIME_S)
        move_grippers(
            [self.bot], [self.unnormalize_gripper(ratio)], move_time=START_GRIPPER_MOVE_TIME_S, dt=self.dt
        )
        self._powered_down = False

    def _move_to_rest(self, move_times: tuple[float, ...]) -> None:
        # Two stages (old sleep pose, then the measured rest pose), gripper opened first: see
        # sleep_modify.py and the sleep-pose comment in constants.py.
        torque_on(self.bot)
        move_grippers([self.bot], [self.gripper_joint_open], move_time=move_times[0], dt=self.dt)
        for pose, move_time in zip((OLD_MASTER_SLEEP_POSITION, MASTER_SLEEP_POSITION), move_times, strict=True):
            move_arms([self.bot], [pose], move_time=move_time)

    def sleep(self) -> None:
        """Go to the rest pose and keep holding it (torque stays on), like `sleep_modify.py --sleep`."""
        self._move_to_rest(MASTER_SLEEP_MOVE_TIMES)

    def shutdown(self) -> None:
        """Go to the rest pose slowly, then cut torque, like `sleep_modify.py --shut_down`."""
        self._move_to_rest(MASTER_SHUTDOWN_MOVE_TIMES)
        time.sleep(0.3)  # settle margin: move_arms() already waited for the move itself
        torque_off(self.bot)
        self._powered_down = True

    def get_action(self) -> dict[str, float]:
        """Get current joint positions from master arm (gripper as its normalised opening). During a
        clutch-style takeover (see `relative_on_release`) the arm joints are offset onto the
        follower: arm pose - offset."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        arm, gripper = self._read_joints()
        arm = [q - offset for q, offset in zip(arm, self._offset, strict=True)]
        self._last_arm_action = arm
        action_dict = {f"{motor}.pos": arm[i] for i, motor in enumerate(ARM_MOTORS)}
        action_dict["gripper.pos"] = self.normalize_gripper(gripper)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        return action_dict

    def _read_joints(self) -> tuple[list[float], float]:
        """The arm's six joint angles and the gripper joint angle (rad), as measured."""
        joint_states: Any = self.bot.dxl.joint_states
        if joint_states is None or len(joint_states.position) < 7:
            raise RuntimeError(f"{self}: joint states not available")
        position = joint_states.position
        return [float(position[i]) for i in range(len(ARM_MOTORS))], float(position[6])

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Drive the (torqued) arm towards joint positions: "<joint>.pos" in rad, "gripper.pos" as the
        normalised opening. Keys that are missing keep their current value. Meant to be called at
        ~30 Hz with interpolated targets (lerobot's teleop_smooth_move_to)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        current, _ = self._read_joints()
        arm_target = [float(feedback.get(f"{motor}.pos", current[i])) for i, motor in enumerate(ARM_MOTORS)]
        # The follower pose a clutch-style takeover is referenced to (see `relative_on_release`)
        self._reference = arm_target
        # Short trajectory per command so consecutive interpolation steps blend into one motion
        move_time = max(2 * self.dt, 0.05)
        self.bot.arm.publish_positions(arm_target, moving_time=move_time, accel_time=move_time / 2, blocking=False)
        if "gripper.pos" in feedback:
            command = JointSingleCommand(name="gripper")
            command.cmd = self.unnormalize_gripper(min(max(float(feedback["gripper.pos"]), 0.0), 1.0))
            self.bot.gripper.core.pub_single.publish(command)

    def disconnect(self) -> None:
        """Walk the arm to the rest pose and power it off (unless shutdown() already did)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._powered_down:
            try:
                logger.info(f"Moving {self} to the rest pose and powering off...")
                self.shutdown()
            except Exception as e:
                logger.warning(f"Error during shutdown sequence: {e}. Continuing with disconnect...")

        self._connected = False
        logger.info(f"{self} disconnected.")
