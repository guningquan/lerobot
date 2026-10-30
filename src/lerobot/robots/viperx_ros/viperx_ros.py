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
import numpy as np
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_viperx_ros import ViperXRosConfig

logger = logging.getLogger(__name__)

# Import constants from aloha_scripts/constants.py
import sys
import os

# Add aloha_scripts directory to path
# __file__ is at: src/lerobot/robots/viperx_ros/viperx_ros.py
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
    OLD_PUPPET_SLEEP_POSITION,
    PUPPET_GRIPPER_JOINT_RANGE,
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN,
    PUPPET_SHUTDOWN_MOVE_TIMES,
    PUPPET_SLEEP_MOVE_TIMES,
    PUPPET_SLEEP_POSITION,
    START_ARM_POSE,
)

# Try to import ROS dependencies
try:
    from interbotix_xs_modules.arm import InterbotixManipulatorXS
    from interbotix_xs_msgs.msg import JointSingleCommand
    from robot_utils import move_arms, move_grippers, setup_puppet_bot, torque_off, torque_on
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS dependencies not available. Install interbotix_xs_modules to use ViperXRos.")

ARM_MOTORS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
# Durations of the slow moves to the start pose (ALOHA2 one_side_teleop.py DEFAULT_PREP_ARM_MOVE_TIME).
START_ARM_MOVE_TIME_S = 2.0
START_GRIPPER_MOVE_TIME_S = 0.5


def clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


class GripperContactGuard:
    """Stop closing the puppet gripper after current rises or position stalls.

    Ported from ALOHA2's one_side_teleop.py. Works in gripper joint space (rad): while the
    commanded target is below the actual position (closing) and the gripper has not moved for
    `stall_cycles` ticks, it is pressing on something, so the command is frozen at the actual
    position instead of letting the position controller push ever harder. Opening the target by
    more than `release_margin` past that hold point releases it.
    """

    def __init__(
        self,
        name: str,
        stall_error: float,
        stall_motion: float,
        stall_cycles: int,
        release_margin: float,
        effort_threshold: float | None = None,
    ):
        self.name = name
        self.stall_error = stall_error
        self.stall_motion = stall_motion
        self.stall_cycles = stall_cycles
        self.release_margin = release_margin
        self.effort_threshold = effort_threshold
        self.reset()

    def reset(self) -> None:
        self.previous_position: float | None = None
        self.stall_count = 0
        self.hold_position: float | None = None

    def filter_command(self, desired_position: float, actual_position: float, effort: float | None = None) -> float:
        desired_position = float(desired_position)
        actual_position = float(actual_position)

        if self.hold_position is not None:
            if desired_position > self.hold_position + self.release_margin:
                self.hold_position = None
                self.stall_count = 0
            else:
                self.previous_position = actual_position
                return self.hold_position

        if self.previous_position is None:
            self.previous_position = actual_position
            return desired_position

        closing_error = actual_position - desired_position
        actual_motion = abs(actual_position - self.previous_position)
        effort_contact = (
            self.effort_threshold is not None
            and effort is not None
            and closing_error > 0
            and abs(float(effort)) >= self.effort_threshold
        )
        if closing_error > self.stall_error and actual_motion < self.stall_motion:
            self.stall_count += 1
        else:
            self.stall_count = 0

        self.previous_position = actual_position
        if effort_contact or self.stall_count >= self.stall_cycles:
            self.hold_position = actual_position
            self.stall_count = 0
            source = "effort/current" if effort_contact else "position stall"
            logger.info(f"{self.name} gripper contact detected by {source}; holding at {self.hold_position:.4f}.")
            return self.hold_position

        return desired_position


class ViperXRos(Robot):
    """
    ViperX Robot using ROS InterbotixManipulatorXS for control.
    Uses lerobot camera system for image capture.
    This is the puppet/follower arm.

    Control follows ALOHA2's act-triple-plus-aloha2/aloha_scripts:
    - the arm runs in position mode with a *time*-based profile, and every command asks the servos
      to arrive in 1 / fps seconds (`fps` in the config);
    - commands go out with publish_positions(), clamped to the joint limits first. Not
      set_joint_positions(): its velocity check compares against a stale moving_time and silently
      rejects a brisk move, after which every later command fails against the same stale baseline
      and the arm freezes;
    - `gripper.pos` (observation and action) is the gripper opening normalised to [0, 1] with this
      arm's own calibration (0 = closed, 1 = open), mapped to the joint angle here;
    - a contact guard stops the gripper from squeezing harder once it stalls on an object;
    - the first commands after connect / move_to_start() are blended in from the current pose;
    - sleep() / shutdown() go to the measured rest pose in two stages before torque is cut.
    """

    config_class = ViperXRosConfig
    name = "viperx_ros"

    def __init__(self, config: ViperXRosConfig):
        if not ROS_AVAILABLE:
            raise ImportError(
                "ROS dependencies not available. Please install: "
                "pip install interbotix-xseries-modules"
            )

        super().__init__(config)
        self.config = config
        assert config.side is not None  # filled in by ViperXRosConfig.__post_init__
        self.side: str = config.side
        self.dt = 1.0 / config.fps
        self.accel_time = self.dt / 2
        self.gripper_joint_close, self.gripper_joint_open = PUPPET_GRIPPER_JOINT_RANGE[self.side]

        # Initialize ROS robot. This only connects to the arm's ROS services; motor modes and
        # torque are set up in connect().
        self.bot = InterbotixManipulatorXS(
            robot_model=config.robot_model,
            group_name="arm",
            gripper_name="gripper",
            robot_name=config.robot_name,
            init_node=config.init_ros_node
        )

        # Setup cameras if provided
        if config.cameras:
            self.cameras = make_cameras_from_configs(config.cameras)
        else:
            self.cameras = {}

        # Gripper command message
        self.gripper_command = JointSingleCommand(name="gripper")
        self.guard = GripperContactGuard(
            f"{config.robot_name}",
            config.gripper_stall_error,
            config.gripper_stall_motion,
            config.gripper_stall_cycles,
            config.gripper_release_margin,
            effort_threshold=config.gripper_effort_threshold,
        )

        self._connected = False
        # True once shutdown() cut the torque at the rest pose; disconnect() then has nothing to do.
        self._powered_down = False
        self._reported_clamped_joints: set[int] = set()
        self._blend_pending = False
        self._blend_s = 0.0
        self._blend_end = 0.0
        self._blend_from_arm = np.zeros(len(ARM_MOTORS))
        self._blend_from_gripper = 0.0

    def normalize_gripper(self, joint_position: float) -> float:
        """Gripper joint angle (rad) -> opening in [0, 1] (not clamped)."""
        return (joint_position - self.gripper_joint_close) / (self.gripper_joint_open - self.gripper_joint_close)

    def unnormalize_gripper(self, opening: float) -> float:
        """Opening in [0, 1] (clamped) -> gripper joint angle (rad)."""
        return self.gripper_joint_close + clamp01(opening) * (self.gripper_joint_open - self.gripper_joint_close)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features."""
        return {f"{motor}.pos": float for motor in (*ARM_MOTORS, "gripper")}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
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
        """Check if robot and cameras are connected."""
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_powered_down(self) -> bool:
        """True once shutdown() left the arm at the rest pose with torque off."""
        return self._powered_down

    def connect(self, calibrate: bool = True) -> None:
        """Set the arm up for teleoperation / policy control and connect the cameras."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        self._connected = True
        if self.config.move_to_start_on_connect:
            self.move_to_start()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """ROS robots don't use lerobot calibration system."""
        return True

    def calibrate(self) -> None:
        """Calibration is handled by ROS."""
        logger.info("Calibration is handled by ROS system. Skipping lerobot calibration.")

    def configure(self) -> None:
        """Puppet setup (ALOHA2 setup_puppet_bot): reboot the gripper motor, arm in position mode with
        a time-based profile at the control cadence, gripper in current-based position mode, torque on.

        Writing the operating mode goes to EEPROM and briefly drops the arm's torque, so this runs
        once, while the arm rests; later cadence changes only touch the RAM profile registers.
        """
        setup_puppet_bot(self.bot, moving_time=self.dt, accel_time=self.accel_time)
        if self.config.gripper_current_limit is not None:
            self._set_gripper_current_limit(self.config.gripper_current_limit)
        self._powered_down = False
        self.guard.reset()
        self.begin_blend()
        logger.info(f"ViperXRos {self.config.robot_name} setup complete ({self.config.fps} Hz)")

    def _set_gripper_current_limit(self, current_limit: int) -> None:
        # Current_Limit is an EEPROM register: it can only be written with the motor's torque off.
        self.bot.dxl.robot_torque_enable("single", "gripper", False)
        try:
            last_error: Exception | None = None
            for register_name in ("Current_Limit", "current_limit"):
                try:
                    self.bot.dxl.robot_set_motor_registers("single", "gripper", register_name, int(current_limit))
                    return
                except Exception as exc:
                    last_error = exc
            logger.warning(f"{self}: could not set the gripper current limit to {current_limit}: {last_error}")
        finally:
            self.bot.dxl.robot_torque_enable("single", "gripper", True)

    def _restore_command_cadence(self) -> None:
        # Slow moves (move_arms) leave the profile at their own duration; per-tick commands need
        # 1 / fps again. set_trajectory_time() only writes RAM registers, so the arm keeps its torque.
        self.bot.arm.moving_time = None
        self.bot.arm.accel_time = None
        self.bot.arm.set_trajectory_time(moving_time=self.dt, accel_time=self.accel_time)

    def begin_blend(self, duration_s: float | None = None) -> None:
        """Ease the next commands in from the arm's current pose over `duration_s` (default:
        `start_blend_s`), instead of jumping straight to them. Use it when control (re)starts, e.g.
        when teleoperation resumes after a pause."""
        self._blend_s = self.config.start_blend_s if duration_s is None else duration_s
        self._blend_pending = True

    def move_to_start(self, start_open_ratio: float | None = None) -> None:
        """Move the arm to START_ARM_POSE and park the gripper at `start_open_ratio` (0 = closed,
        1 = open; default from the config), then return to the per-tick command cadence."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        ratio = self.config.start_open_ratio if start_open_ratio is None else start_open_ratio
        torque_on(self.bot)
        move_arms([self.bot], [START_ARM_POSE[:6]], move_time=START_ARM_MOVE_TIME_S)
        move_grippers(
            [self.bot], [self.unnormalize_gripper(ratio)], move_time=START_GRIPPER_MOVE_TIME_S, dt=self.dt
        )
        self._restore_command_cadence()
        self._powered_down = False
        self.guard.reset()
        self.begin_blend()

    def _move_to_rest(self, move_times: tuple[float, ...]) -> None:
        # Two stages (old sleep pose, then the measured rest pose), gripper opened first: see
        # sleep_modify.py and the sleep-pose comment in constants.py.
        torque_on(self.bot)
        move_grippers([self.bot], [self.gripper_joint_open], move_time=move_times[0], dt=self.dt)
        for pose, move_time in zip((OLD_PUPPET_SLEEP_POSITION, PUPPET_SLEEP_POSITION), move_times, strict=True):
            move_arms([self.bot], [pose], move_time=move_time)

    def sleep(self) -> None:
        """Go to the rest pose and keep holding it (torque stays on), like `sleep_modify.py --sleep`."""
        self._move_to_rest(PUPPET_SLEEP_MOVE_TIMES)
        self.guard.reset()
        self.begin_blend()

    def shutdown(self) -> None:
        """Go to the rest pose slowly, then cut torque, like `sleep_modify.py --shut_down_puppet`."""
        self._move_to_rest(PUPPET_SHUTDOWN_MOVE_TIMES)
        time.sleep(0.3)  # settle margin: move_arms() already waited for the move itself
        torque_off(self.bot)
        self._powered_down = True

    def _joint_states(self) -> Any:
        joint_states = self.bot.dxl.joint_states
        if joint_states is None or len(joint_states.position) < 7:
            raise RuntimeError(f"{self}: joint states not available")
        return joint_states

    def get_observation(self) -> dict[str, Any]:
        """Get observations from robot and cameras."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm positions from ROS
        start = time.perf_counter()
        position = self._joint_states().position
        obs_dict: dict[str, Any] = {f"{motor}.pos": float(position[i]) for i, motor in enumerate(ARM_MOTORS)}
        if self.config.gripper_state_format == "aloha2":
            # Finger position (position[7], metres), normalised exactly like ALOHA2's RealEnv.get_qpos
            obs_dict["gripper.pos"] = float(PUPPET_GRIPPER_POSITION_NORMALIZE_FN(position[7]))
        else:
            obs_dict["gripper.pos"] = self.normalize_gripper(float(position[6]))
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def _clamp_to_joint_limits(self, joint_positions: list[float]) -> list[float]:
        group_info = self.bot.arm.group_info
        clamped = []
        for index, position in enumerate(joint_positions):
            lower = group_info.joint_lower_limits[index]
            upper = group_info.joint_upper_limits[index]
            if not lower <= position <= upper and index not in self._reported_clamped_joints:
                self._reported_clamped_joints.add(index)
                logger.warning(
                    f"{self}: clamping {group_info.joint_names[index]} {position:.3f} to joint limits "
                    f"[{lower:.3f}, {upper:.3f}]; repeated warnings suppressed."
                )
            clamped.append(min(max(position, lower), upper))
        return clamped

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send actions to robot.

        Args:
            action: "waist.pos" ... "wrist_rotate.pos" as joint angles (rad), and "gripper.pos" as
                the gripper opening in [0, 1] (0 = closed, 1 = open).

        Returns:
            The action actually targeted: arm joints clamped to their limits, gripper clamped to [0, 1].
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        arm_target = self._clamp_to_joint_limits([float(action[f"{motor}.pos"]) for motor in ARM_MOTORS])
        gripper_opening = clamp01(float(action["gripper.pos"]))
        gripper_target = self.unnormalize_gripper(gripper_opening)

        now = time.perf_counter()
        if self._blend_pending:
            position = self._joint_states().position
            self._blend_from_arm = np.array(position[:6], dtype=np.float64)
            self._blend_from_gripper = float(position[6])
            self._blend_end = now + self._blend_s
            self._blend_pending = False
        if now < self._blend_end:
            blend = 1.0 - (self._blend_end - now) / self._blend_s
            arm_command = list(self._blend_from_arm + blend * (np.array(arm_target) - self._blend_from_arm))
            gripper_command = self._blend_from_gripper + blend * (gripper_target - self._blend_from_gripper)
        else:
            arm_command = arm_target
            gripper_command = gripper_target

        # Passing moving_time re-applies the per-tick cadence only if something changed it.
        self.bot.arm.publish_positions(
            arm_command, moving_time=self.dt, accel_time=self.accel_time, blocking=False
        )

        if self.config.gripper_contact_guard:
            joint_states = self._joint_states()
            effort = joint_states.effort[6] if len(joint_states.effort) > 6 else None
            gripper_command = self.guard.filter_command(gripper_command, joint_states.position[6], effort=effort)
        self.gripper_command.cmd = gripper_command
        self.bot.gripper.core.pub_single.publish(self.gripper_command)
        self._powered_down = False

        sent = {f"{motor}.pos": value for motor, value in zip(ARM_MOTORS, arm_target, strict=True)}
        sent["gripper.pos"] = gripper_opening
        return sent

    def disconnect(self) -> None:
        """Disconnect from robot and cameras.

        Unless shutdown() already ran, first walks the arm to the rest pose in two stages and cuts
        torque (see shutdown()).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._powered_down:
            try:
                logger.info(f"Moving {self} to the rest pose and powering off...")
                self.shutdown()
            except Exception as e:
                logger.warning(f"Error during shutdown sequence: {e}. Continuing with disconnect...")

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        self._connected = False
        logger.info(f"{self} disconnected.")
