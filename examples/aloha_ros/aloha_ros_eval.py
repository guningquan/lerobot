#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
ALOHA ROS Bimanual Evaluation Script (ALOHA2)

This script evaluates a trained policy on ALOHA ROS dual-arm system (ViperX followers via ROS).

Before running, start the arms and cameras and keep that terminal open:
    roslaunch aloha 4arms_tactile_teleop.launch

Usage (use the fps the training dataset was recorded at):

    # Policy trained on a dataset recorded with aloha_ros_record.py (or lerobot-record)
    python examples/aloha_ros/aloha_ros_eval.py --fps 30 --policy_path <.../pretrained_model>

    # Policy trained on a dataset converted from ALOHA2 (act-triple-plus-aloha2) HDF5 episodes with
    # openpi's convert_aloha_data_to_lerobot*.py, e.g. local/packaging_lerobot: gripper state =
    # ALOHA2 qpos, plus observation.velocity / observation.effort
    python examples/aloha_ros/aloha_ros_eval.py --data_format aloha2 --fps 50 \
        --policy_path <.../pretrained_model> --task "packaging the items"

The defaults of every flag are the constants below. The policy's normalisation statistics come
from the checkpoint itself (like lerobot-rollout), so no dataset is needed.

Keys (n / r / q work too):
    → (Right arrow)  start an episode (the arms wait at the start pose) / end the current one early
    ← (Left arrow)   reset: discard the current episode, back to the start pose, run it again
    ESC (or q)       stop: rest pose (two stages) and power off
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from lerobot.cameras.ros_image import RosImageCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.robots.aloha_ros import AlohaRos, AlohaRosConfig
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.common.control_utils import predict_action
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.robot_utils import precise_sleep

# Import constants from aloha_scripts/constants.py
script_dir = os.path.dirname(os.path.abspath(__file__))
aloha_scripts_dir = os.path.join(script_dir, 'aloha_scripts')
if aloha_scripts_dir not in sys.path:
    sys.path.insert(0, aloha_scripts_dir)

from constants import (
    FPS,
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN,
    PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN,
    ROS_CAMERA_HEIGHT,
    ROS_CAMERA_TOPICS,
    ROS_CAMERA_WIDTH,
)


# Evaluation configuration (defaults of the command line flags)
NUM_EPISODES = 5
EPISODE_TIME_SEC = 30
POLICY_PATH = "/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint/pi05_apple_grasping_theodore/pretrained_model"
# Language instruction for VLA policies (e.g. MolmoAct2, pi0.5); None for policies without one (ACT)
TASK_DESCRIPTION = None

# Policy configuration overrides (optional)
# Set to None to use values from config.json, or set specific values to override
POLICY_TEMPORAL_ENSEMBLE_COEFF = None  # e.g., 0.01 to enable temporal aggregation
POLICY_N_ACTION_STEPS = None  # e.g., 1 (required if temporal_ensemble_coeff is set)

# Video saving configuration
SAVE_VIDEOS = True  # Set to True to save videos of evaluation episodes
VIDEO_OUTPUT_DIR = None  # Directory to save videos; None = the policy path

# Gripper opening the puppets park at before each episode (0.0 = closed, 1.0 = open)
START_OPEN_RATIO = 1.0

# --data_format aloha2: layout of datasets converted from ALOHA2 HDF5 episodes. 14 values per
# feature, LEFT arm first; per arm 6 joints + gripper, exactly as ALOHA2's real_env.RealEnv records:
#   observation.state     joints (rad) + PUPPET_GRIPPER_POSITION_NORMALIZE_FN(finger position)
#   observation.velocity  joint velocities + PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(finger velocity)
#   observation.effort    raw effort of the 6 joints + gripper
#   action                joints (rad) + normalised master gripper (same meaning as AlohaRos)
ALOHA2_ARM_ORDER = ("left", "right")
ALOHA2_MOTORS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")
ALOHA2_ACTION_KEYS = [f"{side}_{motor}.pos" for side in ALOHA2_ARM_ORDER for motor in ALOHA2_MOTORS]
# openpi's convert_aloha_data_to_lerobot*.py *labels* these 14 dims right arm first, but copies
# /observations/qpos and /action unchanged, i.e. left arm first (checked against the HDF5 of
# packaging_lerobot). The labels are wrong; they are only used to recognise checkpoints trained on
# such a dataset (MolmoAct2 stores them in policy.dataset_feature_names).
OPENPI_ALOHA_FEATURE_LABELS = [f"{side}_{motor}" for side in ("right", "left") for motor in ALOHA2_MOTORS]


def make_camera_config(fps: int) -> dict[str, RosImageCameraConfig]:
    """The four cameras, read from the topics the launch publishes."""
    return {
        name: RosImageCameraConfig(topic=topic, width=ROS_CAMERA_WIDTH, height=ROS_CAMERA_HEIGHT, fps=fps)
        for name, topic in ROS_CAMERA_TOPICS.items()
    }


def aloha2_arm_state(arm: Any) -> tuple[list[float], list[float], list[float]]:
    """qpos / qvel / effort of one puppet arm (a ViperXRos), as ALOHA2's RealEnv records them."""
    joint_states = arm.bot.dxl.joint_states
    if joint_states is None or len(joint_states.position) < 8 or len(joint_states.velocity) < 8:
        raise RuntimeError(f"{arm}: joint states not available")
    position, velocity, effort = joint_states.position, joint_states.velocity, joint_states.effort
    qpos = [*position[:6], PUPPET_GRIPPER_POSITION_NORMALIZE_FN(position[7])]
    qvel = [*velocity[:6], PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(velocity[7])]
    return qpos, qvel, list(effort[:7])


def aloha2_observation_frame(robot: AlohaRos, obs: dict[str, Any]) -> dict[str, Any]:
    """Policy input in the --data_format aloha2 layout; camera images are taken from `obs`."""
    state, velocity, effort = [], [], []
    for side in ALOHA2_ARM_ORDER:
        qpos, qvel, eff = aloha2_arm_state(robot.right_arm if side == "right" else robot.left_arm)
        state += qpos
        velocity += qvel
        effort += eff
    frame: dict[str, Any] = {
        OBS_STATE: np.asarray(state, dtype=np.float32),
        f"{OBS_STR}.velocity": np.asarray(velocity, dtype=np.float32),
        f"{OBS_STR}.effort": np.asarray(effort, dtype=np.float32),
    }
    for cam_name in robot.cameras:
        frame[f"{OBS_STR}.images.{cam_name}"] = obs[cam_name]
    return frame


def make_observation_frame(robot: AlohaRos, obs: dict[str, Any], data_format: str) -> dict[str, Any]:
    """Policy input for one control tick, in the layout given by --data_format."""
    if data_format == "aloha2":
        return aloha2_observation_frame(robot, obs)
    from lerobot.utils.feature_utils import build_dataset_frame
    return build_dataset_frame(
        {**hw_to_dataset_features(robot.observation_features, OBS_STR)},
        obs,
        prefix=OBS_STR
    )


def check_policy_matches_data_format(
    policy_cfg: PreTrainedConfig, robot: AlohaRos, data_format: str, task: str | None
) -> None:
    """Refuse to run a policy whose inputs / joint order do not match what this script feeds it."""
    if data_format == "aloha2":
        provided = {OBS_STATE, f"{OBS_STR}.velocity", f"{OBS_STR}.effort"}
        provided |= {f"{OBS_STR}.images.{cam}" for cam in robot.cameras}
        expected_names = {ACTION: OPENPI_ALOHA_FEATURE_LABELS, OBS_STATE: OPENPI_ALOHA_FEATURE_LABELS}
    else:
        provided = set(hw_to_dataset_features(robot.observation_features, OBS_STR))
        expected_names = {
            ACTION: hw_to_dataset_features(robot.action_features, ACTION)[ACTION]["names"],
            OBS_STATE: hw_to_dataset_features(robot.observation_features, OBS_STR)[OBS_STATE]["names"],
        }
    hint = (
        " If the policy was trained on a dataset converted from ALOHA2 HDF5 episodes (openpi's "
        "convert_aloha_data_to_lerobot, with observation.velocity / observation.effort), run with "
        "--data_format aloha2; if it was trained on data recorded with aloha_ros_record.py, use "
        "--data_format lerobot."
    )

    missing = sorted(set(policy_cfg.input_features) - provided)
    if missing:
        raise ValueError(f"The policy expects inputs {missing} that --data_format {data_format} does not provide.{hint}")

    # Policies that store their training feature names (e.g. MolmoAct2) let the joint order be checked
    trained_names = getattr(policy_cfg, "dataset_feature_names", None) or {}
    for key, expected in expected_names.items():
        names = trained_names.get(key)
        if names and list(names) != list(expected):
            raise ValueError(
                f"The policy was trained with {key} = {list(names)}, but --data_format {data_format} "
                f"provides {list(expected)}.{hint}"
            )

    if task is None and policy_cfg.type in ("molmoact2", "pi0", "pi05", "smolvla", "xvla", "groot"):
        raise ValueError(f"A {policy_cfg.type} policy needs a language instruction: pass --task.")


def load_policy(policy_cfg: PreTrainedConfig) -> PreTrainedPolicy:
    """Load the policy weights from policy_cfg.pretrained_path (same as lerobot-rollout)."""
    policy_class = get_policy_class(policy_cfg.type)
    if not getattr(policy_cfg, "use_peft", False):
        return policy_class.from_pretrained(policy_cfg.pretrained_path, config=policy_cfg)

    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(policy_cfg.pretrained_path)
    policy = policy_class.from_pretrained(peft_config.base_model_name_or_path, config=policy_cfg)
    return PeftModel.from_pretrained(policy, policy_cfg.pretrained_path, config=peft_config)


def wait_for_start(events: dict) -> bool:
    """Block until → (start) or ESC / q (quit). Returns False on ESC / q."""
    print("Press → (Right arrow) to start this episode, ESC (or q) to quit.")
    events["exit_early"] = False
    events["rerecord_episode"] = False
    while True:
        if events["stop_recording"]:
            return False
        if events["exit_early"]:
            started = not events["rerecord_episode"]  # ← only sets exit_early together with rerecord
            events["exit_early"] = False
            events["rerecord_episode"] = False
            if started:
                return True
        time.sleep(0.02)


def eval_loop(
    robot: AlohaRos,
    policy: PreTrainedPolicy,
    preprocessor: Any,
    postprocessor: Any,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    control_time_s: float,
    fps: int,
    events: dict,
    data_format: str = "lerobot",
    task: str | None = None,
    display_data: bool = True,
    save_video: bool = False,
    video_writers: dict[str, cv2.VideoWriter] | None = None,
    episode_index: int = 0,
):
    """Main evaluation loop that runs policy on robot."""
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Process observation
        obs_processed = robot_observation_processor(obs)

        # Build observation frame for policy
        observation_frame = make_observation_frame(robot, obs_processed, data_format)

        # Get action from policy (a chunking policy only runs the model when its action queue is empty)
        device = get_safe_torch_device(policy.config.device)
        inference_start = time.perf_counter()
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )
        inference_s = time.perf_counter() - inference_start
        if inference_s > 1.0 / fps:
            print(f"Policy inference took {inference_s * 1e3:.0f} ms (one control tick is {1e3 / fps:.0f} ms)")

        # Convert to robot action format
        if data_format == "aloha2":
            # Left arm first; the gripper is already the normalised opening AlohaRos expects
            act_processed = make_robot_action(action_values, {ACTION: {"names": ALOHA2_ACTION_KEYS}})
        else:
            action_features = hw_to_dataset_features(robot.action_features, ACTION)
            act_processed = make_robot_action(action_values, action_features)

        # Process action through robot action processor
        act_final = robot_action_processor((act_processed, obs_processed))


        print(f"Action: {act_final}")
        # "*_gripper.pos" is the normalised gripper opening (0 = closed, 1 = open); AlohaRos maps it
        # to each puppet's own gripper range, so no per-script gripper mapping is needed here.
        # input("Press Enter to continue...")

        # Send action to robot
        robot.send_action(act_final)

        # # Log to rerun if enabled
        # if display_data:
        #     log_rerun_data(obs_processed, act_final, timestamp)

        # Save video frames if enabled
        if save_video and video_writers is not None:
            # Save frames from all cameras
            for cam_name in robot.cameras.keys():
                if cam_name in obs_processed:
                    frame = obs_processed[cam_name]
                    # Convert to BGR if needed (OpenCV uses BGR)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # Assume RGB, convert to BGR
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    if cam_name in video_writers:
                        video_writers[cam_name].write(frame_bgr)

        # Sleep to maintain fps
        elapsed = time.perf_counter() - start_loop_t
        sleep_time = max(1.0 / fps - elapsed, 0.0)
        precise_sleep(sleep_time)

        timestamp = time.perf_counter() - start_episode_t


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fps", type=int, default=FPS,
        help="Control rate in Hz; use the fps of the training dataset, e.g. 30 or 50 (default: %(default)s).",
    )
    parser.add_argument(
        "--data_format", choices=("lerobot", "aloha2"), default="lerobot",
        help="Observation/action layout the policy was trained on: 'lerobot' = recorded with "
        "aloha_ros_record.py / lerobot-record, 'aloha2' = converted from ALOHA2 HDF5 episodes "
        "(default: %(default)s).",
    )
    parser.add_argument("--policy_path", default=POLICY_PATH, help="The checkpoint's pretrained_model directory.")
    parser.add_argument("--task", default=TASK_DESCRIPTION, help="Language instruction for VLA policies.")
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--episode_time_s", type=float, default=EPISODE_TIME_SEC)
    args = parser.parse_args()
    fps = args.fps
    policy_path = args.policy_path

    print("\n" + "="*60)
    print("ALOHA ROS EVALUATION SCRIPT")
    print("="*60)
    print(f"Policy path: {policy_path}")
    print(f"Data format: {args.data_format}")
    print(f"Task: {args.task}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Episode duration: {args.episode_time_s} seconds")
    print(f"Control rate: {fps} Hz")
    print("="*60 + "\n")

    # Create puppet arms config (robots). connect() moves the arms to the start pose.
    puppet_config = AlohaRosConfig(
        id="aloha_puppet",
        puppet_left_robot_name="puppet_left",
        puppet_right_robot_name="puppet_right",
        robot_model="vx300s",
        init_ros_node=True,
        fps=fps,
        move_to_start_on_connect=True,
        start_open_ratio=START_OPEN_RATIO,
        cameras=make_camera_config(fps),
    )

    # Initialize the robot
    puppet = AlohaRos(puppet_config)

    # Load policy config and check it matches what this script will feed it, before any motion
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    check_policy_matches_data_format(policy_cfg, puppet, args.data_format, args.task)

    # Apply policy configuration overrides if specified
    if POLICY_TEMPORAL_ENSEMBLE_COEFF is not None:
        policy_cfg.temporal_ensemble_coeff = POLICY_TEMPORAL_ENSEMBLE_COEFF
        print(f"Overriding temporal_ensemble_coeff: {POLICY_TEMPORAL_ENSEMBLE_COEFF}")

    if POLICY_N_ACTION_STEPS is not None:
        policy_cfg.n_action_steps = POLICY_N_ACTION_STEPS
        print(f"Overriding n_action_steps: {POLICY_N_ACTION_STEPS}")

    # # Validate configuration after overrides
    # if policy_cfg.temporal_ensemble_coeff is not None and policy_cfg.n_action_steps > 1:
    #     raise ValueError(
    #         "`n_action_steps` must be 1 when using temporal ensembling. "
    #         f"Current values: temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff}, "
    #         f"n_action_steps={policy_cfg.n_action_steps}. "
    #         "Please set POLICY_N_ACTION_STEPS=1 if enabling temporal aggregation."
    #     )

    # Load policy
    print(f"\nLoading policy from {policy_path}...")
    policy = load_policy(policy_cfg)
    policy.eval()
    print(f"Policy loaded: {policy_cfg.type}")
    print(f"Device: {policy.config.device}")

    # Create processors
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # Load preprocessor and postprocessor (with their normalisation statistics) from the checkpoint
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": policy.config.device},
        },
    )

    # Connect the robot (motor setup, then both arms move to the start pose)
    print("\nConnecting to robot and moving to the start pose...")
    puppet.connect()

    if not puppet.is_connected:
        raise ValueError("Robot is not connected!")

    # Initialize keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="aloha_ros_eval")

    print("\n" + "="*60)
    print("EVALUATION SETUP COMPLETE")
    print("="*60)
    print(f"Episodes to evaluate: {args.num_episodes}")
    print(f"Episode duration: {args.episode_time_s} seconds")
    print("\nKeyboard Controls:")
    print("  → (Right Arrow, or n): Start an episode (arms at the start pose) / end the current one early")
    print("  ← (Left Arrow, or r):  Reset - discard the current episode, back to the start pose, run it again")
    print("  ESC (or q):            Stop evaluation, rest pose and power off")
    print("="*60 + "\n")

    evaluated_episodes = 0
    try:
        # Warm up the policy once at the start pose: the first inference compiles the model /
        # captures CUDA graphs (~20 s for MolmoAct2), which would otherwise stall the first episode.
        # eval_loop() resets the policy, so this action is discarded.
        print("Warming up the policy (first inference)...")
        warmup_start = time.perf_counter()
        predict_action(
            observation=make_observation_frame(
                puppet, robot_observation_processor(puppet.get_observation()), args.data_format
            ),
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=args.task,
            robot_type=puppet.robot_type,
        )
        print(f"Warm-up inference took {time.perf_counter() - warmup_start:.1f} s")

        # Create video output directory if saving videos
        video_writers = None
        if SAVE_VIDEOS:
            video_output_path = Path(VIDEO_OUTPUT_DIR or policy_path)
            video_output_path.mkdir(parents=True, exist_ok=True)
            print(f"Videos will be saved to: {video_output_path}")

        # Run evaluation episodes; each one starts from the start pose when → is pressed
        while evaluated_episodes < args.num_episodes and not events["stop_recording"]:
            current_episode_num = evaluated_episodes + 1
            print(f"\n{'='*60}")
            print(f"EPISODE {current_episode_num} of {args.num_episodes}")
            print(f"{'='*60}")
            if listener is None:
                print("No keyboard listener available: starting right away.")
            elif not wait_for_start(events):
                break
            log_say(f"Evaluating episode {current_episode_num}")

            # Initialize video writers for this episode if saving videos
            if SAVE_VIDEOS:
                video_writers = {}
                for cam_name in puppet.cameras.keys():
                    video_path = video_output_path / f"episode_{current_episode_num:03d}_{cam_name}.mp4"
                    # Get camera dimensions
                    cam_config = puppet.config.cameras[cam_name]
                    width = cam_config.width
                    height = cam_config.height
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        float(fps),
                        (width, height)
                    )
                    if not video_writer.isOpened():
                        print(f"Warning: Could not open video writer for {cam_name}")
                    else:
                        video_writers[cam_name] = video_writer
                        print(f"Recording video for {cam_name}: {video_path}")

            # Run evaluation loop (the first commands are eased in from the start pose)
            eval_loop(
                robot=puppet,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                control_time_s=args.episode_time_s,
                fps=fps,
                events=events,
                data_format=args.data_format,
                task=args.task,
                display_data=True,
                save_video=SAVE_VIDEOS,
                video_writers=video_writers,
                episode_index=current_episode_num,
            )

            # Close video writers for this episode
            if SAVE_VIDEOS and video_writers is not None:
                for cam_name, writer in video_writers.items():
                    writer.release()
                    print(f"Video saved for {cam_name}")
                video_writers = None

            if events["stop_recording"]:
                # ESC / q during the episode: stop right away (finally: rest pose and power off)
                print(f"\nEpisode {current_episode_num} stopped (not counted).")
                break

            if events["rerecord_episode"]:
                # ← : discard this attempt and run the same episode again (its videos get overwritten)
                events["rerecord_episode"] = False
                print(f"\nEpisode {current_episode_num} reset (not counted). Moving robots back to the start pose...")
                log_say("Reset episode")
                puppet.move_to_start()
                continue

            evaluated_episodes += 1
            print(f"\nEpisode {current_episode_num} completed.")
            if evaluated_episodes < args.num_episodes:
                # Move back to the start pose for next episode
                print("Moving robots back to the start pose...")
                puppet.move_to_start()
    finally:
        # Clean up
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Evaluated {evaluated_episodes} episodes")
        print("="*60 + "\n")

        log_say("Evaluation complete")
        # Rest pose (two stages) and torque off
        puppet.disconnect()
        if listener:
            listener.stop()


if __name__ == "__main__":
    main()
