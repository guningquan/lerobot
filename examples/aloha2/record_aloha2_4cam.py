"""
ALOHA2 4-Camera Recording Script with RealSense cameras.

Records bimanual teleoperation episodes using:
- 2x ViperX-300 follower arms (left/right puppets)
- 2x WidowX-250 leader arms (left/right masters)
- 4x Intel RealSense cameras (cam_high, cam_low, cam_left_wrist, cam_right_wrist)

Adapted for LeRobot v0.5.2 API.

Usage:
  conda activate lerobot
  export WANDB_API_KEY="..." WANDB_ENTITY="theoliu"
  python record_aloha2_4cam.py

Set RESUME = True to continue appending to an existing dataset.
"""

from pathlib import Path

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.robots.aloha import Aloha, AlohaConfig
from lerobot.teleoperators.aloha_teleop import AlohaTeleop, AlohaTeleopConfig
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# ── Recording configuration ──────────────────────────────────────────
NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 120
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "Describe your task here"
REPO_ID = "theodoreliu/aloha2_4cam_task"  # CHANGE the dataset name
RESUME = False  # Set True to append to existing dataset
DATASET_ROOT = "/home/robot/Dataset_and_Checkpoint/lerobot-dataset"
VIZ_VIDEO_DIR = "/home/robot/videos/visualize/lerobot/act"
CKPT_DIR = "/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint"
DEPLOY_VIDEO_DIR = "/home/robot/videos/deploy_third_person/lerobot/act"

# ── Camera configuration (4 RealSense cameras by serial number) ─────
# Serial numbers verified via: lerobot-find-cameras realsense
CAMERA_CONFIG = {
    "cam_high": RealSenseCameraConfig(
        serial_number_or_name="109422062625",  # D415 overhead
        fps=FPS, width=640, height=480,
    ),
    "cam_low": RealSenseCameraConfig(
        serial_number_or_name="936322072119",  # D435 worms-eye
        fps=FPS, width=640, height=480,
    ),
    "cam_left_wrist": RealSenseCameraConfig(
        serial_number_or_name="134222077139",  # D435i left wrist
        fps=FPS, width=640, height=480,
    ),
    "cam_right_wrist": RealSenseCameraConfig(
        serial_number_or_name="943222070893",  # D435i right wrist
        fps=FPS, width=640, height=480,
    ),
}

# ── Calibration directory ────────────────────────────────────────────
CALIB_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"

# ── ALOHA Robot configuration (dual ViperX-300 follower arms) ────────
robot_config = AlohaConfig(
    id="aloha",
    left_arm_port="/dev/ttyDXL_puppet_left",
    right_arm_port="/dev/ttyDXL_puppet_right",
    left_arm_max_relative_target=20.0,
    right_arm_max_relative_target=20.0,
    left_arm_use_degrees=True,
    right_arm_use_degrees=True,
    calibration_dir=CALIB_DIR,
    cameras=CAMERA_CONFIG,
)

# ── ALOHA Teleoperator configuration (dual WidowX-250 leader arms) ───
# NOTE: Both leader arms have XC430-W150 gripper motors (model 1070).
teleop_config = AlohaTeleopConfig(
    id="aloha_teleop",
    left_arm_port="/dev/ttyDXL_master_left",
    right_arm_port="/dev/ttyDXL_master_right",
    left_arm_gripper_motor="xc430-w150",
    right_arm_gripper_motor="xc430-w150",
    left_arm_use_degrees=True,
    right_arm_use_degrees=True,
    calibration_dir=CALIB_DIR,
)

# ── Processors (LeRobot v0.4.3 requires explicit processor pipelines) ─
teleop_action_processor, robot_action_processor, robot_observation_processor = \
    make_default_processors()

# ── Initialize ────────────────────────────────────────────────────────
robot = Aloha(robot_config)
teleop = AlohaTeleop(teleop_config)

action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

if RESUME:
    log_say(f"Resuming existing dataset: {REPO_ID}")
    dataset = LeRobotDataset(repo_id=REPO_ID, root=DATASET_ROOT)
    if hasattr(robot, "cameras") and len(robot.cameras) > 0:
        dataset.start_image_writer(
            num_processes=0,
            num_threads=4 * len(robot.cameras),
        )
    sanity_check_dataset_robot_compatibility(dataset, robot, FPS, dataset_features)
    log_say(f"Resumed dataset with {dataset.num_episodes} existing episodes")
else:
    log_say(f"Creating new dataset: {REPO_ID}")
    sanity_check_dataset_name(REPO_ID, None)
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4 * len(robot.cameras),
        root=DATASET_ROOT,
    )

# ── Keyboard listener and visualization ──────────────────────────────
_, events = init_keyboard_listener()
init_rerun(session_name="aloha2_recording")

# ── Connect hardware ─────────────────────────────────────────────────
robot.connect()
teleop.connect()

episode_idx = 0
existing_episodes = dataset.num_episodes if RESUME else 0

while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    current_episode = existing_episodes + episode_idx + 1
    log_say(f"Recording episode {current_episode} ({episode_idx + 1}/{NUM_EPISODES})")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# ── Cleanup ──────────────────────────────────────────────────────────
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()

log_say(f"Dataset '{REPO_ID}' now contains {dataset.num_episodes} episodes total")
dataset.push_to_hub()
log_say(f"Dataset pushed to HuggingFace Hub")
