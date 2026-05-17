"""
ALOHA2 Recording Script with 4 RealSense Cameras + 4 XENSE G1-WS Tactile Sensors.

Records bimanual teleoperation episodes using:
- 2x ViperX-300 follower arms (left/right puppets)
- 2x WidowX-250 leader arms (left/right masters)
- 4x Intel RealSense cameras (cam_high, cam_low, cam_left_wrist, cam_right_wrist)
- 4x XENSE G1-WS photonic tactile sensors on follower grippers

The script supports two tactile modes via the TACTILE_MODE variable:
- "simple": 1-channel grayscale (GelSight-equivalent baseline)
- "full":   Multi-channel (Rectify + Depth + Force + ForceNorm + Marker2D)

Usage:
  conda activate lerobot
  export WANDB_API_KEY="..." WANDB_ENTITY="theoliu"
  python record_aloha2_tactile.py
"""

from pathlib import Path

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.robots.aloha import Aloha, AlohaConfig
from lerobot.tactile.config import TactileSensorMode
from lerobot.tactile.xense_g1ws.config_xense_g1ws import XENSEG1WSConfig
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
REPO_ID = "theodoreliu/aloha2_tactile_task"  # CHANGE the dataset name
RESUME = False
DATASET_ROOT = "/home/robot/Dataset_and_Checkpoint/lerobot-dataset"
VIZ_VIDEO_DIR = "/home/robot/videos/visualize/lerobot/act"
CKPT_DIR = "/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint"
DEPLOY_VIDEO_DIR = "/home/robot/videos/deploy_third_person/lerobot/act"

# ── Tactile mode ─────────────────────────────────────────────────────
# "simple" = 1-ch grayscale (GelSight-equivalent)
# "full"   = multi-channel (all physical states)
TACTILE_MODE = TactileSensorMode.FULL
# TACTILE_MODE = TactileSensorMode.SIMPLE

# ── Camera configuration (4 RealSense cameras by serial number) ─────
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

# ── Tactile sensor configuration (4 XENSE G1-WS sensors) ────────────
# Serial numbers need to be discovered via: lerobot-find-cameras xense
# or: python -c "from xensesdk import Sensor; print(Sensor.scanSerialNumber())"
TACTILE_CONFIG = {
    "left_fingertip": XENSEG1WSConfig(
        serial_number="OP000001",  # CHANGE to actual serial
        mode=TACTILE_MODE,
        fps=FPS,
        width=160, height=120,
        mock=True,  # Set False when real hardware is connected
    ),
    "left_knuckle": XENSEG1WSConfig(
        serial_number="OP000002",  # CHANGE to actual serial
        mode=TACTILE_MODE,
        fps=FPS,
        width=160, height=120,
        mock=True,
    ),
    "right_fingertip": XENSEG1WSConfig(
        serial_number="OP000003",  # CHANGE to actual serial
        mode=TACTILE_MODE,
        fps=FPS,
        width=160, height=120,
        mock=True,
    ),
    "right_knuckle": XENSEG1WSConfig(
        serial_number="OP000004",  # CHANGE to actual serial
        mode=TACTILE_MODE,
        fps=FPS,
        width=160, height=120,
        mock=True,
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
    tactile_sensors=TACTILE_CONFIG,
)

# ── ALOHA Teleoperator configuration (dual WidowX-250 leader arms) ───
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

# ── Processors ────────────────────────────────────────────────────────
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
            num_threads=4 * len(robot.cameras) + 4 * len(robot.tactile_sensors),
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
        image_writer_threads=4 * len(robot.cameras) + 4 * len(robot.tactile_sensors),
        root=DATASET_ROOT,
    )

# ── Keyboard listener and visualization ──────────────────────────────
_, events = init_keyboard_listener()
init_rerun(session_name="aloha2_tactile_recording")

# ── Connect hardware ─────────────────────────────────────────────────
robot.connect()
teleop.connect()

episode_idx = 0
existing_episodes = dataset.num_episodes if RESUME else 0

while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    current_episode = existing_episodes + episode_idx + 1
    log_say(f"Recording episode {current_episode} ({episode_idx + 1}/{NUM_EPISODES}) "
            f"[tactile_mode={TACTILE_MODE.value}]")

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
log_say("Dataset pushed to HuggingFace Hub")
