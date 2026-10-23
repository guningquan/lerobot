### Task parameters
import os

if os.getlogin() == 'guningquan':
    DATA_DIR = '/mnt/ssd1/guningquan/Programs_server/act_dataset_checkpoint/dataset'
elif os.getlogin() == 'theodoreliu':
    DATA_DIR = '/home/theodoreliu/Dataset_and_Checkpoint/dataset'
elif os.getlogin() == 'ubuntu20' or os.getlogin() == 'ubuntu22':
    DATA_DIR = '/home/robot/Dataset_and_Checkpoint/dataset'
else:
    raise ValueError(f"Unknown user: {os.getlogin()}")


TASK_CONFIGS = {

'weighing_tie2': {  # include loss
        'dataset_dir': [
            DATA_DIR + '/weighing/place_red_cube',
            DATA_DIR + '/weighing/remove_red_cube',
            DATA_DIR + '/weighing/shutdown_balance',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/weighing/place_red_cube',
            DATA_DIR + '/weighing/remove_red_cube',
            DATA_DIR + '/weighing/shutdown_balance',
        ],
        'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
        'episode_len': 1000,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'cup_load_data_collection': {
            'dataset_dir': DATA_DIR + '/cup_load',
            'episode_len': 1000,  # 900
            'camera_names': ['cam_high',
                             'cam_low',
                             'cam_left_wrist',
                             'cam_right_wrist',
                             ]
        },
    
    'apple_grasping': {
        'dataset_dir': DATA_DIR + '/apple_grasping',
        'episode_len': 450,  # e.g., 15 seconds * 30FPS = 450
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },

}


### ALOHA fixed constants
# Default control rate. The example scripts take --fps (e.g. 30 or 50) and pass it to the robot /
# teleoperator configs, which derive their own per-tick servo trajectory time from it.
DT = 1/30  # FPS=30
FPS = 30

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
#START_ARM_POSE = [0, -0.8, 0.8, 0, 0, 0, 0.02239, -0.02239,  0, -0.8, 0.8, 0, 0, 0, 0.02239, -0.02239]

# Cameras as published by `roslaunch aloha 4arms_tactile_teleop.launch` (usb_cam nodes, rgb8,
# 640x480, ~60Hz). The launch owns the /dev/CAM_* devices, so lerobot subscribes to these topics
# (camera type "ros_image") instead of opening the devices itself.
ROS_CAMERA_TOPICS = {
    "cam_high": "/usb_cam_high/image_raw",
    "cam_low": "/usb_cam_low/image_raw",
    "cam_left_wrist": "/usb_cam_left_wrist/image_raw",
    "cam_right_wrist": "/usb_cam_right_wrist/image_raw",
}
ROS_CAMERA_WIDTH = 640
ROS_CAMERA_HEIGHT = 480

# Sleep poses (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate), taken from ALOHA2's
# sleep_plus.py. Going to sleep is two-stage: first the OLD_* pose (the previously used sleep pose,
# a path these arms have travelled many times), then the measured resting pose *_SLEEP_POSITION,
# which is where torque can be cut without the arm dropping. Cutting torque at the OLD_* pose
# dropped the arms hard (most obviously the master, wrist_angle=0.0). If an arm still drops on
# shutdown, re-read the rest pose with `rostopic echo /puppet_left/joint_states -n1` and update.
OLD_PUPPET_SLEEP_POSITION = (0, -1.7, 1.55, 0.12, 0.65, 0)
OLD_MASTER_SLEEP_POSITION = (0, -1.76, 1.55, 0, 0.0, 0)
PUPPET_SLEEP_POSITION = (0, -1.85, 1.594, 0, 0.782, 0)   # measured resting pose, not the factory one
MASTER_SLEEP_POSITION = (0, -1.88, 1.605, 0, 0.281, 0)   # measured resting pose, not the factory one
# Seconds per stage (old way-point, rest pose); the master gets a slower second stage.
PUPPET_SLEEP_MOVE_TIMES = (1, 1)
MASTER_SLEEP_MOVE_TIMES = (1, 1.5)
# Multiplies every shutdown move time (gripper + both arm way-points). 2.0 = twice as slow as --sleep.
SHUTDOWN_SPEED_SCALE = 2.0
PUPPET_SHUTDOWN_MOVE_TIMES = tuple(t * SHUTDOWN_SPEED_SCALE for t in PUPPET_SLEEP_MOVE_TIMES)
MASTER_SHUTDOWN_MOVE_TIMES = tuple(t * SHUTDOWN_SPEED_SCALE for t in MASTER_SLEEP_MOVE_TIMES)

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
# Only the PUPPET pair is still used (real_env.get_qpos / get_qvel of the HDF5 scripts).
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6]) - measured working range of the ALOHA2 grippers.
# The four grippers do not share one range, so every arm is calibrated separately.
# Re-measure with `python examples/aloha_ros/calibrate_master_gripper.py --side left` (and
# calibrate_puppet_gripper.py) after any gripper is swapped or re-mounted.
MASTER_LEFT_GRIPPER_JOINT_OPEN = 0.7762
MASTER_LEFT_GRIPPER_JOINT_CLOSE = -0.0890
MASTER_RIGHT_GRIPPER_JOINT_OPEN = 0.8038
MASTER_RIGHT_GRIPPER_JOINT_CLOSE = -0.0522
PUPPET_LEFT_GRIPPER_JOINT_OPEN = 0.0767
PUPPET_LEFT_GRIPPER_JOINT_CLOSE = -1.5  # -1.3606
PUPPET_RIGHT_GRIPPER_JOINT_OPEN = 0.0782
PUPPET_RIGHT_GRIPPER_JOINT_CLOSE = -1.5  # -1.3898

# (close, open) per side, for code that handles either arm (lerobot robots / teleoperators).
MASTER_GRIPPER_JOINT_RANGE = {
    "left": (MASTER_LEFT_GRIPPER_JOINT_CLOSE, MASTER_LEFT_GRIPPER_JOINT_OPEN),
    "right": (MASTER_RIGHT_GRIPPER_JOINT_CLOSE, MASTER_RIGHT_GRIPPER_JOINT_OPEN),
}
PUPPET_GRIPPER_JOINT_RANGE = {
    "left": (PUPPET_LEFT_GRIPPER_JOINT_CLOSE, PUPPET_LEFT_GRIPPER_JOINT_OPEN),
    "right": (PUPPET_RIGHT_GRIPPER_JOINT_CLOSE, PUPPET_RIGHT_GRIPPER_JOINT_OPEN),
}

############################ Helper functions ############################

# --- gripper position (metres) ---
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)


def _clamp01(x):
    """Clamp to [0, 1] so master over-travel cannot command the puppet past its limit."""
    return min(max(x, 0.0), 1.0)


# --- gripper joint (rad) ---
# master joint -> normalised gripper action (0: closed, 1: open)
MASTER_LEFT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_LEFT_GRIPPER_JOINT_CLOSE) / (MASTER_LEFT_GRIPPER_JOINT_OPEN - MASTER_LEFT_GRIPPER_JOINT_CLOSE)
MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_RIGHT_GRIPPER_JOINT_CLOSE) / (MASTER_RIGHT_GRIPPER_JOINT_OPEN - MASTER_RIGHT_GRIPPER_JOINT_CLOSE)
MASTER_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_LEFT_GRIPPER_JOINT_OPEN - MASTER_LEFT_GRIPPER_JOINT_CLOSE) + MASTER_LEFT_GRIPPER_JOINT_CLOSE
MASTER_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_RIGHT_GRIPPER_JOINT_OPEN - MASTER_RIGHT_GRIPPER_JOINT_CLOSE) + MASTER_RIGHT_GRIPPER_JOINT_CLOSE
# puppet joint <-> normalised gripper observation / action
PUPPET_LEFT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_LEFT_GRIPPER_JOINT_CLOSE) / (PUPPET_LEFT_GRIPPER_JOINT_OPEN - PUPPET_LEFT_GRIPPER_JOINT_CLOSE)
PUPPET_RIGHT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_RIGHT_GRIPPER_JOINT_CLOSE) / (PUPPET_RIGHT_GRIPPER_JOINT_OPEN - PUPPET_RIGHT_GRIPPER_JOINT_CLOSE)
PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_LEFT_GRIPPER_JOINT_OPEN - PUPPET_LEFT_GRIPPER_JOINT_CLOSE) + PUPPET_LEFT_GRIPPER_JOINT_CLOSE
PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_RIGHT_GRIPPER_JOINT_OPEN - PUPPET_RIGHT_GRIPPER_JOINT_CLOSE) + PUPPET_RIGHT_GRIPPER_JOINT_CLOSE
# master joint -> puppet joint, clamped
MASTER2PUPPET_LEFT_JOINT_FN = lambda x: PUPPET_LEFT_GRIPPER_JOINT_UNNORMALIZE_FN(_clamp01(MASTER_LEFT_GRIPPER_JOINT_NORMALIZE_FN(x)))
MASTER2PUPPET_RIGHT_JOINT_FN = lambda x: PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(_clamp01(MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN(x)))

# --- master "half open" pose ---
MASTER_LEFT_GRIPPER_JOINT_MID = (MASTER_LEFT_GRIPPER_JOINT_OPEN + MASTER_LEFT_GRIPPER_JOINT_CLOSE) / 2
MASTER_RIGHT_GRIPPER_JOINT_MID = (MASTER_RIGHT_GRIPPER_JOINT_OPEN + MASTER_RIGHT_GRIPPER_JOINT_CLOSE) / 2

# --- backward-compatible single-sided aliases (they resolve to the RIGHT arm, as in ALOHA2) ---
# Only the old HDF5 scripts in aloha_scripts/ (real_env.py, one_side_teleop.py,
# record_episodes_compress.py) still import these; new code uses the per-side names above.
MASTER_GRIPPER_JOINT_OPEN = MASTER_RIGHT_GRIPPER_JOINT_OPEN
MASTER_GRIPPER_JOINT_CLOSE = MASTER_RIGHT_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_OPEN = PUPPET_RIGHT_GRIPPER_JOINT_OPEN
PUPPET_GRIPPER_JOINT_CLOSE = PUPPET_RIGHT_GRIPPER_JOINT_CLOSE
MASTER_GRIPPER_JOINT_NORMALIZE_FN = MASTER_RIGHT_GRIPPER_JOINT_NORMALIZE_FN
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = PUPPET_RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN
MASTER2PUPPET_JOINT_FN = MASTER2PUPPET_RIGHT_JOINT_FN
MASTER_GRIPPER_JOINT_MID = MASTER_RIGHT_GRIPPER_JOINT_MID



SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
    'all': {
        'dataset_dir': DATA_DIR + '/',
        'num_episodes': None,
        'episode_len': None,
        'name_filter': lambda n: 'sim' not in n,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    'sim_transfer_cube_scripted_mirror':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_mirror': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

}