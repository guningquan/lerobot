# ALOHA ROS Implementation

This directory contains the ROS-based implementation of ALOHA (A Low-cost Open-source Hardware System for Bimanual Teleoperation) using lerobot framework.

## Architecture

The implementation follows a layered architecture:

### Layer 1: Base Components (Single Arm)

1. **`teleoperators/widowx_ros/`** - Single WidowX master/leader arm via ROS
   - Controls one master arm (e.g., `master_left` or `master_right`)
   - Uses ROS `InterbotixManipulatorXS` for control
   - PWM mode, torque off (for human manipulation)

2. **`robots/viperx_ros/`** - Single ViperX puppet/follower arm via ROS
   - Controls one puppet arm (e.g., `puppet_left` or `puppet_right`)
   - Uses ROS `InterbotixManipulatorXS` for control
   - Position mode, torque on
   - Supports lerobot camera system

### Layer 2: Combined Components (Dual Arm)

3. **`teleoperators/aloha_teleop_ros/`** - Dual WidowX master arms
   - Combines two `WidowXRos` instances
   - Left and right master arms
   - Used to get actions from human operator

4. **`robots/aloha_ros/`** - Dual ViperX puppet arms
   - Combines two `ViperXRos` instances
   - Left and right puppet arms
   - Uses lerobot camera system (shared cameras)
   - Used to send actions and get observations

### Layer 3: Examples

5. **`examples/aloha_ros/`** - Usage examples
   - `aloha_ros_teleop.py` / `aloha_ros_teleop_one_arm.py` - Teleoperation (no recording)
   - `aloha_ros_record.py` - Record a LeRobot dataset
   - `aloha_ros_eval.py` - Run a trained policy
   - `calibrate_master_gripper.py` / `calibrate_puppet_gripper.py` - Measure gripper ranges
   - `aloha_scripts/sleep_modify.py` - Rest pose / power off (`--sleep`, `--shut_down`, `--shut_down_puppet`)

## ALOHA2 control

Arm control follows ALOHA2's `act-triple-plus-aloha2/aloha_scripts`:

- **Servo profile**: the arms run in position mode with a *time*-based profile; every per-tick command
  asks the servos to arrive in `1 / fps` seconds. `fps` (e.g. 30 or 50) is set on `AlohaRosConfig` /
  `AlohaTeleopRosConfig` and must match the loop rate (`--fps` of the example scripts,
  `--robot.fps` + `--dataset.fps` for `lerobot-record`).
- **Gripper**: `*_gripper.pos` in observations and actions is the gripper opening normalised to [0, 1]
  (0 = closed, 1 = open) with each arm's own calibration (`*_LEFT/RIGHT_GRIPPER_JOINT_OPEN/CLOSE` in
  `aloha_scripts/constants.py`). The puppet maps it to its joint angle, so recording and deployment share
  one mapping. Datasets recorded before this change stored raw joint angles and are not compatible.
- **Safety**: commands are clamped to the joint limits, the puppet gripper stops squeezing once it stalls
  on an object, the first commands after (re)starting are blended in from the current pose, and the arms
  go to the measured rest pose in two stages before torque is cut (`disconnect()`, `shutdown()`).
- **Cameras**: the launch owns `/dev/CAM_*` (usb_cam nodes), so the cameras are read from its topics with
  the `ros_image` camera type (rgb8, frames are already RGB).

## Usage

```bash
# Terminal 1: arms + cameras (keep it running)
roslaunch aloha 4arms_tactile_teleop.launch

# Terminal 2
source ~/interbotix_ws_noetic/devel/setup.bash
source .venv/bin/activate
python examples/aloha_ros/aloha_ros_teleop.py --fps 30     # → start/resume, ← pause & lock, ESC stop
python examples/aloha_ros/aloha_ros_record.py --fps 30
python examples/aloha_ros/aloha_ros_eval.py --fps 30
```

### Basic Teleoperation

```python
from lerobot.cameras.ros_image import RosImageCameraConfig
from lerobot.robots.aloha_ros import AlohaRos, AlohaRosConfig
from lerobot.teleoperators.aloha_teleop_ros import AlohaTeleopRos, AlohaTeleopRosConfig

FPS = 30

# Cameras, read from the topics published by the launch
camera_config = {
    name: RosImageCameraConfig(topic=f"/usb_{name}/image_raw", width=640, height=480, fps=FPS)
    for name in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
}

# Master arms (teleoperators): connect() sets them up, moves them to the start pose and releases them
master = AlohaTeleopRos(AlohaTeleopRosConfig(id="aloha_master", init_ros_node=True, fps=FPS))
# Puppet arms (robots): connect() sets them up and moves them to the start pose
puppet = AlohaRos(AlohaRosConfig(id="aloha_puppet", init_ros_node=False, fps=FPS, cameras=camera_config))
master.connect()
puppet.connect()

try:
    while True:
        puppet.send_action(master.get_action())
        obs = puppet.get_observation()
finally:
    puppet.disconnect()  # rest pose (two stages), torque off
    master.disconnect()
```

## Requirements

1. ROS environment must be set up (`source ~/interbotix_ws_noetic/devel/setup.bash`, do not unset `PYTHONPATH`)
2. Install interbotix dependencies:
   ```bash
   pip install interbotix-xseries-modules
   ```
3. `roslaunch aloha 4arms_tactile_teleop.launch` must be running:
   - `master_left` and `master_right` (WidowX arms)
   - `puppet_left` and `puppet_right` (ViperX arms)
   - `/usb_cam_{high,low,left_wrist,right_wrist}/image_raw` (cameras)

## Differences from Direct Motor Control

- **Control Method**: Uses ROS `InterbotixManipulatorXS` instead of direct Dynamixel motor control
- **Camera System**: Uses the lerobot camera system with the `ros_image` type, reading the launch's camera topics
- **Architecture**: Separated into base components (single arm) and combined components (dual arm)

