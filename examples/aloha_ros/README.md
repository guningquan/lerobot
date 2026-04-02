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
   - `aloha_ros_teleop.py` - Complete teleoperation example

## Usage

### Basic Teleoperation

```python
from lerobot.robots.aloha_ros import AlohaRos, AlohaRosConfig
from lerobot.teleoperators.aloha_teleop_ros import AlohaTeleopRos, AlohaTeleopRosConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Configure cameras
camera_config = {
    "cam_high": OpenCVCameraConfig(index_or_path="/dev/CAM_HIGH", width=640, height=480, fps=30),
    "cam_right_wrist": OpenCVCameraConfig(index_or_path="/dev/CAM_RIGHT_WRIST", width=640, height=480, fps=30),
    "cam_left_wrist": OpenCVCameraConfig(index_or_path="/dev/CAM_LEFT_WRIST", width=640, height=480, fps=30),
    "cam_low": OpenCVCameraConfig(index_or_path="/dev/CAM_LOW", width=640, height=480, fps=30),
}

# Create master arms (teleoperators)
master = AlohaTeleopRos(AlohaTeleopRosConfig(
    id="aloha_master",
    left_arm_robot_name="master_left",
    right_arm_robot_name="master_right",
    robot_model="wx250s",
    init_ros_node=True,
))
master.connect()

# Create puppet arms (robots)
puppet = AlohaRos(AlohaRosConfig(
    id="aloha_puppet",
    puppet_left_robot_name="puppet_left",
    puppet_right_robot_name="puppet_right",
    robot_model="vx300s",
    init_ros_node=False,
    cameras=camera_config,
))
puppet.connect()

# Teleoperation loop
while True:
    # Get actions from master arms
    action = master.get_action()
    
    # Get observations from puppet arms
    obs = puppet.get_observation()
    
    # Send actions to puppet arms
    puppet.send_action(action)
    
    time.sleep(0.02)
```

## Requirements

1. ROS environment must be set up
2. Install interbotix dependencies:
   ```bash
   pip install interbotix-xseries-modules
   ```
3. ROS launch files for robots must be running:
   - `master_left` and `master_right` (WidowX arms)
   - `puppet_left` and `puppet_right` (ViperX arms)

## Differences from Direct Motor Control

- **Control Method**: Uses ROS `InterbotixManipulatorXS` instead of direct Dynamixel motor control
- **Camera System**: Uses lerobot camera system (same as ViperX) instead of ROS image topics
- **Architecture**: Separated into base components (single arm) and combined components (dual arm)

