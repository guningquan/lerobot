import time
import sys

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.viperx import ViperX, ViperXConfig
from lerobot.teleoperators.widowx import WidowX, WidowXConfig
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.robot_utils import precise_sleep
# Try to import pynput for keyboard listening
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Press Ctrl+C to exit instead of 'q'.")

# camera_config = {
#     "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
#     "wrist_right": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
#     "wrist_left": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
# }
FPS = 30
camera_config = {
    "cam_high": OpenCVCameraConfig(index_or_path="/dev/CAM_HIGH", width=640, height=480, fps=FPS),
    "cam_right_wrist": OpenCVCameraConfig(index_or_path="/dev/CAM_RIGHT_WRIST", width=640, height=480, fps=FPS),
    "cam_left_wrist": OpenCVCameraConfig(index_or_path="/dev/CAM_LEFT_WRIST", width=640, height=480, fps=FPS),
    "cam_low": OpenCVCameraConfig(index_or_path="/dev/CAM_LOW", width=640, height=480, fps=FPS),
}

config_follower_right = ViperXConfig(
    port="/dev/ttyDXL_puppet_right",
    id="viperx_right",
    max_relative_target=10.0,  # increased from default 5.0 to 10.0
    use_degrees=True,
    cameras=camera_config,
)

config_leader_right = WidowXConfig(
    port="/dev/ttyDXL_master_right",
    id="widowx_right",
    # gripper_motor="xc430-w150",
    use_degrees=True,
)

config_follower_left = ViperXConfig(
    port="/dev/ttyDXL_puppet_left",
    id="viperx_left",
    max_relative_target=10.0,  # increased from default 5.0 to 10.0
    use_degrees=True,
)

config_leader_left = WidowXConfig(
    port="/dev/ttyDXL_master_left",
    id="widowx_left",
    # gripper_motor="xl430-w250",
    use_degrees=True,
)

init_rerun(session_name="teleop")

follower_right = ViperX(config_follower_right)
follower_right.connect()

leader_right = WidowX(config_leader_right)
leader_right.connect()

follower_left = ViperX(config_follower_left)
follower_left.connect()

leader_left = WidowX(config_leader_left)
leader_left.connect()

# Setup keyboard listener for 'q' key to exit
exit_flag = False

def on_press(key):
    global exit_flag
    try:
        if hasattr(key, 'char') and key.char == 'q':
            print("\n'q' key pressed. Exiting...")
            exit_flag = True
            return False  # Stop listener
    except AttributeError:
        pass

listener = None
if PYNPUT_AVAILABLE:
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("Press 'q' to quit")
else:
    print("Press Ctrl+C to quit")

try:
    while not exit_flag:
        t0 = time.perf_counter()
        act_right = leader_right.get_action()
        obs_right = follower_right.get_observation()

        act_left = leader_left.get_action()
        obs_left = follower_left.get_observation()

        # print("=" * 60)
        # print("ACTION (Leader Right):")
        # for key, value in act_right.items():
        #     if key.endswith(".pos"):
        #         print(f"  {key:20}: {value:8.3f}")

        # print("\nOBSERVATION (Follower Right):")
        # for key, value in obs_right.items():
        #     if key.endswith(".pos"):
        #         print(f"  {key:20}: {value:8.3f}")

        # print("=" * 60)
        # print("ACTION (Leader Left):")
        # for key, value in act_left.items():
        #     if key.endswith(".pos"):
        #         print(f"  {key:20}: {value:8.3f}")

        # print("\nOBSERVATION (Follower Left):")
        # for key, value in obs_left.items():
        #     if key.endswith(".pos"):
        #         print(f"  {key:20}: {value:8.3f}")
        # print("=" * 60)

        log_rerun_data({**obs_right, **obs_left}, {**act_right, **act_left})

        follower_right.send_action(act_right)
        follower_left.send_action(act_left)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
        
except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    # Disconnect all devices
    print("Disconnecting devices...")
    leader_left.disconnect()
    follower_left.disconnect()
    leader_right.disconnect()
    follower_right.disconnect()
    
    if PYNPUT_AVAILABLE and listener is not None and listener.is_alive():
        listener.stop()
    print("All devices disconnected. Exiting.")
