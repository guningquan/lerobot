#!/usr/bin/env python
import time
import sys

# Import your lerobot library
from lerobot.teleoperators.widowx_ros import WidowXRos, WidowXRosConfig

# ====== Configuration ======
ARM_SIDE = "right"
# ===========================


def test_gripper_scope():
    print(f"🔄 Connecting to {ARM_SIDE.upper()} master arm for gripper calibration...")

    config = WidowXRosConfig(robot_name=f"master_{ARM_SIDE}")
    teleop = WidowXRos(config)

    try:
        teleop.connect()

        print("\n" + "=" * 55)
        print("🔧 Gripper travel calibration mode (Float issue fixed) 🔧")
        print("=" * 55)
        print("Steps:")
        print("1. Move the gripper manually and watch the values change.")
        print("2. Record values for [fully open] and [fully closed].")
        print("3. Press Ctrl+C to exit.")
        print("-" * 55)

        while True:
            # Get the current joint state/action values
            action = teleop.get_action()

            gripper_pos = None

            # Compatibility check: find keys containing "gripper"
            # In your setup this could be "gripper.pos" or "gripper"
            for key in action.keys():
                if "gripper" in key:
                    val = action[key]
                    # Convert torch tensor to float; use raw value if already float
                    gripper_pos = val.item() if hasattr(val, "item") else val
                    break

            if gripper_pos is not None:
                # Realtime position output: width 8, precision 4
                print(f"\rCurrent {ARM_SIDE} gripper position: {gripper_pos:8.4f}", end="")
            else:
                print(f"\rNo gripper key found in action. Available keys: {list(action.keys())}", end="")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n✅ Stopped reading values.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        print("Disconnecting...")
        # Exit directly to avoid WidowXRos sleep protection causing out-of-range errors
        sys.exit(0)


if __name__ == "__main__":
    test_gripper_scope()