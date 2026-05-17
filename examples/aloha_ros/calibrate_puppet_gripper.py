#!/usr/bin/env python
import time
import sys

# Import both possible classes
from lerobot.robots.viperx_ros import ViperXRos, ViperXRosConfig
from lerobot.teleoperators.widowx_ros import WidowXRos, WidowXRosConfig

# ====== Configuration ======
ARM_TYPE = "puppet"  # "master" or "puppet"
ARM_SIDE = "right"   # "left" or "right"
# ===========================


def calibrate():
    name = f"{ARM_TYPE}_{ARM_SIDE}"
    print(f"🔄 Connecting to {name.upper()} for calibration...")

    if ARM_TYPE == "master":
        config = WidowXRosConfig(robot_name=name)
        arm = WidowXRos(config)
    else:
        config = ViperXRosConfig(robot_name=name)
        arm = ViperXRos(config)

    try:
        arm.connect()

        # Try to disable torque (for puppet arm)
        if ARM_TYPE == "puppet":
            print("🔓 Trying to release torque... If it still cannot be moved, check driver settings.")
            # The `bot` attribute is based on your previous `dir(teleop)` output
            if hasattr(arm, "bot"):
                try:
                    # Try disabling torque for all motors through low-level core API
                    arm.bot.core.robot_set_motor_registers("group", "all", "Torque_Enable", 0)
                except:
                    print("⚠️ Could not disable torque in code; please cut motor power manually or disable it in Wizard.")

        print("\n" + "=" * 55)
        print(f"🔧 {name.upper()} gripper calibration mode 🔧")
        print("=" * 55)
        print("Operation: 1. Move the gripper manually  2. Record values  3. Ctrl+C to exit")
        print("-" * 55)

        while True:
            # Automatically choose an available data getter
            data = {}
            if hasattr(arm, "get_observation"):
                data = arm.get_observation()
            elif hasattr(arm, "get_action"):
                data = arm.get_action()

            gripper_pos = None
            for key in data.keys():
                if "gripper" in key:
                    val = data[key]
                    gripper_pos = val.item() if hasattr(val, "item") else val
                    break

            if gripper_pos is not None:
                print(f"\rCurrent {name} gripper position: {gripper_pos:8.4f}", end="")
            else:
                print(f"\rNo gripper data found. Available keys: {list(data.keys())}", end="")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n✅ Stopped reading values.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        print("Disconnecting...")
        sys.exit(0)


if __name__ == "__main__":
    calibrate()