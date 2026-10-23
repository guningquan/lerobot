#!/usr/bin/env python
"""Measure one puppet (follower) gripper's joint range for constants.py.

    python examples/aloha_ros/calibrate_puppet_gripper.py --side left

Needs `roslaunch aloha 4arms_tactile_teleop.launch` running. The puppet gripper normally holds its
position, so this first reboots it, switches it to PWM mode and cuts its torque (only the gripper,
the arm is untouched) so it can be moved by hand. Close it fully and open it fully, then Ctrl+C:
the printed min/max go into PUPPET_<SIDE>_GRIPPER_JOINT_CLOSE / _OPEN in aloha_scripts/constants.py.
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
aloha_scripts_dir = os.path.join(script_dir, "aloha_scripts")
if aloha_scripts_dir not in sys.path:
    sys.path.insert(0, aloha_scripts_dir)

from test_gripper_range import run  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--no_release", action="store_true", help="Do not release the gripper torque before reading."
    )
    args = parser.parse_args()
    raise SystemExit(run(f"puppet_{args.side}", release=not args.no_release))
