#!/usr/bin/env python
"""Measure one master (leader) gripper's joint range for constants.py.

    python examples/aloha_ros/calibrate_master_gripper.py --side left

Needs `roslaunch aloha 4arms_tactile_teleop.launch` running. The master gripper can be moved by hand
as is. Squeeze it fully closed and open it fully, then Ctrl+C: the printed min/max go into
MASTER_<SIDE>_GRIPPER_JOINT_CLOSE / _OPEN in aloha_scripts/constants.py.
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
    args = parser.parse_args()
    raise SystemExit(run(f"master_{args.side}", release=False))
