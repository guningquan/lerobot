from interbotix_xs_modules.arm import InterbotixManipulatorXS
try:
    from aloha_scripts.robot_utils import move_arms, torque_on, move_grippers, torque_off
    from aloha_scripts.constants import MASTER_LEFT_GRIPPER_JOINT_OPEN, MASTER_RIGHT_GRIPPER_JOINT_OPEN
    from aloha_scripts.constants import PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN
    from aloha_scripts.constants import OLD_PUPPET_SLEEP_POSITION, OLD_MASTER_SLEEP_POSITION
    from aloha_scripts.constants import PUPPET_SLEEP_POSITION, MASTER_SLEEP_POSITION
    from aloha_scripts.constants import PUPPET_SLEEP_MOVE_TIMES, MASTER_SLEEP_MOVE_TIMES
    from aloha_scripts.constants import PUPPET_SHUTDOWN_MOVE_TIMES, MASTER_SHUTDOWN_MOVE_TIMES
except ImportError:
    from robot_utils import move_arms, torque_on, move_grippers,torque_off
    from constants import MASTER_LEFT_GRIPPER_JOINT_OPEN, MASTER_RIGHT_GRIPPER_JOINT_OPEN
    from constants import PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN
    from constants import OLD_PUPPET_SLEEP_POSITION, OLD_MASTER_SLEEP_POSITION
    from constants import PUPPET_SLEEP_POSITION, MASTER_SLEEP_POSITION
    from constants import PUPPET_SLEEP_MOVE_TIMES, MASTER_SLEEP_MOVE_TIMES
    from constants import PUPPET_SHUTDOWN_MOVE_TIMES, MASTER_SHUTDOWN_MOVE_TIMES
import rospy
import threading,time
import argparse

# Ported from ALOHA2's aloha_scripts/sleep_plus.py; same functions and the same CLI as before
# (--sleep / --shut_down / --shut_down_puppet).
#
# The sleep poses live in constants.py (so the lerobot robot classes can use them without
# importing this ROS script): OLD_*_SLEEP_POSITION is the first way-point and *_SLEEP_POSITION the
# measured resting pose. Going straight to a single "sleep" target, or cutting torque before the
# arm actually arrived, both let an arm drop hard under its own weight - hence two stages, and
# move_arms() (robot_utils.py) setting the servo's own moving_time so that sleeping move_time
# afterwards is accurate.

# (old way-point, measured rest pose)
PUPPET_SLEEP_WAYPOINTS = (OLD_PUPPET_SLEEP_POSITION, PUPPET_SLEEP_POSITION)
MASTER_SLEEP_WAYPOINTS = (OLD_MASTER_SLEEP_POSITION, MASTER_SLEEP_POSITION)

def move_robot(robot, gripper_position, arm_waypoints, move_times):
    """Like move_shut_robot, but leaves torque on afterwards (used for '--sleep': the arm should
    stay actively held at the rest pose, not powered down). Two stages for the same reason as
    move_shut_robot: the old, well-traveled pose first, then the small final correction onto the
    measured rest pose, rather than one big jump straight from wherever the arm was working.
    """
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_times[0])
    for arm_position, move_time in zip(arm_waypoints, move_times):
        move_arms([robot], [arm_position], move_time=move_time)

def move_shut_robot(robot, gripper_position, arm_waypoints, move_times):
    """Visit arm_waypoints (a sequence of 6-joint poses) in order, then cut torque."""
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_times[0])
    for arm_position, move_time in zip(arm_waypoints, move_times):
        move_arms([robot], [arm_position], move_time=move_time)
    # move_arms() sets the servo's own moving_time to match and sleeps that long itself, so by now
    # the arm really is at the last waypoint - this is just a small settle margin.
    time.sleep(0.3)
    torque_off(robot)

def _make_puppets():
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    return puppet_bot_left, puppet_bot_right

def _make_masters():
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)
    return master_bot_left, master_bot_right

def _run_threads(jobs):
    # 创建线程
    threads = [threading.Thread(target=target, args=args) for target, args in jobs]
    # 启动所有线程
    for thread in threads:
        thread.start()
    # 等待所有线程完成
    for thread in threads:
        thread.join()

def shut_down_all_robots():
    puppet_bot_left, puppet_bot_right = _make_puppets()
    master_bot_left, master_bot_right = _make_masters()
    _run_threads([
        (move_shut_robot, (puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_SLEEP_WAYPOINTS, PUPPET_SHUTDOWN_MOVE_TIMES)),
        (move_shut_robot, (puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, PUPPET_SLEEP_WAYPOINTS, PUPPET_SHUTDOWN_MOVE_TIMES)),
        (move_shut_robot, (master_bot_left, MASTER_LEFT_GRIPPER_JOINT_OPEN, MASTER_SLEEP_WAYPOINTS, MASTER_SHUTDOWN_MOVE_TIMES)),
        (move_shut_robot, (master_bot_right, MASTER_RIGHT_GRIPPER_JOINT_OPEN, MASTER_SLEEP_WAYPOINTS, MASTER_SHUTDOWN_MOVE_TIMES)),
    ])

def shut_down_puppet_robots():
    puppet_bot_left, puppet_bot_right = _make_puppets()
    _run_threads([
        (move_shut_robot, (puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_SLEEP_WAYPOINTS, PUPPET_SHUTDOWN_MOVE_TIMES)),
        (move_shut_robot, (puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, PUPPET_SLEEP_WAYPOINTS, PUPPET_SHUTDOWN_MOVE_TIMES)),
    ])

def sleep_all_robots():
    puppet_bot_left, puppet_bot_right = _make_puppets()
    master_bot_left, master_bot_right = _make_masters()
    _run_threads([
        (move_robot, (puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_SLEEP_WAYPOINTS, PUPPET_SLEEP_MOVE_TIMES)),
        (move_robot, (puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, PUPPET_SLEEP_WAYPOINTS, PUPPET_SLEEP_MOVE_TIMES)),
        (move_robot, (master_bot_left, MASTER_LEFT_GRIPPER_JOINT_OPEN, MASTER_SLEEP_WAYPOINTS, MASTER_SLEEP_MOVE_TIMES)),
        (move_robot, (master_bot_right, MASTER_RIGHT_GRIPPER_JOINT_OPEN, MASTER_SLEEP_WAYPOINTS, MASTER_SLEEP_MOVE_TIMES)),
    ])

def sleep_puppet_robots():
    puppet_bot_left, puppet_bot_right = _make_puppets()
    _run_threads([
        (move_robot, (puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_SLEEP_WAYPOINTS, PUPPET_SLEEP_MOVE_TIMES)),
        (move_robot, (puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, PUPPET_SLEEP_WAYPOINTS, PUPPET_SLEEP_MOVE_TIMES)),
    ])

def sleep_master_robots():
    master_bot_left, master_bot_right = _make_masters()
    _run_threads([
        (move_robot, (master_bot_left, MASTER_LEFT_GRIPPER_JOINT_OPEN, MASTER_SLEEP_WAYPOINTS, MASTER_SLEEP_MOVE_TIMES)),
        (move_robot, (master_bot_right, MASTER_RIGHT_GRIPPER_JOINT_OPEN, MASTER_SLEEP_WAYPOINTS, MASTER_SLEEP_MOVE_TIMES)),
    ])


def main():
    parser = argparse.ArgumentParser(description="Control script for robots")
    parser.add_argument("--sleep", action="store_true",
                        help="Put all robots into sleep mode")
    parser.add_argument("--shut_down", action="store_true",
                        help="Shut down all robots")
    parser.add_argument("--shut_down_puppet", action="store_true",
                        help="Shut down puppet robots")

    args = parser.parse_args()

    if args.sleep:
        sleep_all_robots()
    elif args.shut_down:
        shut_down_all_robots()
    elif args.shut_down_puppet:
        shut_down_puppet_robots()
    else:
        print("Please provide a valid argument (--sleep or --shut_down).")

if __name__ == "__main__":
    rospy.init_node('sleep_modify')
    main()
