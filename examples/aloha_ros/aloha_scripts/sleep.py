from interbotix_xs_modules.arm import InterbotixManipulatorXS
try:
    from aloha_scripts.robot_utils import move_arms, torque_on, move_grippers, torque_off
except ImportError:
    from robot_utils import move_arms, torque_on, move_grippers,torque_off
import rospy
import threading,time
import argparse

def move_robot(robot, gripper_position, arm_position, move_time):
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_time)
    move_arms([robot], [arm_position], move_time=move_time)
    # torque_off(robot)

def move_shut_robot(robot, gripper_position, arm_position, move_time):
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_time)
    move_arms([robot], [arm_position], move_time=move_time)
    time.sleep(1)
    torque_off(robot)

def shut_down_all_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    puppet_sleep_position = (0, -1.85, 1.6, 0.12, 0.65, 0)
    master_sleep_position = (0, -1.85, 1.6, 0, 0.0, 0)
    PUPPET_GRIPPER_JOINT_OPEN = 0.0414
    MASTER_GRIPPER_JOINT_OPEN = 0.7400

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_left, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_right, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_left, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_right, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

def shut_down_puppet_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)

    puppet_sleep_position = (0, -1.85, 1.6, 0.12, 0.65, 0)
    PUPPET_GRIPPER_JOINT_OPEN = 0.0414

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_left, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_right, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def sleep_all_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    puppet_sleep_position = (0, -1.85, 1.6, 0.12, 0.65, 0)
    master_sleep_position = (0, -1.85, 1.6, 0, 0.0, 0)
    PUPPET_GRIPPER_JOINT_OPEN = 0.0414
    MASTER_GRIPPER_JOINT_OPEN = 0.7409

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_left, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_right, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_left, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_right, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def sleep_puppet_robots():
    # rospy.init_node('sleep')

    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)

    puppet_sleep_position = (0, -1.85, 1.6, 0.12, 0.65, 0)
    PUPPET_GRIPPER_JOINT_OPEN = 0.0414

    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_left, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_right, PUPPET_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

def shut_down_master_robots():
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)
    master_sleep_position = (0, -1.85, 1.6, 0, 0.0, 0)
    MASTER_GRIPPER_JOINT_OPEN = 0.7400
    threads = []
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_left, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_right, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def sleep_master_robots():
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)
    master_sleep_position = (0, -1.85, 1.6, 0, 0.0, 0)
    MASTER_GRIPPER_JOINT_OPEN = 0.7409
    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_left, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_right, MASTER_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()





def main():
    parser = argparse.ArgumentParser(description="Control script for robots")
    parser.add_argument("--sleep", action="store_true",
                        help="Put robots into sleep mode")
    parser.add_argument("--shut_down", action="store_true",
                        help="Shut down robots (sleep + torque off)")
    parser.add_argument("--arms", choices=("all", "puppet", "master"), default="all",
                        help="Which arms to control (default: all)")
    args = parser.parse_args()

    if args.shut_down:
        if args.arms == "puppet":
            shut_down_puppet_robots()
        elif args.arms == "master":
            shut_down_master_robots()
        else:
            shut_down_all_robots()
    elif args.sleep:
        if args.arms == "puppet":
            sleep_puppet_robots()
        elif args.arms == "master":
            sleep_master_robots()
        else:
            sleep_all_robots()
    else:
        parser.print_help()

if __name__ == "__main__":
    rospy.init_node('sleep')
    main()
