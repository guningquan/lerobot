import threading

import numpy as np
import time
from constants import DT
from interbotix_xs_msgs.msg import JointSingleCommand

# import IPython
# e = IPython.embed

class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.is_debug = is_debug
        self.bridge = CvBridge()

        # Add "gel" to our list of cameras
        self.camera_names = [
            'cam_high',
            'cam_low',
            'cam_left_wrist',
            'cam_right_wrist',
            # 'gel',   # @gnq important !
        ]

        if init_node:
            rospy.init_node('image_recorder', anonymous=True)

        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)

            # Select callback based on camera name
            if cam_name == 'cam_high':
                callback_func = self.image_cb_cam_high
                topic_name = "/usb_cam_high/image_raw"

            elif cam_name == 'cam_low':
                callback_func = self.image_cb_cam_low
                topic_name = "/usb_cam_low/image_raw"

            elif cam_name == 'cam_left_wrist':
                callback_func = self.image_cb_cam_left_wrist
                topic_name = "/usb_cam_left_wrist/image_raw"

            elif cam_name == 'cam_right_wrist':
                callback_func = self.image_cb_cam_right_wrist
                topic_name = "/usb_cam_right_wrist/image_raw"

            elif cam_name == 'gel':
                callback_func = self.image_cb_gel
                # Make sure this matches the topic used in digit.py
                topic_name = "/gel/camera/image_color"

            else:
                # If you add more names later, handle them here
                raise NotImplementedError

            # Subscribe to the correct topic with the selected callback
            rospy.Subscriber(topic_name, Image, callback_func)

            # If debug, store the last 50 timestamps for frequency diagnostics
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))

        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        setattr(self, f'{cam_name}_image', cv_image)
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)

        if self.is_debug:
            ts_list = getattr(self, f'{cam_name}_timestamps')
            ts_list.append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)

    def image_cb_cam_high(self, data):
        return self.image_cb('cam_high', data)

    def image_cb_cam_low(self, data):
        return self.image_cb('cam_low', data)

    def image_cb_cam_left_wrist(self, data):
        return self.image_cb('cam_left_wrist', data)

    def image_cb_cam_right_wrist(self, data):
        return self.image_cb('cam_right_wrist', data)

    def image_cb_gel(self, data):
        return self.image_cb('gel', data)

    def get_images(self):
        image_dict = {}
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(list_of_timestamps):
            arr = np.array(list_of_timestamps)
            diff = arr[1:] - arr[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            ts_list = getattr(self, f'{cam_name}_timestamps', [])
            if len(ts_list) > 1:
                image_freq = 1 / dt_helper(ts_list)
                print(f'{cam_name} frequency: {image_freq:.2f} Hz')
            else:
                print(f'{cam_name} no timestamps recorded yet.')
        print()

class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node('recorder', anonymous=True)
        rospy.Subscriber(f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_group", JointGroupCommand, self.puppet_arm_commands_cb)
        rospy.Subscriber(f"/puppet_{side}/commands/joint_single", JointSingleCommand, self.puppet_gripper_commands_cb)
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def puppet_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def puppet_arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def puppet_gripper_commands_cb(self, data):
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n')

def get_arm_joint_positions(bot):
    return bot.arm.core.joint_states.position[:6]

def get_arm_gripper_positions(bot):
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position

def move_arms(bot_list, target_pose_list, move_time=1):
    """Move every bot in bot_list to its target in target_pose_list, in lockstep, over move_time.

    One trajectory command per bot, with moving_time=move_time - the servo's own time-based
    Profile_Velocity/Profile_Acceleration already interpolates smoothly from wherever the arm
    currently is, so there is no need to also step through a python-side linspace.

    (From ALOHA2's robot_utils.py.) The previous implementation did both at once and they fought
    each other: it published a new micro-step target every DT but never told the servo to expect
    anything that fast, so each arm's trajectory-time registers stayed at whatever they were last
    set to (2.0s from every InterbotixManipulatorXS(...) construction). Every tick the servo
    restarted a fresh "reach this in 2 seconds" plan toward a target only 1/num_steps of the way
    along, so the arm was still well short of the destination when move_time elapsed, and a
    caller that cut torque right after could catch it mid-trajectory and let it drop.

    set_trajectory_time() is called on its own first because set_joint_positions() runs
    check_joint_limits() BEFORE it applies the new moving_time, so its velocity check would
    otherwise divide by whatever moving_time was left over from the arm's previous use (e.g. the
    per-tick teleop value 1/fps), reject an ordinary move as tens of rad/s, and silently return
    False.
    """
    accel_time = max(move_time / 4.0, 0.05)
    for bot, target in zip(bot_list, target_pose_list):
        bot.arm.set_trajectory_time(moving_time=move_time, accel_time=accel_time)
        if not bot.arm.set_joint_positions(list(target), moving_time=move_time, accel_time=accel_time, blocking=False):
            print("Warning: {} arm target rejected (would exceed joint limits); holding position."
                  .format(bot.dxl.robot_name))
    time.sleep(move_time)

def move_grippers(bot_list, target_pose_list, move_time, dt=DT):
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = max(int(move_time / dt), 1)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(dt)

def set_arm_position_time_profile(bot, moving_time, accel_time=None):
    """Put the arm group in position mode with a time-based profile (from ALOHA2's one_side_teleop.py).

    robot_set_operating_modes() defaults profile_type to "velocity" with profile_velocity=0 (no
    speed limit) when not told otherwise; in that mode the servo reads every later moving_time as
    a raw velocity limit and snaps to its target near max speed. "time" makes moving_time mean
    what every caller here assumes: the duration of the move.

    This rewrites the EEPROM Operating_Mode, which briefly drops the arm's torque - only call it
    while the arm is resting. To change the cadence later, use bot.arm.set_trajectory_time().
    """
    if accel_time is None:
        accel_time = min(DT, moving_time / 2)
    bot.dxl.robot_set_operating_modes(
        "group",
        "arm",
        "position",
        "time",
        int(moving_time * 1000),
        int(accel_time * 1000),
    )
    # Force set_trajectory_time() to rewrite the registers: its python-side cache no longer
    # matches what the servo holds after the mode switch.
    bot.arm.moving_time = None
    bot.arm.accel_time = None
    bot.arm.set_trajectory_time(moving_time=moving_time, accel_time=accel_time)

def setup_puppet_bot(bot, moving_time=DT, accel_time=None):
    if accel_time is None:
        accel_time = min(DT, moving_time / 2)
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    set_arm_position_time_profile(bot, moving_time, accel_time)
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)

def setup_master_bot(bot):
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)

def set_standard_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 800)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def set_low_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 100)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)
    bot.dxl.robot_torque_enable("single", "gripper", False)

def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)
    bot.dxl.robot_torque_enable("single", "gripper", True)

def run_parallel(*callables):
    """Run each zero-argument callable in its own thread and wait for all of them.

    Used to move several arms at once (e.g. master and puppet to their start pose), the same way
    sleep_modify.py moves all four arms in parallel. Re-raises the first exception, if any.
    """
    errors = []

    def _run(fn):
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread below
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(fn,)) for fn in callables]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
