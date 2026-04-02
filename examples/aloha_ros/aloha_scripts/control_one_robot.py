import sys
import time
import numpy as np
from interbotix_xs_modules.arm import InterbotixManipulatorXS

START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0 ]
DT = 1/30  # FPS=30


def control_master_left_to_joint_position(target_joint_positions=None):
    """
    Control master_left robot arm to specified joint positions.
    
    Args:
        target_joint_positions: List of 6 joint positions [waist, shoulder, elbow, 
                              forearm_roll, wrist_angle, wrist_rotate].
                              If None, uses default START_ARM_POSE positions.
    """
    # Initialize master_left robot
    print("Initializing master_left robot...")
    master_bot = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name='master_right',
        init_node=True
    )
    
    try:
        # Set operating mode to position
        print("Setting operating mode to position...")
        master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
        
        # Enable torque
        print("Enabling torque...")
        master_bot.dxl.robot_torque_enable("group", "arm", True)
        
        # Use default position if target not provided
        if target_joint_positions is None:
            target_joint_positions = START_ARM_POSE[:6]
            print(f"Using default START_ARM_POSE positions: {target_joint_positions}")
        else:
            print(f"Moving to target joint positions: {target_joint_positions}")
        
        # Get current joint positions
        current_positions = list(master_bot.arm.core.joint_states.position[:6])
        print(f"Current joint positions: {current_positions}")
        
        # Move to target joint positions smoothly over 1.5 seconds
        move_time = 1.5
        print(f"Moving robot arm to target position over {move_time} seconds...")
        num_steps = int(move_time / DT)
        trajectory = np.linspace(current_positions, target_joint_positions, num_steps)
        
        for step in range(num_steps):
            master_bot.arm.set_joint_positions(trajectory[step], blocking=False)
            time.sleep(DT)
        
        print("Robot arm has reached target position.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    finally:
        # Cleanup would be handled by ROS node shutdown
        pass


def main():
    """Main function to parse command line arguments and control robot."""
    # Parse command line arguments for joint positions
    if len(sys.argv) > 1:
        if len(sys.argv) == 7:
            # Parse 6 joint positions from command line
            try:
                target_positions = [float(sys.argv[i]) for i in range(1, 7)]
                print(f"Target joint positions from command line: {target_positions}")
            except ValueError as e:
                print(f"Error: Invalid joint position values. All positions must be numbers.")
                print(f"Usage: python control_one_robot.py [waist shoulder elbow forearm_roll wrist_angle wrist_rotate]")
                sys.exit(1)
        else:
            print(f"Error: Expected 6 joint positions, got {len(sys.argv) - 1}")
            print(f"Usage: python control_one_robot.py [waist shoulder elbow forearm_roll wrist_angle wrist_rotate]")
            sys.exit(1)
    else:
        target_positions = None
    
    # Control robot to target position
    control_master_left_to_joint_position(target_positions)


if __name__ == '__main__':
    main()

