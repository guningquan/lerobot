"""
Evaluate trained ACT policy on ALOHA2 hardware.

Loads a trained ACT checkpoint and runs closed-loop control on the
ALOHA2 bimanual robot with 4 RealSense camera observations.

Usage:
  conda activate lerobot
  python eval_aloha2_act.py --checkpoint_dir=/path/to/checkpoint

Safety: max_relative_target defaults to 5.0 degrees per step.
Increase gradually once the policy behavior is verified.
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.aloha import Aloha, AlohaConfig
from lerobot.utils.utils import log_say


# ── Camera configuration (same serial numbers as recording) ──────────
FPS = 30
CAMERA_CONFIG = {
    "cam_high": RealSenseCameraConfig(
        serial_number_or_name="109422062625", fps=FPS, width=640, height=480,
    ),
    "cam_low": RealSenseCameraConfig(
        serial_number_or_name="936322072119", fps=FPS, width=640, height=480,
    ),
    "cam_left_wrist": RealSenseCameraConfig(
        serial_number_or_name="134222077139", fps=FPS, width=640, height=480,
    ),
    "cam_right_wrist": RealSenseCameraConfig(
        serial_number_or_name="943222070893", fps=FPS, width=640, height=480,
    ),
}

CALIB_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"


def load_policy(checkpoint_dir: str) -> ACTPolicy:
    """Load trained ACT policy from checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.is_dir():
        # Find the latest checkpoint
        ckpt_files = sorted(checkpoint_path.glob("**/checkpoint-*/model.safetensors"))
        if not ckpt_files:
            ckpt_files = sorted(checkpoint_path.glob("*.safetensors"))
        if not ckpt_files:
            ckpt_files = sorted(checkpoint_path.glob("**/pretrained_model/**/*.safetensors"))
        if not ckpt_files:
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")

        ckpt_path = ckpt_files[-1]
        log_say(f"Loading checkpoint: {ckpt_path}")
    else:
        ckpt_path = Path(checkpoint_path)

    policy = ACTPolicy.from_pretrained(ckpt_path.parent)
    policy.eval()
    if torch.cuda.is_available():
        policy = policy.cuda()
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to trained ACT checkpoint directory")
    parser.add_argument("--max_relative_target", type=float, default=5.0,
                        help="Max joint movement per step in degrees (safety limit)")
    parser.add_argument("--episode_length", type=int, default=300,
                        help="Max steps per evaluation episode")
    parser.add_argument("--temporal_ensemble", type=float, default=None,
                        help="Temporal ensemble coefficient (e.g. 0.01). None = disabled")
    args = parser.parse_args()

    # ── Load policy ──────────────────────────────────────────────────
    policy = load_policy(args.checkpoint_dir)
    log_say("Policy loaded successfully")

    # ── Connect robot ────────────────────────────────────────────────
    robot_config = AlohaConfig(
        id="aloha",
        left_arm_port="/dev/ttyDXL_puppet_left",
        right_arm_port="/dev/ttyDXL_puppet_right",
        left_arm_max_relative_target=args.max_relative_target,
        right_arm_max_relative_target=args.max_relative_target,
        left_arm_use_degrees=True,
        right_arm_use_degrees=True,
        calibration_dir=CALIB_DIR,
        cameras=CAMERA_CONFIG,
    )

    robot = Aloha(robot_config)
    robot.connect()
    log_say("Robot connected")

    # Warm up cameras
    time.sleep(1.0)

    # ── Control loop ─────────────────────────────────────────────────
    try:
        for step in range(args.episode_length):
            start_time = time.perf_counter()

            # Get observation
            obs = robot.get_observation()

            # Convert to policy input format
            # obs dict has keys like: left_waist.pos, right_gripper.pos
            # plus cam_high, cam_low, cam_left_wrist, cam_right_wrist
            state = np.array([obs[f"{side}_{motor}.pos"]
                              for side in ["left", "right"]
                              for motor in robot.left_arm.bus.motors])

            # Prepare batch (add batch dim)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            if torch.cuda.is_available():
                state_tensor = state_tensor.cuda()

            images = {}
            for cam_key in CAMERA_CONFIG:
                img = obs[cam_key]
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                images[cam_key] = img_tensor

            # Get action from policy
            with torch.inference_mode():
                action = policy.select_action({
                    "observation.state": state_tensor,
                    **{f"observation.images.{k}": v for k, v in images.items()},
                })

            # action is a numpy array of shape (n_action_steps, action_dim)
            first_action = action[0]  # take first action step

            # Convert flat action array back to robot action dict
            motor_names = list(robot.left_arm.bus.motors)
            n_per_arm = len(motor_names)
            action_dict = {}
            for i, motor in enumerate(motor_names):
                action_dict[f"left_{motor}.pos"] = float(first_action[i])
                action_dict[f"right_{motor}.pos"] = float(first_action[n_per_arm + i])

            # Send to robot
            robot.send_action(action_dict)

            # Maintain ~30Hz
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, (1.0 / FPS) - elapsed)
            time.sleep(sleep_time)

            if step % 30 == 0:
                log_say(f"Step {step}/{args.episode_length}")

    except KeyboardInterrupt:
        log_say("Evaluation interrupted by user")
    finally:
        robot.disconnect()
        log_say("Robot disconnected")


if __name__ == "__main__":
    main()
