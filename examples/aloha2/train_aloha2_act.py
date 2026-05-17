"""
Train ACT policy on ALOHA2 4-camera dataset.

Usage:
  conda activate lerobot
  python train_aloha2_act.py --dataset_repo_id=your_username/aloha2_4cam_task

Or edit the config dict below and run directly.
"""

import argparse
from pathlib import Path

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train


def build_config(dataset_repo_id: str, output_dir: str, steps: int = 100000) -> TrainPipelineConfig:
    return TrainPipelineConfig(
        dataset=dict(
            repo_id=dataset_repo_id,
            root=None,
        ),
        policy=ACTConfig(
            # ── Observation / action ──
            n_obs_steps=1,
            chunk_size=100,
            n_action_steps=100,

            # ── Vision backbone ──
            vision_backbone="resnet18",
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            replace_final_stride_with_dilation=False,

            # ── Transformer architecture ──
            dim_model=512,
            n_heads=8,
            dim_feedforward=3200,
            feedforward_activation="relu",
            n_encoder_layers=4,
            n_decoder_layers=1,       # LeRobot default (matches original ACT effective behavior)
            pre_norm=False,
            dropout=0.1,

            # ── VAE ──
            use_vae=True,
            latent_dim=32,
            n_vae_encoder_layers=4,
            kl_weight=10.0,

            # ── Temporal ensembling (disabled during training, enabled at eval) ──
            temporal_ensemble_coeff=None,

            # ── Optimization ──
            optimizer_lr=1e-5,
            optimizer_lr_backbone=1e-5,
            optimizer_weight_decay=1e-4,

            # ── Normalization ──
            normalization_mapping="VISUAL:MEAN_STD,STATE:MEAN_STD,ACTION:MEAN_STD",
        ),
        output_dir=output_dir,
        batch_size=8,
        steps=steps,
        num_workers=4,
        save_every=10000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                        help="HuggingFace dataset repo ID, e.g. 'your_username/aloha2_4cam_task'")
    parser.add_argument("--output_dir", type=str,
                        default="/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint/aloha2_4cam_baseline",
                        help="Directory to save checkpoints")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Number of training steps")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = build_config(
        dataset_repo_id=args.dataset_repo_id,
        output_dir=args.output_dir,
        steps=args.steps,
    )

    print(f"Training ACT on dataset: {args.dataset_repo_id}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.steps}")
    print(f"Policy config: {config.policy}")
    print(f"Batch size: {config.batch_size}")

    train(config)
