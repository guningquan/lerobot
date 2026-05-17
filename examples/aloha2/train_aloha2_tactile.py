"""
Train ACT policy on ALOHA2 dataset with flexible tactile channel selection.

Supports per-channel ablation to quantify the contribution of each tactile dimension.

Usage:
  conda activate lerobot

  # All 17 channels (full)
  python train_aloha2_tactile.py --dataset_repo_id=theodoreliu/aloha_wear_shoe --tactile_channels=all

  # No tactile (visual baseline)
  python train_aloha2_tactile.py --dataset_repo_id=theodoreliu/aloha_wear_shoe --tactile_channels=visual_only

  # Single dimension or physical-state group
  python train_aloha2_tactile.py --dataset_repo_id=theodoreliu/aloha_wear_shoe --tactile_channels=rectify
  python train_aloha2_tactile.py --dataset_repo_id=theodoreliu/aloha_wear_shoe --tactile_channels=force
  python train_aloha2_tactile.py --dataset_repo_id=theodoreliu/aloha_wear_shoe --tactile_channels=force_all
  python train_aloha2_tactile.py --dataset_repo_id=theodoreliu/aloha_wear_shoe --tactile_channels=depth
  python train_aloha2_tactile.py --dataset_repo_id=theodoreliu/aloha_wear_shoe --tactile_channels=marker2d

  # Combinations
  python train_aloha2_tactile.py --dataset_repo_id=theodoreliu/aloha_wear_shoe --tactile_channels=rectify,force,depth

Tactile channel mapping (default output_types order):
  rectify(1) + difference(1) + depth(1) + force(3) + force_norm(3) + force_resultant(6) + marker2d(2) = 17
  Indices:  rectify=[0], difference=[1], depth=[2], force=[3,4,5], force_norm=[6,7,8],
            force_resultant=[9..14], marker2d=[15,16]
  force_all = force + force_norm + force_resultant = 12 force-related channels.
"""

import argparse
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

# Channel name → indices in the 17-channel tactile tensor
TACTILE_CHANNEL_MAP: dict[str, list[int]] = {
    "rectify":         [0],
    "difference":      [1],
    "depth":           [2],
    "force":           [3, 4, 5],
    "force_norm":      [6, 7, 8],
    "force_resultant": [9, 10, 11, 12, 13, 14],
    "force_all":       [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "marker2d":        [15, 16],
}

ALL_CHANNELS = sorted(set(
    i for indices in TACTILE_CHANNEL_MAP.values() for i in indices
))


def parse_tactile_channels(arg: str) -> tuple[str, bool, list[int] | None]:
    """Parse --tactile_channels argument.
    Returns (label, use_tactile, indices_or_None_for_all_channels).
    """
    if arg == "visual_only":
        return "visual_only", False, None
    if arg == "all":
        return "all", True, ALL_CHANNELS
    parts = [p.strip() for p in arg.split(",")]
    indices = []
    for p in parts:
        if p not in TACTILE_CHANNEL_MAP:
            raise ValueError(f"Unknown tactile channel: {p}. Choices: {list(TACTILE_CHANNEL_MAP)}")
        indices.extend(TACTILE_CHANNEL_MAP[p])
    return "-".join(parts), True, sorted(set(indices))


def build_config(
    dataset_repo_id: str,
    output_dir: str,
    tactile_label: str,
    use_tactile: bool,
    tactile_indices: list[int] | None,
    steps: int = 100000,
    batch_size: int = 8,
) -> TrainPipelineConfig:
    """Build the training config."""

    norm = {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    if use_tactile:
        norm["TACTILE"] = NormalizationMode.MEAN_STD

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=dataset_repo_id,
            root=f"/home/robot/Dataset_and_Checkpoint/lerobot-dataset/{dataset_repo_id}",
        ),
        policy=ACTConfig(
            repo_id=f"theodoreliu/act_{tactile_label}",
            n_obs_steps=1,
            chunk_size=100,
            n_action_steps=100,
            vision_backbone="resnet18",
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
            replace_final_stride_with_dilation=False,
            dim_model=512,
            n_heads=8,
            dim_feedforward=3200,
            feedforward_activation="relu",
            n_encoder_layers=4,
            n_decoder_layers=1,
            pre_norm=False,
            dropout=0.1,
            use_vae=True,
            latent_dim=32,
            n_vae_encoder_layers=4,
            kl_weight=10.0,
            temporal_ensemble_coeff=None,
            optimizer_lr=1e-5,
            optimizer_lr_backbone=1e-5,
            optimizer_weight_decay=1e-4,
            normalization_mapping=norm,
            tactile_channel_indices=tactile_indices,
            use_tactile=use_tactile,
        ),
        output_dir=output_dir,
        batch_size=batch_size,
        steps=steps,
        num_workers=4,
        save_freq=10000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ACT with flexible tactile channel selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Channel choices: " + ", ".join(TACTILE_CHANNEL_MAP)
    )
    parser.add_argument("--dataset_repo_id", type=str, required=True)
    parser.add_argument(
        "--tactile_channels", type=str, default="visual_only",
        help="Comma-separated channels: visual_only, all, or e.g. rectify,force,depth",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/robot/Dataset_and_Checkpoint/lerobot-checkpoint",
    )
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    tactile_label, use_tactile, tactile_indices = parse_tactile_channels(args.tactile_channels)
    n_channels = len(tactile_indices or []) if use_tactile else 0
    print(f"Tactile: {tactile_label} -> use_tactile={use_tactile}, indices={tactile_indices} ({n_channels} channels)")

    task_name = args.dataset_repo_id.split("/")[-1]
    output_dir = str(Path(args.output_dir) / task_name / tactile_label)

    config = build_config(
        dataset_repo_id=args.dataset_repo_id,
        output_dir=Path(output_dir),
        tactile_label=tactile_label,
        use_tactile=use_tactile,
        tactile_indices=tactile_indices,
        steps=args.steps,
        batch_size=args.batch_size,
    )

    print(f"Training ACT on: {args.dataset_repo_id}")
    print(f"Tactile mode: {tactile_label}")
    print(f"Output: {output_dir}")
    print(f"Steps: {args.steps}")

    train(config)
