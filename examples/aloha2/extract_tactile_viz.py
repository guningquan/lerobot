#!/usr/bin/env python3
"""Extract tactile frames from LeRobot FULL-mode dataset and save as viewable PNGs.

Usage:
  conda activate lerobot
  python extract_tactile_viz.py --task aloha_wear_shoe --episode 0 --frames 5

Output structure per frame:
  {output_dir}/
    frame_000/
      left_fingertip_rectify.png     # 校正图像
      left_fingertip_depth.png       # 深度图 (热力图)
      left_fingertip_force.png       # 法向力分布 (热力图)
      left_fingertip_force_norm.png  # 归一化力
      left_fingertip_marker2d.png    # 切向位移场
      ... (same for other 3 sensors)
      montage.png                    # 4 传感器 × 各通道汇总图
"""
import argparse, sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Channel mapping (must match xense_g1ws/config_xense_g1ws.py default output_types)
CHANNELS = {
    "rectify":         [0],
    "difference":      [1],
    "depth":           [2],
    "force":           [3, 4, 5],
    "force_norm":      [6, 7, 8],
    "force_resultant": [9, 10, 11, 12, 13, 14],
    "marker2d":        [15, 16],
}

SENSORS = [
    "left_fingertip", "left_knuckle",
    "right_fingertip", "right_knuckle",
]


def save_channel(data, fpath, cmap="gray", title=""):
    """Save a single-channel or 3-channel tactile slice as PNG."""
    fig, ax = plt.subplots(figsize=(3, 3))
    if data.ndim == 2:
        im = ax.imshow(data, cmap=cmap if cmap else "viridis")
    else:
        im = ax.imshow(data)
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(fpath, dpi=100, bbox_inches="tight")
    plt.close(fig)


def extract_frame(dataset, frame_idx: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = dataset[frame_idx]
    print(f"\nFrame {frame_idx}:")

    for sensor in SENSORS:
        key = f"observation.tactiles.tactile_{sensor}"
        if key not in frame:
            print(f"  {sensor}: NOT FOUND")
            continue
        data = frame[key].numpy()  # (17, 120, 160)
        print(f"  {sensor}: shape={data.shape}")

        # Rectify (channel 0 — grayscale contact image)
        save_channel(data[0], out_dir / f"{sensor}_rectify.png", cmap="gray",
                     title=f"{sensor} Rectify")

        # Depth (channel 2 — depth map)
        save_channel(data[2], out_dir / f"{sensor}_depth.png", cmap="inferno",
                     title=f"{sensor} Depth (mm)")

        # Force norm (average of channels 6,7,8)
        force_norm = data[6:9].mean(axis=0)
        save_channel(force_norm, out_dir / f"{sensor}_force.png", cmap="hot",
                     title=f"{sensor} Force Norm")

        # Marker2D (channels 15,16 — displacement magnitude)
        marker_mag = np.sqrt(data[15]**2 + data[16]**2)
        save_channel(marker_mag, out_dir / f"{sensor}_marker2d.png", cmap="coolwarm",
                     title=f"{sensor} Marker2D Mag")

    print(f"  saved to {out_dir}")


def make_montage(frame_idx: int, base_dir: Path):
    """Create a 4-sensor × 4-channel montage."""
    channels = ["rectify", "depth", "force", "marker2d"]
    fig, axes = plt.subplots(len(SENSORS), len(channels), figsize=(12, 10))
    for s_idx, sensor in enumerate(SENSORS):
        for c_idx, ch in enumerate(channels):
            ax = axes[s_idx][c_idx]
            img_path = base_dir / f"{sensor}_{ch}.png"
            if img_path.exists():
                img = plt.imread(str(img_path))
                ax.imshow(img)
            ax.set_title(f"{sensor}\n{ch}" if s_idx == 0 else ch, fontsize=7)
            ax.axis("off")
    fig.suptitle(f"Frame {frame_idx} — Tactile Overview", fontsize=12)
    fig.tight_layout()
    fig.savefig(base_dir / "montage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMontage saved to {base_dir / 'montage.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="aloha_wear_shoe")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--output", default="/mnt/server/Programs_server/metalerobot/outputs/tactile_viz")
    args = parser.parse_args()

    import pandas as pd

    data_dir = Path(f"/home/robot/Dataset_and_Checkpoint/lerobot-dataset/theodoreliu/{args.task}/data/chunk-000")
    pq_files = sorted(data_dir.glob("*.parquet"))
    if not pq_files:
        print(f"No parquet files in {data_dir}")
        sys.exit(1)

    print(f"Loading {pq_files[0]} ...")
    df = pd.read_parquet(pq_files[0])
    print(f"Loaded {len(df)} frames, columns: {list(df.columns)}")

    tactile_cols = [c for c in df.columns if "tactile" in c]
    print(f"Tactile columns: {tactile_cols}")

    for i in range(min(args.frames, len(df))):
        frame_idx = int(len(df) * i / min(args.frames, len(df)))
        if frame_idx >= len(df):
            frame_idx = len(df) - 1
        row = df.iloc[frame_idx]
        print(f"\n{'='*50}")
        print(f"Frame {frame_idx}:")

        out_dir = Path(args.output) / f"frame_{i:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for sensor in SENSORS:
            col = f"observation.tactiles.tactile_{sensor}"
            if col not in df.columns:
                print(f"  {sensor}: NOT FOUND")
                continue
            raw = row[col]
            if isinstance(raw, np.ndarray) and raw.dtype == object:
                data = np.stack([np.stack(ch) for ch in raw])  # (17,) of (120,) of (160,) → (17,120,160)
            else:
                data = np.array(raw)
            print(f"  {sensor}: shape={data.shape}")

            save_channel(data[0], out_dir / f"{sensor}_rectify.png", cmap="gray",
                         title=f"{sensor} Rectify")
            save_channel(data[2], out_dir / f"{sensor}_depth.png", cmap="inferno",
                         title=f"{sensor} Depth")
            fn_mean = data[6:9].mean(axis=0)
            save_channel(fn_mean, out_dir / f"{sensor}_force.png", cmap="hot",
                         title=f"{sensor} Force Norm")
            mm = np.sqrt(data[15]**2 + data[16]**2)
            save_channel(mm, out_dir / f"{sensor}_marker2d.png", cmap="coolwarm",
                         title=f"{sensor} Marker2D Mag")
        print(f"  saved to {out_dir}")
        make_montage(i, out_dir)

    print(f"\nDone. View: {args.output}/")


if __name__ == "__main__":
    main()
