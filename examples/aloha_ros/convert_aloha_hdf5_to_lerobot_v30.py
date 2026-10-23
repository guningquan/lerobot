#!/usr/bin/env python
"""Convert ALOHA hdf5 episodes directly into a LeRobot v3.0 dataset.

Replaces the two-stage route (openpi converter -> LeRobot v2.1 -> convert_dataset_v21_to_v30 ->
fix_feature_names -> augment_dataset_quantile_stats) with a single pass.

Raw episode layout (written by aloha_scripts/record_episodes*.py):
    /action                         (T, 14) float32
    /observations/qpos              (T, 14) float32
    /observations/qvel              (T, 14) float32   optional
    /observations/effort            (T, 14) float32   optional
    /observations/images/<camera>   (T, H, W, 3) uint8, or (T, N) uint8 JPEG buffers when the
                                    file attribute `compress` is True

The 14 joint values are ordered [left arm x6, left gripper, right arm x6, right gripper]
(see real_env.get_qpos), so features use the same names as the aloha_ros robot
(`left_waist.pos`, ..., `right_gripper.pos`). The openpi converter labelled them right-first.

Images are decoded with `cv2.imdecode(buf, 1)` and stored as decoded, without a BGR/RGB swap,
matching openpi's convert_aloha_data_to_lerobot_no_bgr2rgb.py (verified to give correct RGB).
Cameras are stored as MP4 videos by default (LeRobot's default); --use-images stores lossless PNG
frames instead, pixel-identical to the openpi image-mode datasets but ~40x larger.

meta/stats.json gets exact whole-dataset quantiles (q01/q10/q50/q90/q99) for every numeric
feature, computed from the raw arrays, so augment_dataset_quantile_stats.py is not needed.

Example:
    python examples/aloha_ros/convert_aloha_hdf5_to_lerobot_v30.py \
        --raw-dir /mnt/ssd2/guningquan/Dataset/packaging \
        --output-dir /mnt/ssd2/guningquan/Dataset_v30/packaging_v30 \
        --task "packaging the items" \
        --fps 50
"""

import argparse
import logging
import re
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ARM_MOTORS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")
JOINT_NAMES = [f"{side}_{motor}" for side in ("left", "right") for motor in ARM_MOTORS]

# hdf5 dataset path -> (LeRobot feature key, name suffix)
NUMERIC_SOURCES = {
    "/observations/qpos": ("observation.state", ".pos"),
    "/action": ("action", ".pos"),
    "/observations/qvel": ("observation.velocity", ".vel"),
    "/observations/effort": ("observation.effort", ".effort"),
}


def episode_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse an episode number from {path.name}")
    return int(match.group(1))


def decode_image(raw: np.ndarray, compressed: bool) -> np.ndarray:
    if not compressed:
        return np.asarray(raw)
    image = cv2.imdecode(np.asarray(raw), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode failed on a compressed frame")
    return image


def inspect_first_episode(
    path: Path, cameras: list[str] | None
) -> tuple[list[str], dict, bool, dict[str, tuple]]:
    """Return (cameras, numeric sources present, compressed, decoded image shape per camera)."""
    with h5py.File(path, "r") as f:
        compressed = bool(f.attrs.get("compress", False))
        available = sorted(f["/observations/images"].keys())
        cameras = cameras or [cam for cam in available if "depth" not in cam]
        missing = [cam for cam in cameras if cam not in available]
        if missing:
            raise ValueError(f"Cameras {missing} not in {path.name}; available: {available}")
        sources = {src: dst for src, dst in NUMERIC_SOURCES.items() if src in f}
        for required in ("/observations/qpos", "/action"):
            if required not in sources:
                raise ValueError(f"{path.name} has no {required}")
        for src in sources:
            if f[src].shape[1] != len(JOINT_NAMES):
                raise ValueError(f"{path.name}:{src} has shape {f[src].shape}, expected (T, {len(JOINT_NAMES)})")
        image_shapes = {
            cam: decode_image(f[f"/observations/images/{cam}"][0], compressed).shape for cam in cameras
        }
    return cameras, sources, compressed, image_shapes


def build_features(sources: dict, image_shapes: dict, use_videos: bool) -> dict:
    features = {}
    for key, suffix in sources.values():
        features[key] = {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": [f"{name}{suffix}" for name in JOINT_NAMES],
        }
    for cam, shape in image_shapes.items():
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"Camera {cam} decodes to shape {shape}, expected (H, W, 3)")
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": tuple(shape),
            "names": ["height", "width", "channels"],
        }
    return features


def write_exact_quantiles(root: Path, numeric: dict[str, list[np.ndarray]]) -> None:
    """Replace the per-episode quantile envelope in meta/stats.json with exact dataset quantiles."""
    stats = load_stats(root)
    for key, chunks in numeric.items():
        values = np.concatenate(chunks, axis=0).astype(np.float64)
        for q in DEFAULT_QUANTILES:
            stats[key][f"q{int(round(q * 100)):02d}"] = np.quantile(values, q, axis=0)
    write_stats(stats, root)


def convert(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    files = sorted(raw_dir.glob("episode_*.hdf5"), key=episode_number)
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 files in {raw_dir}")
    if args.episodes is not None:
        wanted = set(args.episodes)
        files = [p for p in files if episode_number(p) in wanted]
        if len(files) != len(wanted):
            raise ValueError(f"Some requested episodes were not found in {raw_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    cameras, sources, compressed, image_shapes = inspect_first_episode(files[0], args.cameras)
    features = build_features(sources, image_shapes, args.use_videos)
    logging.info(
        f"{len(files)} episodes ({files[0].name} .. {files[-1].name}), cameras={cameras}, "
        f"numeric={[dst[0] for dst in sources.values()]}, compressed={compressed}, "
        f"storage={'video' if args.use_videos else 'image'}, fps={args.fps}"
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id or f"local/{output_dir.name}",
        root=output_dir,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=args.use_videos,
        # PNG encoding holds the GIL, so threads alone serialize; processes give real parallelism.
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    numeric: dict[str, list[np.ndarray]] = {key: [] for key, _ in sources.values()}
    episode_map = []
    for path in tqdm(files, desc="episodes"):
        with h5py.File(path, "r") as f:
            if bool(f.attrs.get("compress", False)) != compressed:
                raise ValueError(f"{path.name}: compress attribute differs from {files[0].name}")
            arrays = {key: np.asarray(f[src][:], dtype=np.float32) for src, (key, _) in sources.items()}
            num_frames = arrays["action"].shape[0]
            for key, values in arrays.items():
                if values.shape[0] != num_frames:
                    raise ValueError(f"{path.name}: {key} has {values.shape[0]} frames, action has {num_frames}")
                numeric[key].append(values)
            images = {cam: f[f"/observations/images/{cam}"] for cam in cameras}
            for cam, data in images.items():
                if data.shape[0] != num_frames:
                    raise ValueError(f"{path.name}: {cam} has {data.shape[0]} frames, action has {num_frames}")

            for i in range(num_frames):
                frame = {key: values[i] for key, values in arrays.items()}
                for cam, data in images.items():
                    frame[f"observation.images.{cam}"] = decode_image(data[i], compressed)
                frame["task"] = args.task
                dataset.add_frame(frame)
        dataset.save_episode()
        episode_map.append(f"{len(episode_map)}\t{path.name}\t{num_frames}")

    dataset.finalize()
    # PNG frames are embedded into parquet on save_episode; drop the emptied staging tree.
    staging = output_dir / "images"
    if staging.is_dir() and not any(p.is_file() for p in staging.rglob("*")):
        shutil.rmtree(staging)
    write_exact_quantiles(output_dir, numeric)
    (output_dir / "meta" / "source_episodes.tsv").write_text(
        "episode_index\tsource_file\tnum_frames\n" + "\n".join(episode_map) + "\n"
    )
    logging.info(f"Done: {len(files)} episodes, {sum(len(c) for c in numeric['action'])} frames -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="Directory with episode_*.hdf5 files")
    parser.add_argument("--output-dir", required=True, help="Dataset root to create (must not exist)")
    parser.add_argument("--task", required=True, help="Language instruction stored with every frame")
    parser.add_argument("--fps", type=int, default=50, help="Recording rate (1 / DT of the recording script)")
    parser.add_argument("--repo-id", default=None, help="Dataset name in metadata (default: local/<output dir name>)")
    parser.add_argument("--robot-type", default="aloha")
    parser.add_argument("--cameras", nargs="+", default=None, help="Cameras to keep (default: all in the hdf5)")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Only these episode numbers")
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--use-videos", dest="use_videos", action="store_true", help="Store cameras as MP4 videos (default)"
    )
    storage.add_argument(
        "--use-images",
        dest="use_videos",
        action="store_false",
        help="Store cameras as lossless PNG frames inside parquet (pixel-identical to the decoded hdf5)",
    )
    parser.set_defaults(use_videos=True)
    # Both modes stage every frame as PNG first (video mode encodes the staged frames per episode).
    parser.add_argument("--image-writer-processes", type=int, default=8, help="Processes writing PNG frames")
    parser.add_argument("--image-writer-threads", type=int, default=4, help="Threads per PNG writer process")
    parser.add_argument("--overwrite", action="store_true", help="Delete --output-dir first if it exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    convert(args)


if __name__ == "__main__":
    main()
