#!/usr/bin/env python3
"""Validate an ALOHA2 + XENSE tactile data collection batch.

Run this immediately after a 5-episode pilot collection before recording more:

  conda activate lerobot
  python examples/aloha2/check_collection_batch.py --task aloha_flip_switch --min_episodes 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path("/home/robot/Dataset_and_Checkpoint/lerobot-dataset")
DEFAULT_OWNER = "theodoreliu"
FPS = 30
EXPECTED_TACTILE_SHAPE = (17, 120, 160)

CAMERAS = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
SENSORS = ["left_fingertip", "left_knuckle", "right_fingertip", "right_knuckle"]

TASK_EXPECTED_FRAMES = {
    "aloha_flip_switch": 300,
    "aloha_hidden_property_grasp": 450,
    "aloha_slip_hold_or_pull": 750,
    "aloha_pressure_wipe": 450,
    "aloha_peg_insertion": 600,
    "aloha_towel_unfold": 600,
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def dataset_dir(root: Path, owner: str, task_or_repo: str) -> Path:
    if "/" in task_or_repo:
        return root / task_or_repo
    return root / owner / task_or_repo


def as_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        return np.stack([np.stack(ch) for ch in arr])
    return arr


def find_files(base: Path, pattern: str) -> list[Path]:
    return sorted(base.glob(pattern))


def check_meta(base: Path) -> list[CheckResult]:
    results = []
    for rel in ["meta/info.json", "meta/stats.json", "meta/tasks.parquet", "meta/episodes"]:
        path = base / rel
        results.append(CheckResult(rel, path.exists(), "exists" if path.exists() else "missing"))
    return results


def read_info(base: Path) -> dict[str, Any]:
    info_path = base / "meta" / "info.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text())
    except json.JSONDecodeError:
        return {}


def check_videos(base: Path, min_episodes: int) -> list[CheckResult]:
    results = []
    info = read_info(base)
    total_episodes = int(info.get("total_episodes", 0))
    for cam in CAMERAS:
        video_dir = base / "videos" / f"observation.images.{cam}"
        files = find_files(video_dir, "chunk-*/*.mp4")
        ok = len(files) > 0 and total_episodes >= min_episodes
        results.append(
            CheckResult(
                f"video:{cam}",
                ok,
                f"{len(files)} chunk mp4 files, total_episodes={total_episodes}, under {video_dir}",
            )
        )
    return results


def check_parquet_and_tactile(
    base: Path,
    task_name: str,
    min_episodes: int,
    frame_tolerance: int,
) -> list[CheckResult]:
    results = []
    pq_files = find_files(base / "data", "chunk-*/*.parquet")
    results.append(CheckResult("parquet_files", len(pq_files) >= min_episodes, f"{len(pq_files)} files"))
    if not pq_files:
        return results

    expected_frames = TASK_EXPECTED_FRAMES.get(task_name)
    seen_episodes: set[int | str] = set()
    bad_shapes: list[str] = []
    bad_lengths: list[str] = []
    readable = 0

    for pq in pq_files:
        try:
            df = pd.read_parquet(pq)
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(f"read:{pq.name}", False, f"{type(exc).__name__}: {exc}"))
            continue

        readable += 1
        if "episode_index" in df.columns:
            seen_episodes.update(int(x) for x in df["episode_index"].dropna().unique())
        else:
            seen_episodes.add(pq.stem)

        if expected_frames is not None and abs(len(df) - expected_frames) > frame_tolerance:
            bad_lengths.append(f"{pq.name}: {len(df)} rows, expected {expected_frames}")

        if df.empty:
            bad_lengths.append(f"{pq.name}: empty")
            continue

        row = df.iloc[min(len(df) // 2, len(df) - 1)]
        for sensor in SENSORS:
            col = f"observation.tactiles.tactile_{sensor}"
            if col not in df.columns:
                bad_shapes.append(f"{pq.name}:{sensor}: missing column")
                continue
            shape = as_array(row[col]).shape
            if shape != EXPECTED_TACTILE_SHAPE:
                bad_shapes.append(f"{pq.name}:{sensor}: shape={shape}")

    results.append(CheckResult("readable_parquet", readable == len(pq_files), f"{readable}/{len(pq_files)} readable"))
    results.append(
        CheckResult(
            "episode_count",
            len(seen_episodes) >= min_episodes,
            f"{len(seen_episodes)} episodes detected",
        )
    )
    results.append(
        CheckResult(
            "episode_lengths",
            not bad_lengths,
            "ok" if not bad_lengths else "; ".join(bad_lengths[:5]),
        )
    )
    results.append(
        CheckResult(
            "tactile_shape",
            not bad_shapes,
            f"all sampled tactile frames are {EXPECTED_TACTILE_SHAPE}" if not bad_shapes else "; ".join(bad_shapes[:8]),
        )
    )
    return results


def check_stats_channels(base: Path) -> list[CheckResult]:
    stats_path = base / "meta" / "stats.json"
    if not stats_path.exists():
        return [CheckResult("stats_tactile", False, "meta/stats.json missing")]
    try:
        stats = json.loads(stats_path.read_text())
    except json.JSONDecodeError as exc:
        return [CheckResult("stats_tactile", False, f"invalid stats.json: {exc}")]

    missing = []
    for sensor in SENSORS:
        key = f"observation.tactiles.tactile_{sensor}"
        if key not in stats:
            missing.append(key)
    return [
        CheckResult(
            "stats_tactile",
            not missing,
            "all tactile stats present" if not missing else "missing: " + ", ".join(missing),
        )
    ]


def print_results(results: list[CheckResult]) -> bool:
    all_ok = True
    for result in results:
        mark = "OK" if result.ok else "FAIL"
        print(f"[{mark}] {result.name}: {result.detail}")
        all_ok = all_ok and result.ok
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ALOHA2 formal-task collection readiness.")
    parser.add_argument("--task", required=True, help="Task name, e.g. aloha_flip_switch, or repo id.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--min_episodes", type=int, default=5)
    parser.add_argument("--frame_tolerance", type=int, default=60)
    args = parser.parse_args()

    base = dataset_dir(args.root, args.owner, args.task)
    task_name = args.task.split("/")[-1]
    print(f"Dataset: {base}")
    print(f"Expected tactile shape: {EXPECTED_TACTILE_SHAPE}")
    print(f"Expected frames for task: {TASK_EXPECTED_FRAMES.get(task_name, 'unknown')}")

    results = [CheckResult("dataset_dir", base.exists(), "exists" if base.exists() else "missing")]
    if base.exists():
        results.extend(check_meta(base))
        results.extend(check_videos(base, args.min_episodes))
        results.extend(check_parquet_and_tactile(base, task_name, args.min_episodes, args.frame_tolerance))
        results.extend(check_stats_channels(base))

    ok = print_results(results)
    if ok:
        print("\nCollection batch passed. It is reasonable to continue recording this task.")
        return 0
    print("\nCollection batch has issues. Fix them before recording more episodes.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
