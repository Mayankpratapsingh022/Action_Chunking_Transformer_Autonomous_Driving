#!/usr/bin/env python3
"""Convert accepted Urban VLA episodes into a LeRobotDataset v3 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/urban-vla-expert-v1")
    parser.add_argument("--repo-id", default="local/urban-vla-expert-v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        import cv2
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise SystemExit(
            "Install converter dependencies first: "
            "python3 -m pip install 'lerobot>=0.4.0' opencv-python-headless numpy"
        ) from error

    args = parse_args()
    root = Path(args.root).resolve()
    source = root / "raw" / "accepted"
    output = root / "lerobot"
    if not source.exists():
        raise SystemExit(f"Accepted episode directory does not exist: {source}")
    if (output / "meta").exists():
        if not args.overwrite:
            raise SystemExit(f"LeRobot output already exists: {output}; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["speed_mps", "steering", "previous_throttle", "previous_brake"],
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["throttle", "brake", "steering"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        fps=10,
        robot_type="urban_driving_sim",
        features=features,
        use_videos=True,
    )

    episode_directories = sorted(path for path in source.iterdir() if path.is_dir())
    for episode_number, episode_directory in enumerate(episode_directories, start=1):
        metadata = json.loads((episode_directory / "episode.json").read_text())
        telemetry = [
            json.loads(line)
            for line in (episode_directory / "telemetry.jsonl").read_text().splitlines()
            if line.strip()
        ]
        capture = cv2.VideoCapture(str(episode_directory / "front.mp4"))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {episode_directory / 'front.mp4'}")
        decoded = 0
        for row in telemetry:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended early in {episode_directory.name} at frame {decoded}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dataset.add_frame(
                {
                    "observation.images.front": rgb,
                    "observation.state": np.asarray(row["observation"]["state"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "task": metadata["episode"]["instruction"],
                }
            )
            decoded += 1
        extra_frame, _ = capture.read()
        capture.release()
        if extra_frame:
            raise RuntimeError(f"Video has more frames than telemetry in {episode_directory.name}")
        dataset.save_episode()
        print(f"[{episode_number}/{len(episode_directories)}] {episode_directory.name}: {decoded} frames")

    dataset.finalize()
    print(f"LeRobotDataset v3 written to {output}")


if __name__ == "__main__":
    main()
