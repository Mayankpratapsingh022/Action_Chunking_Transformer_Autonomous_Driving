from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from left_turn_vla.constants import CAPTURE_FPS  # noqa: E402
from left_turn_vla.env import load_env_file  # noqa: E402
from left_turn_vla.raw_dataset import (  # noqa: E402
    accepted_sources,
    analyze_directory,
    iter_lerobot_frames,
    lerobot_features,
    load_episode,
)

DEFAULT_INPUT_DIR = PROJECT_DIR.parent.parent / "left-turn-target"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "left-turn-cruise-lerobot"
DEFAULT_REPO_ID = "Mayank022/urban-vla-left-turn-cruise-human"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert independent simulator left-turn recordings into LeRobot v3 format."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO", DEFAULT_REPO_ID))
    parser.add_argument("--include-recovery", action="store_true")
    parser.add_argument("--min-frames", type=int, default=60)
    parser.add_argument("--min-progress", type=float, default=0.95)
    parser.add_argument("--max-final-lateral-error-m", type=float, default=2.0)
    parser.add_argument("--max-final-heading-error-rad", type=float, default=0.15)
    parser.add_argument("--max-stopped-frames", type=int, default=20)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", help="Inspect and report without writing a dataset.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_env_file(PROJECT_DIR / ".env")
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    inspections, report = analyze_directory(
        input_dir,
        include_recovery=args.include_recovery,
        min_frames=args.min_frames,
        min_progress=args.min_progress,
        max_final_lateral_error_m=args.max_final_lateral_error_m,
        max_final_heading_error_rad=args.max_final_heading_error_rad,
        max_stopped_frames=args.max_stopped_frames,
        shuffle_seed=args.shuffle_seed,
        eval_split=args.eval_split,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2), flush=True)

    accepted = accepted_sources(inspections)
    if args.dry_run:
        rejected = [item for item in report["episodes"] if not item["accepted"]]
        if rejected:
            print(f"Rejected episodes: {len(rejected)}", flush=True)
            for item in rejected[:20]:
                print(f"  {item['source']}: {item['rejection_reason']}", flush=True)
        if not accepted:
            print("No vla-urban-4 target-action episodes have passed yet.", flush=True)
        return
    if not accepted:
        raise RuntimeError("No vla-urban-4 target-action episodes passed the conversion filters")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobot is not installed. Run `python -m pip install -e .` inside vla_training first."
        ) from error

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_dir,
        fps=CAPTURE_FPS,
        robot_type="vla-urban-simulator-car",
        features=lerobot_features(),
        use_videos=True,
        image_writer_threads=max(2, min(8, os.cpu_count() or 2)),
        batch_encoding_size=1,
    )

    converted_frames = 0
    try:
        for episode_index, source_name in enumerate(accepted, start=1):
            payload = load_episode(input_dir / source_name)
            for frame in iter_lerobot_frames(payload):
                dataset.add_frame(frame)
                converted_frames += 1
            dataset.save_episode(parallel_encoding=False)
            percent = episode_index / len(accepted) * 100
            print(
                f"[{episode_index:03d}/{len(accepted):03d}] {percent:6.2f}%  "
                f"frames={converted_frames:06d}  {source_name}",
                flush=True,
            )
    finally:
        dataset.finalize()

    report["output_dir"] = str(output_dir)
    report["repo_id"] = args.repo_id
    report["converted_frames"] = converted_frames
    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"LeRobot dataset written to {output_dir}", flush=True)
    print(f"Quality report: {report_path}", flush=True)

    if args.push_to_hub:
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("HF_TOKEN is required with --push-to-hub")
        dataset.push_to_hub(
            tags=["lerobot", "smolvla", "autonomous-driving", "left-turn", "human-demonstrations"],
            private=args.private,
            upload_large_folder=True,
        )
        print(f"Published dataset: https://huggingface.co/datasets/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
