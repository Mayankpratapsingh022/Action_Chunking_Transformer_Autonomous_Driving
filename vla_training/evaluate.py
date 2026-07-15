from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from left_turn_vla.constants import ACTION_NAMES, MAX_SPEED_MPS  # noqa: E402
from left_turn_vla.inference import SmolVLADriver  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a left-turn SmolVLA checkpoint on held-out episodes.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-repo", default="Mayank022/urban-vla-left-turn-cruise-human")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.eval_split < 0.5:
        raise ValueError("eval_split must be greater than 0 and less than 0.5")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError("LeRobot is required for checkpoint evaluation") from error

    dataset = LeRobotDataset(args.dataset_repo, root=args.dataset_root, video_backend=args.video_backend)
    total_episodes = dataset.num_episodes
    eval_count = math.ceil(total_episodes * args.eval_split)
    episode_ids = list(range(total_episodes - eval_count, total_episodes))
    if args.max_episodes > 0:
        episode_ids = episode_ids[: args.max_episodes]
    if not episode_ids:
        raise RuntimeError("No held-out episodes are available")

    driver = SmolVLADriver(args.model_path, device=args.device)
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    episode_summaries: list[dict[str, float | int]] = []

    for position, episode_id in enumerate(episode_ids, start=1):
        driver.reset()
        start = int(dataset.meta.episodes["dataset_from_index"][episode_id])
        end = int(dataset.meta.episodes["dataset_to_index"][episode_id])
        episode_targets: list[np.ndarray] = []
        episode_predictions: list[np.ndarray] = []
        for frame_index in range(start, end):
            frame = dict(dataset[frame_index])
            target = np.asarray(frame["action"], dtype=np.float32)
            prediction = driver.predict_frame(frame)
            targets.append(target)
            predictions.append(prediction)
            episode_targets.append(target)
            episode_predictions.append(prediction)
        episode_metrics = calculate_metrics(np.stack(episode_targets), np.stack(episode_predictions))
        episode_summaries.append(
            {
                "episode": episode_id,
                "frames": end - start,
                "mean_action_mae": episode_metrics["mean_action_mae"],
            }
        )
        print(
            f"[{position:02d}/{len(episode_ids):02d}] episode={episode_id:03d} "
            f"frames={end - start:03d} mae={episode_metrics['mean_action_mae']:.4f}",
            flush=True,
        )

    target_array = np.stack(targets)
    prediction_array = np.stack(predictions)
    metrics = calculate_metrics(target_array, prediction_array)
    metrics.update(
        {
            "model_path": str(args.model_path),
            "dataset_repo": args.dataset_repo,
            "total_dataset_episodes": total_episodes,
            "held_out_episodes": episode_ids,
            "evaluated_frames": len(target_array),
            "episode_metrics": episode_summaries,
        }
    )
    metrics["quality_gates"] = quality_gates(metrics)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_plots(target_array, prediction_array, output_dir)
    print(json.dumps(metrics, indent=2), flush=True)


def calculate_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"targets and predictions must both have shape [frames, {len(ACTION_NAMES)}]")
    error = predictions - targets
    absolute = np.abs(error)
    mae = absolute.mean(axis=0)
    rmse = np.sqrt(np.square(error).mean(axis=0))
    normalized_absolute = absolute.copy()
    normalized_absolute[:, 0] /= MAX_SPEED_MPS
    active = {"target_speed_mps": _activity_metrics(targets[:, 0], predictions[:, 0], threshold=0.5)}
    steering_mask = np.abs(targets[:, 1]) > 0.1
    steering_direction_accuracy = (
        float((np.sign(targets[steering_mask, 1]) == np.sign(predictions[steering_mask, 1])).mean())
        if steering_mask.any()
        else 1.0
    )
    return {
        "mean_action_mae": float(normalized_absolute.mean()),
        "action_mae": {name: float(mae[index]) for index, name in enumerate(ACTION_NAMES)},
        "action_rmse": {name: float(rmse[index]) for index, name in enumerate(ACTION_NAMES)},
        "activity": active,
        "steering_direction_accuracy": steering_direction_accuracy,
        "stopped_target_rate": float((predictions[:, 0] < 0.5).mean()),
    }


def quality_gates(metrics: dict[str, object]) -> dict[str, bool]:
    action_mae = metrics["action_mae"]
    activity = metrics["activity"]
    assert isinstance(action_mae, dict) and isinstance(activity, dict)
    gates = {
        "mean_action_mae": float(metrics["mean_action_mae"]) < 0.10,
        "target_speed_mae": float(action_mae["target_speed_mps"]) < 1.5,
        "target_steering_mae": float(action_mae["target_steering"]) < 0.15,
        "target_speed_recall": float(activity["target_speed_mps"]["recall"]) >= 0.80,
        "steering_direction_accuracy": float(metrics["steering_direction_accuracy"]) >= 0.70,
    }
    gates["all_open_loop_passed"] = all(gates.values())
    return gates


def _activity_metrics(target: np.ndarray, prediction: np.ndarray, threshold: float = 0.05) -> dict[str, float]:
    expected = target > threshold
    actual = prediction > threshold
    true_positive = int((expected & actual).sum())
    false_positive = int((~expected & actual).sum())
    false_negative = int((expected & ~actual).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def write_plots(targets: np.ndarray, predictions: np.ndarray, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(ACTION_NAMES), figsize=(10, 4))
    for index, (axis, name) in enumerate(zip(np.atleast_1d(axes), ACTION_NAMES, strict=True)):
        axis.scatter(targets[:, index], predictions[:, index], s=5, alpha=0.25)
        low = -1.0 if name == "target_steering" else 0.0
        high = 1.0 if name == "target_steering" else max(12.0, float(targets[:, index].max()))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set(title=name, xlabel="human target", ylabel="VLA prediction", xlim=(low, high), ylim=(low, high))
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_scatter.png", dpi=160)
    plt.close(figure)

    count = min(300, len(targets))
    figure, axes = plt.subplots(len(ACTION_NAMES), 1, figsize=(11, 5), sharex=True)
    for index, (axis, name) in enumerate(zip(axes, ACTION_NAMES, strict=True)):
        axis.plot(targets[:count, index], label="human", linewidth=1.4)
        axis.plot(predictions[:count, index], label="SmolVLA", linewidth=1.1, alpha=0.85)
        axis.set_ylabel(name)
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("held-out frame")
    figure.tight_layout()
    figure.savefig(output_dir / "action_trace.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
