from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from urban_act.metrics import ACTION_NAMES, EvaluationSamples

COLORS = ("#1f77b4", "#d62728", "#2ca02c")


def write_all_plots(
    output_dir: str | Path,
    history: dict[str, list[dict[str, Any]]],
    metrics: dict[str, Any],
    samples: EvaluationSamples,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, root / "training_curves.png")
    plot_action_mae(metrics, root / "action_mae.png")
    plot_prediction_scatter(samples, root / "prediction_scatter.png")
    plot_sample_chunks(samples, root / "sample_chunks.png")


def plot_training_curves(history: dict[str, list[dict[str, Any]]], path: Path) -> None:
    train = history.get("train", [])
    validation = history.get("validation", [])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if train:
        steps = [row["step"] for row in train]
        axes[0].plot(steps, [row["loss"] for row in train], label="total")
        axes[0].plot(steps, [row["reconstruction_loss"] for row in train], label="reconstruction")
        for key, label in (
            ("activity_loss", "activity"),
            ("magnitude_loss", "magnitude"),
            ("steering_loss", "steering"),
            ("overlap_loss", "throttle/brake overlap"),
        ):
            if key in train[0]:
                axes[0].plot(steps, [row[key] for row in train], label=label, alpha=0.8)
        axes[0].plot(steps, [row["kl_loss"] for row in train], label="KL")
    axes[0].set(title="Training losses", xlabel="Step", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    if validation:
        validation_steps = [row["step"] for row in validation]
        axes[1].plot(
            validation_steps,
            [row["mean_action_mae"] for row in validation],
            color="#9467bd",
            label="all-step MAE",
        )
        if "selection_score" in validation[0]:
            axes[1].plot(
                validation_steps,
                [row["selection_score"] for row in validation],
                color="#ff7f0e",
                label="selection score",
            )
    axes[1].set(title="Validation policy quality", xlabel="Step", ylabel="Lower is better")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_action_mae(metrics: dict[str, Any], path: Path) -> None:
    values = np.asarray([metrics["action_mae"][name] for name in ACTION_NAMES])
    active_values = np.asarray([metrics.get("active_action_mae", metrics["action_mae"])[name] for name in ACTION_NAMES])
    baseline_values = np.asarray(
        [metrics.get("zero_baseline_mae", metrics["action_mae"])[name] for name in ACTION_NAMES]
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(ACTION_NAMES))
    width = 0.25
    axis.bar(positions - width, values, width, label="all steps", color="#1f77b4")
    axis.bar(positions, active_values, width, label="active targets", color="#ff7f0e")
    axis.bar(positions + width, baseline_values, width, label="zero baseline", color="#7f7f7f")
    axis.set_xticks(positions, ACTION_NAMES)
    axis.set(title="Test action error", ylabel="Mean absolute error")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_prediction_scatter(samples: EvaluationSamples, path: Path) -> None:
    if not samples.predicted_points:
        _write_empty_plot(path, "No prediction samples were collected")
        return
    predicted = np.concatenate(samples.predicted_points, axis=0)
    target = np.concatenate(samples.target_points, axis=0)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ranges = ((0, 1), (0, 1), (-1, 1))
    for index, (axis, name, limits) in enumerate(zip(axes, ACTION_NAMES, ranges, strict=True)):
        axis.scatter(target[:, index], predicted[:, index], s=8, alpha=0.25, color=COLORS[index])
        axis.plot(limits, limits, color="#222222", linewidth=1)
        axis.set(title=name, xlabel="Target", ylabel="Prediction", xlim=limits, ylim=limits)
        axis.grid(alpha=0.2)
    figure.suptitle("First action in each predicted chunk")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_sample_chunks(samples: EvaluationSamples, path: Path) -> None:
    if not samples.predicted_chunks:
        _write_empty_plot(path, "No action chunks were collected")
        return
    count = min(4, len(samples.predicted_chunks))
    figure, axes = plt.subplots(count, 3, figsize=(14, 3.1 * count), squeeze=False)
    for row in range(count):
        predicted = samples.predicted_chunks[row]
        target = samples.target_chunks[row]
        horizon = np.arange(len(predicted)) / 10.0
        for column, name in enumerate(ACTION_NAMES):
            axes[row, column].plot(horizon, target[:, column], label="target", color="#222222")
            axes[row, column].plot(horizon, predicted[:, column], label="prediction", color=COLORS[column])
            axes[row, column].set(title=f"Sample {row + 1}: {name}", xlabel="Seconds")
            axes[row, column].grid(alpha=0.2)
            if row == 0 and column == 0:
                axes[row, column].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_empty_plot(path: Path, message: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
