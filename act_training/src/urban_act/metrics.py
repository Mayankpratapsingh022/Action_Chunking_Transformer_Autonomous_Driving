from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch import Tensor

ACTION_NAMES = ("throttle", "brake", "steering")
LONGITUDINAL_NAMES = ACTION_NAMES[:2]


@dataclass(slots=True)
class EvaluationSamples:
    predicted_points: list[np.ndarray] = field(default_factory=list)
    target_points: list[np.ndarray] = field(default_factory=list)
    predicted_chunks: list[np.ndarray] = field(default_factory=list)
    target_chunks: list[np.ndarray] = field(default_factory=list)


class ActionMetricAccumulator:
    def __init__(
        self,
        *,
        max_points: int = 2_048,
        max_chunks: int = 8,
        activity_threshold: float = 0.05,
        steering_active_threshold: float = 0.15,
        min_startup_throttle_recall: float = 0.8,
        min_throttle_recall: float = 0.55,
        min_brake_recall: float = 0.55,
        min_steering_direction_accuracy: float = 0.65,
        min_zero_baseline_improvement: float = 0.05,
        max_throttle_brake_overlap_rate: float = 0.01,
    ) -> None:
        self.absolute_sum = np.zeros(3, dtype=np.float64)
        self.square_sum = np.zeros(3, dtype=np.float64)
        self.zero_baseline_sum = np.zeros(3, dtype=np.float64)
        self.active_absolute_sum = np.zeros(3, dtype=np.float64)
        self.active_count = np.zeros(3, dtype=np.int64)
        self.valid_steps = 0
        self.activity_confusion = np.zeros((2, 4), dtype=np.int64)  # tp, fp, tn, fn
        self.direction_correct = 0
        self.direction_count = 0
        self.overlap_count = 0
        self.startup_correct = 0
        self.startup_count = 0
        self.startup_predicted_sum = 0.0
        self.startup_target_sum = 0.0
        self.intent_absolute: dict[str, float] = defaultdict(float)
        self.intent_count: dict[str, int] = defaultdict(int)
        self.max_points = max_points
        self.max_chunks = max_chunks
        self.activity_threshold = activity_threshold
        self.steering_active_threshold = steering_active_threshold
        self.min_startup_throttle_recall = min_startup_throttle_recall
        self.min_throttle_recall = min_throttle_recall
        self.min_brake_recall = min_brake_recall
        self.min_steering_direction_accuracy = min_steering_direction_accuracy
        self.min_zero_baseline_improvement = min_zero_baseline_improvement
        self.max_throttle_brake_overlap_rate = max_throttle_brake_overlap_rate
        self.samples = EvaluationSamples()

    def update(
        self,
        predicted: Tensor,
        target: Tensor,
        mask: Tensor,
        task_ids: list[str],
        *,
        states: Tensor | None = None,
    ) -> None:
        predicted_np = predicted.detach().float().cpu().numpy()
        target_np = target.detach().float().cpu().numpy()
        mask_np = mask.detach().bool().cpu().numpy()
        errors = predicted_np - target_np

        valid_errors = errors[mask_np]
        valid_predictions = predicted_np[mask_np]
        valid_targets = target_np[mask_np]
        self.absolute_sum += np.abs(valid_errors).sum(axis=0)
        self.square_sum += np.square(valid_errors).sum(axis=0)
        self.zero_baseline_sum += np.abs(valid_targets).sum(axis=0)
        self.valid_steps += int(mask_np.sum())

        longitudinal_targets = valid_targets[:, :2] > self.activity_threshold
        longitudinal_predictions = valid_predictions[:, :2] > self.activity_threshold
        for index in range(2):
            target_active = longitudinal_targets[:, index]
            predicted_active = longitudinal_predictions[:, index]
            self.activity_confusion[index] += (
                int((target_active & predicted_active).sum()),
                int((~target_active & predicted_active).sum()),
                int((~target_active & ~predicted_active).sum()),
                int((target_active & ~predicted_active).sum()),
            )
            self.active_absolute_sum[index] += np.abs(valid_errors[target_active, index]).sum()
            self.active_count[index] += int(target_active.sum())

        meaningful_turn = np.abs(valid_targets[:, 2]) > self.steering_active_threshold
        self.active_absolute_sum[2] += np.abs(valid_errors[meaningful_turn, 2]).sum()
        self.active_count[2] += int(meaningful_turn.sum())
        self.direction_correct += int(
            (np.sign(valid_predictions[meaningful_turn, 2]) == np.sign(valid_targets[meaningful_turn, 2])).sum()
        )
        self.direction_count += int(meaningful_turn.sum())
        self.overlap_count += int(
            (
                (valid_predictions[:, 0] > self.activity_threshold)
                & (valid_predictions[:, 1] > self.activity_threshold)
            ).sum()
        )

        if states is not None and predicted_np.shape[1] > 0:
            states_np = states.detach().float().cpu().numpy()
            startup = mask_np[:, 0] & (np.abs(states_np[:, 0]) < 0.5) & (target_np[:, 0, 0] > self.activity_threshold)
            startup_predictions = predicted_np[startup, 0, 0]
            startup_targets = target_np[startup, 0, 0]
            self.startup_correct += int((startup_predictions > self.activity_threshold).sum())
            self.startup_count += int(startup.sum())
            self.startup_predicted_sum += float(startup_predictions.sum())
            self.startup_target_sum += float(startup_targets.sum())

        for batch_index, task_id in enumerate(task_ids):
            episode_mask = mask_np[batch_index]
            self.intent_absolute[task_id] += float(np.abs(errors[batch_index][episode_mask]).sum())
            self.intent_count[task_id] += int(episode_mask.sum()) * 3

        remaining_points = self.max_points - sum(len(item) for item in self.samples.predicted_points)
        if remaining_points > 0:
            first_step_mask = mask_np[:, 0]
            self.samples.predicted_points.append(predicted_np[first_step_mask, 0][:remaining_points])
            self.samples.target_points.append(target_np[first_step_mask, 0][:remaining_points])
        remaining_chunks = self.max_chunks - len(self.samples.predicted_chunks)
        if remaining_chunks > 0:
            for batch_index in range(min(remaining_chunks, predicted_np.shape[0])):
                valid_length = int(mask_np[batch_index].sum())
                self.samples.predicted_chunks.append(predicted_np[batch_index, :valid_length])
                self.samples.target_chunks.append(target_np[batch_index, :valid_length])

    def compute(self) -> dict[str, Any]:
        denominator = max(self.valid_steps, 1)
        mae = self.absolute_sum / denominator
        rmse = np.sqrt(self.square_sum / denominator)
        baseline_mae = self.zero_baseline_sum / denominator
        active_mae = self.active_absolute_sum / np.maximum(self.active_count, 1)
        baseline_improvement = np.divide(
            baseline_mae - mae,
            baseline_mae,
            out=np.zeros_like(mae),
            where=baseline_mae > 1e-12,
        )
        activity = {name: self._activity_metrics(index) for index, name in enumerate(LONGITUDINAL_NAMES)}
        steering_direction_accuracy = self.direction_correct / max(self.direction_count, 1)
        startup_recall = self.startup_correct / max(self.startup_count, 1)
        overlap_rate = self.overlap_count / denominator
        quality_gates = {
            "startup_throttle_recall": bool(startup_recall >= self.min_startup_throttle_recall),
            "throttle_recall": bool(activity["throttle"]["recall"] >= self.min_throttle_recall),
            "brake_recall": bool(activity["brake"]["recall"] >= self.min_brake_recall),
            "steering_direction_accuracy": bool(steering_direction_accuracy >= self.min_steering_direction_accuracy),
            "throttle_zero_baseline_improvement": bool(baseline_improvement[0] >= self.min_zero_baseline_improvement),
            "brake_zero_baseline_improvement": bool(baseline_improvement[1] >= self.min_zero_baseline_improvement),
            "throttle_brake_overlap_rate": bool(overlap_rate <= self.max_throttle_brake_overlap_rate),
        }
        quality_gates["all_passed"] = all(quality_gates.values())

        classification_penalty = np.mean(
            (
                1.0 - activity["throttle"]["f1"],
                1.0 - activity["brake"]["f1"],
                1.0 - steering_direction_accuracy,
                1.0 - startup_recall,
            )
        )
        selection_score = float(active_mae.mean() + 0.25 * classification_penalty + overlap_rate)
        collapse_detected = bool(
            startup_recall < 0.1 or activity["throttle"]["recall"] < 0.1 or activity["brake"]["recall"] < 0.1
        )

        return {
            "selection_score": selection_score,
            "mean_action_mae": float(mae.mean()),
            "action_mae": dict(zip(ACTION_NAMES, map(float, mae), strict=True)),
            "action_rmse": dict(zip(ACTION_NAMES, map(float, rmse), strict=True)),
            "active_action_mae": dict(zip(ACTION_NAMES, map(float, active_mae), strict=True)),
            "zero_baseline_mae": dict(zip(ACTION_NAMES, map(float, baseline_mae), strict=True)),
            "zero_baseline_improvement": dict(zip(ACTION_NAMES, map(float, baseline_improvement), strict=True)),
            "activity": activity,
            "brake_accuracy": activity["brake"]["accuracy"],
            "steering_direction_accuracy": steering_direction_accuracy,
            "throttle_brake_overlap_rate": overlap_rate,
            "startup_throttle": {
                "recall": startup_recall,
                "predicted_mean": self.startup_predicted_sum / max(self.startup_count, 1),
                "target_mean": self.startup_target_sum / max(self.startup_count, 1),
                "count": self.startup_count,
            },
            "quality_gates": quality_gates,
            "collapse_detected": collapse_detected,
            "per_intent_mae": {
                task: self.intent_absolute[task] / max(self.intent_count[task], 1)
                for task in sorted(self.intent_absolute)
            },
            "valid_action_steps": self.valid_steps,
        }

    def _activity_metrics(self, index: int) -> dict[str, Any]:
        tp, fp, tn, fn = map(int, self.activity_confusion[index])
        count = tp + fp + tn + fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "accuracy": (tp + tn) / max(count, 1),
            "target_positive_rate": (tp + fn) / max(count, 1),
            "predicted_positive_rate": (tp + fp) / max(count, 1),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        }
