from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch import Tensor

ACTION_NAMES = ("throttle", "brake", "steering")


@dataclass(slots=True)
class EvaluationSamples:
    predicted_points: list[np.ndarray] = field(default_factory=list)
    target_points: list[np.ndarray] = field(default_factory=list)
    predicted_chunks: list[np.ndarray] = field(default_factory=list)
    target_chunks: list[np.ndarray] = field(default_factory=list)


class ActionMetricAccumulator:
    def __init__(self, *, max_points: int = 2_048, max_chunks: int = 8) -> None:
        self.absolute_sum = np.zeros(3, dtype=np.float64)
        self.square_sum = np.zeros(3, dtype=np.float64)
        self.valid_steps = 0
        self.brake_correct = 0
        self.brake_count = 0
        self.direction_correct = 0
        self.direction_count = 0
        self.overlap_count = 0
        self.intent_absolute: dict[str, float] = defaultdict(float)
        self.intent_count: dict[str, int] = defaultdict(int)
        self.max_points = max_points
        self.max_chunks = max_chunks
        self.samples = EvaluationSamples()

    def update(self, predicted: Tensor, target: Tensor, mask: Tensor, task_ids: list[str]) -> None:
        predicted_np = predicted.detach().float().cpu().numpy()
        target_np = target.detach().float().cpu().numpy()
        mask_np = mask.detach().bool().cpu().numpy()
        errors = predicted_np - target_np

        valid_errors = errors[mask_np]
        self.absolute_sum += np.abs(valid_errors).sum(axis=0)
        self.square_sum += np.square(valid_errors).sum(axis=0)
        self.valid_steps += int(mask_np.sum())

        valid_predictions = predicted_np[mask_np]
        valid_targets = target_np[mask_np]
        target_braking = valid_targets[:, 1] > 0.2
        predicted_braking = valid_predictions[:, 1] > 0.2
        self.brake_correct += int((target_braking == predicted_braking).sum())
        self.brake_count += len(valid_targets)

        meaningful_turn = np.abs(valid_targets[:, 2]) > 0.15
        self.direction_correct += int(
            (np.sign(valid_predictions[meaningful_turn, 2]) == np.sign(valid_targets[meaningful_turn, 2])).sum()
        )
        self.direction_count += int(meaningful_turn.sum())
        self.overlap_count += int(((valid_predictions[:, 0] > 0.2) & (valid_predictions[:, 1] > 0.2)).sum())

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
        return {
            "mean_action_mae": float(mae.mean()),
            "action_mae": dict(zip(ACTION_NAMES, map(float, mae), strict=True)),
            "action_rmse": dict(zip(ACTION_NAMES, map(float, rmse), strict=True)),
            "brake_accuracy": self.brake_correct / max(self.brake_count, 1),
            "steering_direction_accuracy": self.direction_correct / max(self.direction_count, 1),
            "throttle_brake_overlap_rate": self.overlap_count / denominator,
            "per_intent_mae": {
                task: self.intent_absolute[task] / max(self.intent_count[task], 1)
                for task in sorted(self.intent_absolute)
            },
            "valid_action_steps": self.valid_steps,
        }
