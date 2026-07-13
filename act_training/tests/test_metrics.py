from __future__ import annotations

import json

import torch

from urban_act.metrics import ActionMetricAccumulator


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.8, 0.0, 0.4], [0.0, 0.9, -0.4]],
            [[0.7, 0.0, -0.3], [0.0, 0.8, 0.3], [0.0, 0.0, 0.0]],
        ]
    )
    mask = torch.ones(2, 3, dtype=torch.bool)
    state = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    return target, mask, state


def test_all_zero_policy_is_detected_as_collapsed() -> None:
    target, mask, state = _batch()
    accumulator = ActionMetricAccumulator()
    accumulator.update(
        torch.zeros_like(target),
        target,
        mask,
        ["go", "yield"],
        states=state,
    )

    metrics = accumulator.compute()

    assert metrics["collapse_detected"] is True
    assert metrics["quality_gates"]["all_passed"] is False
    assert metrics["activity"]["throttle"]["recall"] == 0.0
    assert metrics["activity"]["brake"]["recall"] == 0.0
    assert metrics["startup_throttle"]["recall"] == 0.0
    json.dumps(metrics)


def test_correct_policy_passes_activity_and_baseline_gates() -> None:
    target, mask, state = _batch()
    accumulator = ActionMetricAccumulator()
    accumulator.update(target, target, mask, ["go", "yield"], states=state)

    metrics = accumulator.compute()

    assert metrics["collapse_detected"] is False
    assert metrics["quality_gates"]["all_passed"] is True
    assert metrics["selection_score"] == 0.0
    assert metrics["zero_baseline_improvement"]["throttle"] == 1.0
    assert metrics["zero_baseline_improvement"]["brake"] == 1.0


def test_simultaneous_throttle_and_brake_fails_the_overlap_gate() -> None:
    target, mask, state = _batch()
    predicted = target.clone()
    predicted[..., 0] = 0.8
    predicted[..., 1] = 0.8
    accumulator = ActionMetricAccumulator()
    accumulator.update(predicted, target, mask, ["go", "yield"], states=state)

    metrics = accumulator.compute()

    assert metrics["throttle_brake_overlap_rate"] == 1.0
    assert metrics["quality_gates"]["throttle_brake_overlap_rate"] is False
    assert metrics["quality_gates"]["all_passed"] is False
