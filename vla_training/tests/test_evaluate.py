import numpy as np
import pytest

from evaluate import calculate_metrics, quality_gates


def test_target_action_metrics_accept_two_value_smolvla_predictions() -> None:
    targets = np.asarray(
        [
            [12.0, 0.0],
            [8.0, -0.8],
            [8.0, -0.5],
        ],
        dtype=np.float32,
    )
    predictions = np.asarray(
        [
            [11.9, 0.0],
            [8.1, -0.75],
            [8.0, -0.45],
        ],
        dtype=np.float32,
    )

    metrics = calculate_metrics(targets, predictions)

    assert set(metrics["action_mae"]) == {"target_speed_mps", "target_steering"}
    assert metrics["steering_direction_accuracy"] == 1.0
    assert quality_gates(metrics)["all_open_loop_passed"] is True


def test_target_action_metrics_reject_legacy_three_value_predictions() -> None:
    with pytest.raises(ValueError, match=r"shape \[frames, 2\]"):
        calculate_metrics(np.zeros((2, 3)), np.zeros((2, 3)))
