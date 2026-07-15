import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inference_server import (
    action_for_simulator,
    create_app,
    decode_image_data_url,
    parse_args,
    parse_prediction_request,
    sanitize_action,
    state_for_model,
)
from left_turn_vla.constants import LEFT_TURN_INSTRUCTION


def _data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), (0, 0, 0)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_prediction_request_contract() -> None:
    image, state, instruction = parse_prediction_request(
        {"type": "predict", "image": _data_url(), "state": [0.0, 0.0, 0.0, 0.0], "instruction": LEFT_TURN_INSTRUCTION}
    )
    assert image.shape == (128, 128, 3)
    np.testing.assert_array_equal(state, np.zeros(4, dtype=np.float32))
    assert instruction == LEFT_TURN_INSTRUCTION


def test_invalid_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="base64"):
        decode_image_data_url("data:image/png;base64,not-valid")


def test_action_sanitizer_bounds_and_excludes_overlap() -> None:
    action = sanitize_action(np.asarray([1.4, 0.4, -2.0], dtype=np.float32))
    assert action == {"throttle": 1.0, "brake": 0.0, "steer": -1.0}


def test_target_action_sanitizer_uses_negative_steering_for_left() -> None:
    action = sanitize_action(np.asarray([30.0, -0.7], dtype=np.float32))
    assert action == {"target_speed_mps": 24.0, "target_steer": pytest.approx(-0.7), "mode": "turn_left"}


def test_legacy_checkpoint_steering_is_adapted_to_the_current_simulator() -> None:
    model_state = state_for_model(np.asarray([12.0, -0.4, 0.7, 0.0], dtype=np.float32), 3)
    simulator_action = action_for_simulator(np.asarray([0.8, 0.0, 0.6], dtype=np.float32), 3)
    np.testing.assert_allclose(model_state, [12.0, 0.4, 0.7, 0.0])
    np.testing.assert_allclose(simulator_action, [0.8, 0.0, -0.6])


def test_target_checkpoint_keeps_the_current_steering_convention() -> None:
    state = np.asarray([12.0, -0.4, 8.0, -0.6], dtype=np.float32)
    action = np.asarray([8.0, -0.6], dtype=np.float32)
    np.testing.assert_array_equal(state_for_model(state, 2), state)
    np.testing.assert_array_equal(action_for_simulator(action, 2), action)


class MockDriver:
    device = "cpu"
    image_size = 128
    state_dim = 4
    action_dim = 3
    chunk_size = 20
    action_steps = 1

    def predict(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        return np.asarray([0.5, 0.0, 0.25], dtype=np.float32)

    def reset(self) -> None:
        return None

    def supports_instruction(self, instruction: str) -> bool:
        return instruction == LEFT_TURN_INSTRUCTION


def test_health_reports_the_verified_legacy_checkpoint_contract() -> None:
    response = TestClient(create_app(MockDriver())).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "policy": "smolvla",
        "task": "protected-left-turn",
        "instruction": LEFT_TURN_INSTRUCTION,
        "device": "cpu",
        "image_size": 128,
        "state_dim": 4,
        "state_schema": ["speed_mps", "steering", "previous_throttle", "previous_brake"],
        "action_dim": 3,
        "action_space": "legacy_control",
        "action_schema": ["throttle", "brake", "steering"],
        "chunk_size": 20,
        "action_steps": 1,
    }


def test_action_steps_default_to_closed_loop_replanning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLA_ACTION_STEPS", raising=False)
    assert parse_args([]).action_steps == 1
    assert parse_args(["--action-steps", "3"]).action_steps == 3
