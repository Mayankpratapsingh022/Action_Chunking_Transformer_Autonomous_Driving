from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inference_server import create_app, decode_image_data_url, sanitize_action


class FakePolicy:
    config = SimpleNamespace(image_size=128, state_dim=4, action_dim=3, chunk_size=3)
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def predict_chunk(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        assert image.shape == (16, 16, 3)
        np.testing.assert_allclose(state, [7.5, 0.1, 0.6, 0.0])
        assert instruction == "Turn left at the intersection."
        self.calls += 1
        return np.asarray(
            (
                (0.7, 0.02, -0.2),
                (0.6, 0.03, -0.1),
                (0.0, 0.8, 0.04),
            ),
            dtype=np.float32,
        )


def image_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=(20, 40, 60)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def request(request_id: int) -> dict:
    return {
        "type": "predict",
        "request_id": request_id,
        "image": image_data_url(),
        "instruction": "Turn left at the intersection.",
        "state": [7.5, 0.1, 0.6, 0.0],
    }


def test_health_and_receding_horizon_websocket() -> None:
    policy = FakePolicy()
    client = TestClient(create_app(policy, replan_interval=3))
    assert client.get("/health").json() == {
        "status": "ready",
        "device": "cpu",
        "image_size": 128,
        "state_dim": 4,
        "action_dim": 3,
        "chunk_size": 3,
        "replan_interval": 3,
    }

    with client.websocket_connect("/ws") as websocket:
        responses = []
        for request_id in range(1, 4):
            websocket.send_json(request(request_id))
            responses.append(websocket.receive_json())
        assert policy.calls == 1
        assert responses[0]["action"] == pytest.approx({"throttle": 0.7, "brake": 0.0, "steer": -0.2})
        assert responses[2]["action"] == pytest.approx({"throttle": 0.0, "brake": 0.8, "steer": 0.04})

        websocket.send_json({"type": "reset"})
        assert websocket.receive_json() == {"type": "reset_ack"}
        websocket.send_json(request(4))
        websocket.receive_json()
        assert policy.calls == 2


def test_invalid_request_returns_protocol_error() -> None:
    client = TestClient(create_app(FakePolicy()))
    with client.websocket_connect("/ws") as websocket:
        payload = request(1)
        payload["state"] = [1, 2]
        websocket.send_json(payload)
        response = websocket.receive_json()
    assert response == {
        "type": "error",
        "request_id": 1,
        "error": "state must contain [speed, steering, previous throttle, previous brake]",
    }


def test_image_validation_and_action_safety() -> None:
    decoded = decode_image_data_url(image_data_url())
    assert decoded.shape == (16, 16, 3)
    assert sanitize_action(np.asarray((0.8, 0.3, 1.4))) == pytest.approx(
        {"throttle": 0.8, "brake": 0.0, "steer": 1.0}
    )
