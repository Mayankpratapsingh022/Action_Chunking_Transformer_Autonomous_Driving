from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from left_turn_vla.constants import LEFT_TURN_INSTRUCTION  # noqa: E402
from left_turn_vla.env import load_env_file  # noqa: E402
from left_turn_vla.inference import SmolVLADriver  # noqa: E402

DEFAULT_MODEL = "Mayank022/urban-vla-left-turn-smolvla-v2"
DEFAULT_ACTION_STEPS = 1
MAX_IMAGE_BYTES = 2_000_000
MAX_IMAGE_SIDE = 1_024


class Driver(Protocol):
    device: str
    action_steps: int
    chunk_size: int
    action_dim: int
    state_dim: int
    image_size: int

    def predict(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray: ...

    def reset(self) -> None: ...

    def supports_instruction(self, instruction: str) -> bool: ...


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the fine-tuned left-turn SmolVLA policy.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("VLA_MODEL_PATH") or os.environ.get("HF_MODEL_REPO") or DEFAULT_MODEL,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--action-steps",
        type=positive_int,
        default=positive_int(os.environ.get("VLA_ACTION_STEPS", str(DEFAULT_ACTION_STEPS))),
        help="Actions to execute from each predicted chunk before replanning (default: 1)",
    )
    return parser.parse_args(argv)


def create_app(driver: Driver) -> FastAPI:
    app = FastAPI(title="SmolVLA Left-Turn Driving Inference", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "policy": "smolvla",
            "task": "protected-left-turn",
            "instruction": LEFT_TURN_INSTRUCTION,
            "device": str(driver.device),
            "image_size": int(driver.image_size),
            "state_dim": int(driver.state_dim),
            "state_schema": state_schema_for_dim(driver.action_dim),
            "action_dim": int(driver.action_dim),
            "action_space": action_space_for_dim(driver.action_dim),
            "action_schema": action_schema_for_dim(driver.action_dim),
            "chunk_size": int(driver.chunk_size),
            "action_steps": int(driver.action_steps),
        }

    @app.websocket("/ws")
    async def inference_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        driver.reset()
        await websocket.send_json(
            {
                "type": "ready",
                "image_size": int(driver.image_size),
                "state_dim": int(driver.state_dim),
                "state_schema": state_schema_for_dim(driver.action_dim),
                "action_dim": int(driver.action_dim),
                "action_space": action_space_for_dim(driver.action_dim),
                "action_schema": action_schema_for_dim(driver.action_dim),
                "action_steps": int(driver.action_steps),
            }
        )
        try:
            while True:
                payload = await websocket.receive_json()
                message_type = payload.get("type", "predict") if isinstance(payload, dict) else None
                if message_type == "reset":
                    driver.reset()
                    await websocket.send_json({"type": "reset_ack"})
                    continue
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                try:
                    image, state, instruction = parse_prediction_request(payload)
                    if not driver.supports_instruction(instruction):
                        raise ValueError("This model only supports the protected left-turn instruction")
                    started = time.perf_counter()
                    model_state = state_for_model(state, driver.action_dim)
                    raw_model_action = await asyncio.to_thread(
                        driver.predict,
                        image,
                        model_state,
                        instruction,
                    )
                    latency_ms = (time.perf_counter() - started) * 1_000
                    raw_action = action_for_simulator(raw_model_action, driver.action_dim)
                    action = sanitize_action(raw_action)
                    action_space = action_space_for_dim(len(raw_action))
                    if action_space == "target_speed_steering":
                        raw_payload = {
                            "target_speed_mps": float(raw_action[0]),
                            "target_steer": float(raw_action[1]),
                        }
                    else:
                        raw_payload = {
                            "throttle": float(raw_action[0]),
                            "brake": float(raw_action[1]),
                            "steer": float(raw_action[2]),
                        }
                    await websocket.send_json(
                        {
                            "type": "prediction",
                            "request_id": request_id,
                            "action_space": action_space,
                            "action": action,
                            "raw_action": raw_payload,
                            "latency_ms": round(latency_ms, 2),
                        }
                    )
                except (TypeError, ValueError) as error:
                    await websocket.send_json({"type": "error", "request_id": request_id, "error": str(error)})
        except WebSocketDisconnect:
            return

    return app


def parse_prediction_request(payload: Any) -> tuple[np.ndarray, np.ndarray, str]:
    if not isinstance(payload, dict) or payload.get("type", "predict") != "predict":
        raise ValueError("Expected a predict message")
    image = decode_image_data_url(payload.get("image"))
    raw_state = payload.get("state")
    if not isinstance(raw_state, list) or len(raw_state) != 4:
        raise ValueError("state must contain [speed, steering, previous action 0, previous action 1]")
    try:
        state = np.asarray(raw_state, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("state values must be numeric") from error
    if not np.isfinite(state).all():
        raise ValueError("state values must be finite")
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if len(instruction) > 512:
        raise ValueError("instruction must be at most 512 characters")
    return image, state, instruction.strip()


def decode_image_data_url(value: Any) -> np.ndarray:
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("image must be a base64 image data URL")
    encoded = value.split(",", maxsplit=1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("image contains invalid base64 data") from error
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image must be between 1 and {MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width > MAX_IMAGE_SIDE or image.height > MAX_IMAGE_SIDE:
                raise ValueError(f"image dimensions must not exceed {MAX_IMAGE_SIDE} x {MAX_IMAGE_SIDE}")
            rgb = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
            return np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("image data is not a supported image") from error


def sanitize_action(action: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(action, dtype=np.float32)
    if values.ndim != 1 or len(values) not in (2, 3) or not np.isfinite(values).all():
        raise ValueError("model returned an invalid action")
    if len(values) == 2:
        target_speed = float(np.clip(values[0], 0.0, 24.0))
        target_steer = float(np.clip(values[1], -1.0, 1.0))
        if target_speed < 0.1:
            target_speed = 0.0
        if abs(target_steer) < 0.015:
            target_steer = 0.0
        mode = (
            "stopped"
            if target_speed == 0.0
            else "turn_left"
            if target_steer < -0.12
            else "turn_right"
            if target_steer > 0.12
            else "cruise"
        )
        return {"target_speed_mps": target_speed, "target_steer": target_steer, "mode": mode}
    throttle = float(np.clip(values[0], 0.0, 1.0))
    brake = float(np.clip(values[1], 0.0, 1.0))
    steer = float(np.clip(values[2], -1.0, 1.0))
    if throttle < 0.03:
        throttle = 0.0
    if brake < 0.05:
        brake = 0.0
    if abs(steer) < 0.015:
        steer = 0.0
    if throttle > 0 and brake > 0:
        if throttle >= brake:
            brake = 0.0
        else:
            throttle = 0.0
    return {"throttle": throttle, "brake": brake, "steer": steer}


def action_space_for_dim(action_dim: int) -> str:
    if action_dim == 2:
        return "target_speed_steering"
    if action_dim == 3:
        return "legacy_control"
    raise ValueError(f"Unsupported action dimension: {action_dim}")


def state_for_model(state: np.ndarray, action_dim: int) -> np.ndarray:
    values = np.asarray(state, dtype=np.float32).copy()
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("model state must contain four finite values")
    if action_dim == 3:
        # The legacy dataset used positive-left steering; the current simulator uses negative-left.
        values[1] *= -1
    elif action_dim != 2:
        raise ValueError(f"Unsupported action dimension: {action_dim}")
    return values


def action_for_simulator(action: np.ndarray, action_dim: int) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).copy()
    if values.shape != (action_dim,) or not np.isfinite(values).all():
        raise ValueError("model returned an invalid action")
    if action_dim == 3:
        values[2] *= -1
    elif action_dim != 2:
        raise ValueError(f"Unsupported action dimension: {action_dim}")
    return values


def state_schema_for_dim(action_dim: int) -> list[str]:
    if action_dim == 2:
        return ["speed_mps", "steering", "previous_target_speed_mps", "previous_target_steering"]
    if action_dim == 3:
        return ["speed_mps", "steering", "previous_throttle", "previous_brake"]
    raise ValueError(f"Unsupported action dimension: {action_dim}")


def action_schema_for_dim(action_dim: int) -> list[str]:
    if action_dim == 2:
        return ["target_speed_mps", "target_steering"]
    if action_dim == 3:
        return ["throttle", "brake", "steering"]
    raise ValueError(f"Unsupported action dimension: {action_dim}")


def main() -> None:
    load_env_file(PROJECT_DIR / ".env")
    args = parse_args()
    driver = SmolVLADriver(
        args.model_path,
        device=args.device,
        cache_dir=PROJECT_DIR / ".cache",
        action_steps=args.action_steps,
    )
    print(
        f"SmolVLA ready: device={driver.device}, action_dim={driver.action_dim}, "
        f"chunk={driver.chunk_size}, execute={driver.action_steps}",
        flush=True,
    )
    uvicorn.run(create_app(driver), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
