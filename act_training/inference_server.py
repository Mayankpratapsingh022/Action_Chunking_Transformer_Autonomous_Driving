from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from urban_act.inference import ACTPolicy, RecedingHorizonController  # noqa: E402

DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "act-driving-v1"
MAX_IMAGE_BYTES = 2_000_000
MAX_IMAGE_SIDE = 1_024


class Policy(Protocol):
    config: Any
    device: Any

    def predict_chunk(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the trained ACT driving policy over WebSocket.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--replan-interval", type=int, default=3)
    return parser.parse_args()


def create_app(policy: Policy, *, replan_interval: int = 3) -> FastAPI:
    if replan_interval < 1:
        raise ValueError("replan_interval must be positive")

    app = FastAPI(title="ACT Driving Inference", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "device": str(policy.device),
            "image_size": int(policy.config.image_size),
            "state_dim": int(policy.config.state_dim),
            "action_dim": int(policy.config.action_dim),
            "chunk_size": int(policy.config.chunk_size),
            "replan_interval": replan_interval,
        }

    @app.websocket("/ws")
    async def inference_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        controller = RecedingHorizonController(policy, replan_interval=replan_interval)
        active_instruction: str | None = None
        try:
            while True:
                payload = await websocket.receive_json()
                message_type = payload.get("type", "predict") if isinstance(payload, dict) else None
                if message_type == "reset":
                    controller.reset()
                    active_instruction = None
                    await websocket.send_json({"type": "reset_ack"})
                    continue
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                try:
                    image, state, instruction = parse_prediction_request(payload)
                    if instruction != active_instruction:
                        controller.reset()
                        active_instruction = instruction
                    started = time.perf_counter()
                    raw_action = await asyncio.to_thread(controller.act, image, state, instruction)
                    latency_ms = (time.perf_counter() - started) * 1_000
                    action = sanitize_action(raw_action)
                    await websocket.send_json(
                        {
                            "type": "prediction",
                            "request_id": request_id,
                            "action": action,
                            "raw_action": {
                                "throttle": float(raw_action[0]),
                                "brake": float(raw_action[1]),
                                "steer": float(raw_action[2]),
                            },
                            "latency_ms": round(latency_ms, 2),
                        }
                    )
                except (TypeError, ValueError) as error:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "error": str(error),
                        }
                    )
        except WebSocketDisconnect:
            return

    return app


def parse_prediction_request(payload: Any) -> tuple[np.ndarray, np.ndarray, str]:
    if not isinstance(payload, dict) or payload.get("type", "predict") != "predict":
        raise ValueError("Expected a predict message")
    image = decode_image_data_url(payload.get("image"))
    raw_state = payload.get("state")
    if not isinstance(raw_state, list) or len(raw_state) != 4:
        raise ValueError("state must contain [speed, steering, previous throttle, previous brake]")
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
    _, encoded = value.split(",", maxsplit=1)
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
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("image data is not a supported image") from error


def sanitize_action(action: np.ndarray) -> dict[str, float]:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("model returned an invalid action")
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


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    required = ("config.json", "model.safetensors", "tokenizer")
    missing = [name for name in required if not (artifact_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing inference artifacts in {artifact_dir}: {', '.join(missing)}. "
            "Run scripts/download_hf_run.py --run-name act-driving-v1 first."
        )
    policy = ACTPolicy(
        artifact_dir,
        device=args.device,
        cache_dir=PROJECT_DIR / ".cache" / "huggingface",
    )
    print(
        f"ACT policy ready: device={policy.device}, image={policy.config.image_size}, "
        f"chunk={policy.config.chunk_size}",
        flush=True,
    )
    uvicorn.run(create_app(policy, replan_interval=args.replan_interval), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
