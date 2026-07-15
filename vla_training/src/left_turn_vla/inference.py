from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from left_turn_vla.constants import ACTION_KEY, CAMERA_KEY, IMAGE_SIZE, LEFT_TURN_INSTRUCTION, STATE_KEY, STATE_NAMES

SUPPORTED_ACTION_DIMS = (2, 3)


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SmolVLADriver:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        action_steps: int | None = 1,
    ):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.model_path = str(model_path)
        self.device = resolve_device(device)
        config = PreTrainedConfig.from_pretrained(self.model_path, cache_dir=cache_dir)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"Expected a SmolVLA checkpoint, got policy type {config.type!r}")
        if action_steps is not None:
            if not 1 <= action_steps <= int(config.chunk_size):
                raise ValueError(f"action_steps must be between 1 and chunk_size ({config.chunk_size})")
            config.n_action_steps = action_steps
        config.device = self.device
        self.policy = SmolVLAPolicy.from_pretrained(
            self.model_path,
            config=config,
            cache_dir=cache_dir,
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            self.model_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.policy.eval()
        self._torch = torch
        self._lock = threading.Lock()

        action_feature = self.policy.config.output_features.get("action")
        if action_feature is None or not action_feature.shape:
            raise ValueError("SmolVLA checkpoint does not declare an action feature")
        configured_action_dim = int(action_feature.shape[0])
        normalized_action_dim = _normalizer_feature_dim(self.postprocessor, ACTION_KEY)
        if normalized_action_dim is None:
            normalized_action_dim = _normalizer_feature_dim(self.preprocessor, ACTION_KEY)
        if normalized_action_dim is not None and normalized_action_dim != configured_action_dim:
            raise ValueError(
                "Checkpoint action metadata does not match its learned normalizer: "
                f"config={configured_action_dim}, normalizer={normalized_action_dim}"
            )
        if configured_action_dim not in SUPPORTED_ACTION_DIMS:
            raise ValueError(
                f"Unsupported checkpoint action dimension {configured_action_dim}; expected 2 or 3"
            )
        self._action_dim = configured_action_dim

        normalized_state_dim = _normalizer_feature_dim(self.preprocessor, STATE_KEY)
        if normalized_state_dim is None:
            raise ValueError("Checkpoint does not contain learned observation.state normalization statistics")
        if normalized_state_dim != len(STATE_NAMES):
            raise ValueError(
                "Checkpoint state normalizer is incompatible with the simulator: "
                f"checkpoint={normalized_state_dim}, simulator={len(STATE_NAMES)}"
            )
        self._state_dim = normalized_state_dim

    @property
    def action_steps(self) -> int:
        return int(self.policy.config.n_action_steps)

    @property
    def chunk_size(self) -> int:
        return int(self.policy.config.chunk_size)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def image_size(self) -> int:
        return IMAGE_SIZE

    def reset(self) -> None:
        with self._lock:
            self.policy.reset()

    def predict(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        if not self.supports_instruction(instruction):
            raise ValueError("This checkpoint only supports the protected left-turn instruction")
        if image.shape != (self.image_size, self.image_size, 3):
            raise ValueError(
                f"Expected a {self.image_size} x {self.image_size} RGB image, got {image.shape}"
            )
        state_values = np.asarray(state, dtype=np.float32)
        if state_values.shape != (self.state_dim,) or not np.isfinite(state_values).all():
            raise ValueError(f"state must contain {self.state_dim} finite values")
        frame = {
            CAMERA_KEY: self._torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
            .permute(2, 0, 1)
            .to(dtype=self._torch.float32)
            / 255.0,
            STATE_KEY: self._torch.from_numpy(state_values.copy()),
            "task": LEFT_TURN_INSTRUCTION,
        }
        return self.predict_frame(frame)

    def predict_frame(self, frame: dict[str, Any]) -> np.ndarray:
        observation = {
            key: value
            for key, value in frame.items()
            if key.startswith("observation.") or key in {"task", "task_index"}
        }
        observation["task"] = LEFT_TURN_INSTRUCTION
        with self._lock, self._torch.inference_mode():
            batch = self.preprocessor(observation)
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        values = action.detach().to("cpu", dtype=self._torch.float32).numpy()
        if values.ndim == 2:
            values = values[0]
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.action_dim,) or not np.isfinite(values).all():
            raise ValueError(
                f"SmolVLA returned an invalid action with shape {values.shape}; "
                f"expected ({self.action_dim},)"
            )
        return values

    @staticmethod
    def supports_instruction(instruction: str) -> bool:
        def normalize(value: str) -> str:
            return " ".join(value.strip().lower().split())

        return normalize(instruction) == normalize(LEFT_TURN_INSTRUCTION)


def _normalizer_feature_dim(pipeline: Any, feature_key: str) -> int | None:
    """Read the learned tensor shape instead of stale base-policy feature metadata."""

    for step in getattr(pipeline, "steps", ()):  # LeRobot processor pipelines expose ordered steps.
        feature_stats = getattr(step, "_tensor_stats", {}).get(feature_key)
        if not feature_stats:
            continue
        for statistic in ("mean", "std", "min", "max", "q50"):
            tensor = feature_stats.get(statistic)
            shape = getattr(tensor, "shape", None)
            if shape is not None and len(shape) == 1 and int(shape[0]) > 0:
                return int(shape[0])
    return None


__all__ = ["SmolVLADriver", "resolve_device"]
