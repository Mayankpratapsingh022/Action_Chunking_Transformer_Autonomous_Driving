from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    dataset_repo: str = "Mayank022/urban-vla-left-turn-human"
    model_repo: str = "Mayank022/urban-vla-left-turn-smolvla"
    base_model: str = "lerobot/smolvla_base"
    run_name: str = "smolvla-left-turn-v1"
    seed: int = 42
    fps: int = 10
    steps: int = 20_000
    batch_size: int = 32
    num_workers: int = 8
    chunk_size: int = 20
    action_steps: int = 3
    eval_split: float = 0.1
    eval_interval: int = 1_000
    save_interval: int = 2_000
    log_interval: int = 20
    warmup_steps: int = 1_000
    max_eval_samples: int = 2_000
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    image_augmentation: bool = True
    push_to_hub: bool = True
    rename_map: dict[str, str] = field(
        default_factory=lambda: {"observation.images.front": "observation.images.camera1"}
    )

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("Training configuration must be a JSON object")
        config = cls(**payload)
        config.validate()
        return config

    def with_overrides(self, **values: Any) -> TrainConfig:
        payload = asdict(self)
        payload.update({key: value for key, value in values.items() if value is not None})
        config = TrainConfig(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("dataset_repo", "model_repo", "base_model", "run_name"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if any(not source.strip() or not destination.strip() for source, destination in self.rename_map.items()):
            raise ValueError("rename_map keys and values must be non-empty")
        if self.steps < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("steps and batch_size must be positive; num_workers cannot be negative")
        if not 0 < self.eval_split < 0.5:
            raise ValueError("eval_split must be greater than 0 and less than 0.5")
        if not 1 <= self.action_steps <= self.chunk_size:
            raise ValueError("action_steps must be between 1 and chunk_size")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than steps")
        for name in ("eval_interval", "save_interval", "log_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
