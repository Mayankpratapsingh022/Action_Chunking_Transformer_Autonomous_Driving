from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    dataset_repo: str = "Mayank022/urban-vla-expert-v1"
    dataset_revision: str = "main"
    model_repo: str = "Mayank022/urban-vla-language-act"
    run_name: str = "urban-act-v1"
    artifact_root: str = "/artifacts/runs"
    cache_dir: str = "/cache/huggingface"
    seed: int = 42
    fps: int = 10
    image_size: int = 256
    state_dim: int = 4
    action_dim: int = 3
    chunk_size: int = 20
    train_stride: int = 2
    eval_stride: int = 4
    shuffle_buffer: int = 96
    batch_size: int = 32
    num_workers: int = 4
    max_steps: int = 20_000
    warmup_steps: int = 1_000
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    kl_weight: float = 10.0
    d_model: int = 256
    nhead: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 6
    latent_dim: int = 32
    dropout: float = 0.1
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    freeze_text_encoder: bool = True
    pretrained_vision: bool = True
    mixed_precision: str = "bf16"
    log_interval: int = 25
    eval_interval: int = 1_000
    checkpoint_interval: int = 1_000
    eval_batches: int = 200
    test_batches: int = 500
    resume: str = "auto"
    push_to_hub: bool = True
    hub_private: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TrainConfig:
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.run_name or "/" in self.run_name or ".." in self.run_name:
            raise ValueError("run_name must be a single safe directory name")
        if self.state_dim != 4 or self.action_dim != 3:
            raise ValueError("Urban VLA v1 expects state_dim=4 and action_dim=3")
        if self.image_size % 32:
            raise ValueError("image_size must be divisible by 32 for ResNet-18")
        if self.chunk_size < 2:
            raise ValueError("chunk_size must be at least 2")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if self.batch_size < 1 or self.max_steps < 1:
            raise ValueError("batch_size and max_steps must be positive")
        if self.train_stride < 1 or self.eval_stride < 1:
            raise ValueError("frame strides must be positive")
        if self.mixed_precision not in {"bf16", "fp16", "none"}:
            raise ValueError("mixed_precision must be bf16, fp16, or none")
        for interval in (self.log_interval, self.eval_interval, self.checkpoint_interval):
            if interval < 1:
                raise ValueError("logging, evaluation, and checkpoint intervals must be positive")

