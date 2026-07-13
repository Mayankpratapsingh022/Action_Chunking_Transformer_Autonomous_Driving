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
    run_name: str = "act-driving-v2"
    artifact_root: str = "/artifacts/runs"
    cache_dir: str = "/cache/huggingface"
    seed: int = 42
    policy_version: str = "v2"
    fps: int = 10
    image_size: int = 128
    state_dim: int = 4
    action_dim: int = 3
    chunk_size: int = 20
    train_stride: int = 2
    eval_stride: int = 4
    shuffle_buffer: int = 96
    batch_size: int = 64
    num_workers: int = 4
    max_steps: int = 10_000
    warmup_steps: int = 500
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    activity_threshold: float = 0.05
    activity_probability_threshold: float = 0.5
    steering_active_threshold: float = 0.15
    activity_loss_weight: float = 1.0
    magnitude_loss_weight: float = 1.0
    steering_loss_weight: float = 2.0
    steering_active_weight: float = 3.0
    overlap_loss_weight: float = 0.25
    positive_weight_cap: float = 12.0
    startup_window_weight: float = 6.0
    throttle_window_weight: float = 2.0
    brake_window_weight: float = 4.0
    turn_window_weight: float = 2.5
    recovery_window_weight: float = 2.0
    kl_weight: float = 0.1
    kl_warmup_steps: int = 2_000
    min_startup_throttle_recall: float = 0.8
    min_throttle_recall: float = 0.55
    min_brake_recall: float = 0.55
    min_steering_direction_accuracy: float = 0.65
    min_zero_baseline_improvement: float = 0.05
    max_throttle_brake_overlap_rate: float = 0.01
    require_quality_gates: bool = True
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
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: int = 500
    eval_batches: int = 100
    test_batches: int = 250
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
            raise ValueError("Urban VLA expects state_dim=4 and action_dim=3")
        if self.policy_version not in {"v1", "v2"}:
            raise ValueError("policy_version must be v1 or v2")
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
        if (
            not 0 < self.activity_threshold < 1
            or not 0 < self.activity_probability_threshold < 1
            or not 0 < self.steering_active_threshold < 1
        ):
            raise ValueError("action activity thresholds must be between zero and one")
        positive_values = (
            self.activity_loss_weight,
            self.magnitude_loss_weight,
            self.steering_loss_weight,
            self.steering_active_weight,
            self.overlap_loss_weight,
            self.positive_weight_cap,
            self.startup_window_weight,
            self.throttle_window_weight,
            self.brake_window_weight,
            self.turn_window_weight,
            self.recovery_window_weight,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("loss and critical-window weights must be positive")
        if self.kl_weight < 0 or self.kl_warmup_steps < 0:
            raise ValueError("KL settings must be non-negative")
        quality_thresholds = (
            self.min_startup_throttle_recall,
            self.min_throttle_recall,
            self.min_brake_recall,
            self.min_steering_direction_accuracy,
            self.min_zero_baseline_improvement,
            self.max_throttle_brake_overlap_rate,
        )
        if any(not 0 <= value <= 1 for value in quality_thresholds):
            raise ValueError("quality gate thresholds must be between zero and one")
        for interval in (self.log_interval, self.eval_interval, self.checkpoint_interval):
            if interval < 1:
                raise ValueError("logging, evaluation, and checkpoint intervals must be positive")
