from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def training_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    global_step: int,
    epoch: int,
    elapsed_seconds: float,
    best_validation_score: float,
    best_step: int,
    best_validation_metrics: dict[str, Any] | None,
    history: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "elapsed_seconds": elapsed_seconds,
        "best_validation_score": best_validation_score,
        "best_step": best_step,
        "best_validation_metrics": best_validation_metrics,
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])


def save_inference_weights(model: nn.Module, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    weights = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(weights, destination, metadata={"format": "pt"})
