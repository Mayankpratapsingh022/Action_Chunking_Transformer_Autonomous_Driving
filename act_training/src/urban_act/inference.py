from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor
from torchvision.transforms.functional import resize
from transformers import AutoConfig, AutoModel, AutoTokenizer

from urban_act.model import LanguageConditionedACT, ModelConfig


class ACTPolicy:
    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        root = Path(artifact_dir)
        raw_config = json.loads((root / "config.json").read_text())
        raw_config.pop("architectures", None)
        raw_config.pop("model_type", None)
        self.config = ModelConfig.from_dict(raw_config)
        self.device = _select_device(device)
        cache_root = Path(cache_dir) if cache_dir is not None else root.parent / ".cache" / "huggingface"
        cache_root.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
        try:
            text_config = AutoConfig.from_pretrained(
                self.config.text_model_name,
                cache_dir=cache_root,
                local_files_only=True,
            )
        except OSError:
            text_config = AutoConfig.from_pretrained(self.config.text_model_name, cache_dir=cache_root)
        text_encoder = AutoModel.from_config(text_config)
        runtime_config = replace(self.config, pretrained_vision=False)
        self.model = LanguageConditionedACT(runtime_config, text_encoder=text_encoder)
        self.model.load_state_dict(load_file(root / "model.safetensors", device="cpu"))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_chunk(self, image: np.ndarray | Tensor, state: np.ndarray | Tensor, instruction: str) -> np.ndarray:
        image_tensor = torch.as_tensor(image)
        if image_tensor.ndim != 3:
            raise ValueError("image must be HWC or CHW")
        if image_tensor.shape[-1] == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = resize(image_tensor, [self.config.image_size, self.config.image_size], antialias=True)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        if state_tensor.shape != (self.config.state_dim,):
            raise ValueError(f"state must have shape ({self.config.state_dim},)")
        encoded = self.tokenizer(
            [instruction],
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with _autocast(self.device):
            output = self.model(
                image_tensor.unsqueeze(0).to(self.device),
                state_tensor.unsqueeze(0).to(self.device),
                encoded["input_ids"],
                encoded["attention_mask"],
            )
        return output["actions"][0].float().cpu().numpy()


class RecedingHorizonController:
    """Execute a few actions from each chunk, then ask the policy to replan."""

    def __init__(self, policy: ACTPolicy, *, replan_interval: int = 3) -> None:
        if replan_interval < 1:
            raise ValueError("replan_interval must be positive")
        self.policy = policy
        self.replan_interval = replan_interval
        self._chunk: np.ndarray | None = None
        self._cursor = 0

    def act(self, image: np.ndarray | Tensor, state: np.ndarray | Tensor, instruction: str) -> np.ndarray:
        if self._chunk is None or self._cursor >= min(self.replan_interval, len(self._chunk)):
            self._chunk = self.policy.predict_chunk(image, state, instruction)
            self._cursor = 0
        action = self._chunk[self._cursor]
        self._cursor += 1
        return action

    def reset(self) -> None:
        self._chunk = None
        self._cursor = 0


def _autocast(device: torch.device) -> Any:
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else _NoOpContext()


def _select_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _NoOpContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: Any) -> None:
        return None
