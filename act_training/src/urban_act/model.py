from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel


@dataclass(slots=True)
class ModelConfig:
    image_size: int = 128
    state_dim: int = 4
    action_dim: int = 3
    chunk_size: int = 20
    d_model: int = 256
    nhead: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 6
    latent_dim: int = 32
    dropout: float = 0.1
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_hidden_size: int = 384
    vision_channels: int = 512
    freeze_text_encoder: bool = True
    pretrained_vision: bool = True
    state_mean: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    state_std: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        for name in ("state_mean", "state_std"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


class LanguageConditionedACT(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        text_encoder: nn.Module | None = None,
        vision_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.freeze_text_encoder = config.freeze_text_encoder

        if vision_encoder is None:
            weights = ResNet18_Weights.DEFAULT if config.pretrained_vision else None
            backbone = resnet18(weights=weights)
            vision_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.vision_backbone = vision_encoder

        if text_encoder is None:
            text_encoder = AutoModel.from_pretrained(config.text_model_name)
            config.text_hidden_size = int(text_encoder.config.hidden_size)
        self.text_encoder = text_encoder
        if self.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)

        self.vision_projection = nn.Conv2d(config.vision_channels, config.d_model, kernel_size=1)
        self.state_projection = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.language_projection = nn.Sequential(
            nn.Linear(config.text_hidden_size, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.latent_projection = nn.Linear(config.latent_dim, config.d_model)

        vision_side = config.image_size // 32
        self.vision_position = nn.Parameter(torch.empty(1, vision_side * vision_side, config.d_model))
        self.special_position = nn.Parameter(torch.empty(1, 3, config.d_model))
        self.action_queries = nn.Parameter(torch.empty(1, config.chunk_size, config.d_model))

        context_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.action_head = nn.Linear(config.d_model, config.action_dim)

        self.posterior_cls = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.posterior_action_projection = nn.Linear(config.action_dim, config.d_model)
        self.posterior_position = nn.Parameter(torch.empty(1, config.chunk_size + 2, config.d_model))
        posterior_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.posterior_encoder = nn.TransformerEncoder(
            posterior_layer,
            num_layers=2,
            norm=nn.LayerNorm(config.d_model),
        )
        self.posterior_head = nn.Linear(config.d_model, config.latent_dim * 2)

        self.register_buffer("state_mean", torch.tensor(config.state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(config.state_std, dtype=torch.float32))
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in (
            self.vision_position,
            self.special_position,
            self.action_queries,
            self.posterior_cls,
            self.posterior_position,
        ):
            nn.init.normal_(parameter, std=0.02)

    def train(self, mode: bool = True) -> LanguageConditionedACT:
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def forward(
        self,
        images: Tensor,
        state: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        target_actions: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_size = images.shape[0]
        normalized_state = (state - self.state_mean) / self.state_std.clamp_min(1e-6)
        state_token = self.state_projection(normalized_state).unsqueeze(1)

        image_values = images.float()
        if images.dtype == torch.uint8:
            image_values = image_values / 255.0
        image_values = (image_values - self.image_mean) / self.image_std
        vision_map = self.vision_projection(self.vision_backbone(image_values))
        vision_tokens = vision_map.flatten(2).transpose(1, 2)
        if vision_tokens.shape[1] != self.vision_position.shape[1]:
            raise ValueError(
                f"Expected {self.vision_position.shape[1]} vision tokens, got {vision_tokens.shape[1]}"
            )

        language_hidden = self._encode_language(input_ids, attention_mask)
        language_token = self.language_projection(language_hidden).unsqueeze(1)

        if target_actions is not None:
            if action_mask is None:
                action_mask = torch.ones(target_actions.shape[:2], dtype=torch.bool, device=target_actions.device)
            latent, mean, log_variance = self._sample_posterior(normalized_state, target_actions, action_mask)
        else:
            latent = torch.zeros(batch_size, self.config.latent_dim, device=state.device, dtype=state.dtype)
            mean = torch.zeros_like(latent)
            log_variance = torch.zeros_like(latent)
        latent_token = self.latent_projection(latent).unsqueeze(1)

        special_tokens = torch.cat((state_token, language_token, latent_token), dim=1)
        special_tokens = special_tokens + self.special_position
        vision_tokens = vision_tokens + self.vision_position
        memory = self.context_encoder(torch.cat((special_tokens, vision_tokens), dim=1))

        queries = self.action_queries.expand(batch_size, -1, -1)
        decoded = self.action_decoder(queries, memory)
        raw_actions = self.action_head(decoded)
        actions = torch.stack(
            (
                raw_actions[..., 0].sigmoid(),
                raw_actions[..., 1].sigmoid(),
                raw_actions[..., 2].tanh(),
            ),
            dim=-1,
        )
        return {"actions": actions, "posterior_mean": mean, "posterior_log_variance": log_variance}

    def _encode_language(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if self.freeze_text_encoder:
            with torch.no_grad():
                output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        return (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _sample_posterior(self, state: Tensor, actions: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = state.shape[0]
        cls_token = self.posterior_cls.expand(batch_size, -1, -1)
        state_token = self.state_projection(state).unsqueeze(1)
        action_tokens = self.posterior_action_projection(actions)
        tokens = torch.cat((cls_token, state_token, action_tokens), dim=1) + self.posterior_position
        prefix_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=actions.device)
        padding_mask = torch.cat((prefix_mask, ~action_mask.bool()), dim=1)
        encoded = self.posterior_encoder(tokens, src_key_padding_mask=padding_mask)
        mean, log_variance = self.posterior_head(encoded[:, 0]).chunk(2, dim=-1)
        log_variance = log_variance.clamp(-10.0, 10.0)
        standard_deviation = torch.exp(0.5 * log_variance)
        latent = mean + standard_deviation * torch.randn_like(standard_deviation)
        return latent, mean, log_variance
