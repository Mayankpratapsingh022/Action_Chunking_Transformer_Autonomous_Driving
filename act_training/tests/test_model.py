from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from urban_act.losses import act_loss
from urban_act.model import LanguageConditionedACT, ModelConfig


class DummyTextEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def make_model(image_size: int = 64) -> LanguageConditionedACT:
    config = ModelConfig(
        image_size=image_size,
        chunk_size=4,
        d_model=32,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        latent_dim=8,
        dropout=0.0,
        text_hidden_size=16,
        vision_channels=16,
        pretrained_vision=False,
    )
    vision_side = image_size // 32
    vision = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=1),
        nn.AdaptiveAvgPool2d((vision_side, vision_side)),
    )
    return LanguageConditionedACT(config, text_encoder=DummyTextEncoder(16), vision_encoder=vision)


def test_model_predicts_bounded_action_chunks_and_backpropagates() -> None:
    torch.manual_seed(3)
    model = make_model()
    images = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    state = torch.randn(2, 4)
    input_ids = torch.randint(0, 32, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    targets = torch.rand(2, 4, 3)
    targets[..., 2] = targets[..., 2] * 2 - 1
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

    output = model(
        images,
        state,
        input_ids,
        attention_mask,
        target_actions=targets,
        action_mask=mask,
    )
    assert output["actions"].shape == (2, 4, 3)
    assert torch.all((0 <= output["actions"][..., :2]) & (output["actions"][..., :2] <= 1))
    assert torch.all((-1 <= output["actions"][..., 2]) & (output["actions"][..., 2] <= 1))

    losses = act_loss(
        output["actions"],
        targets,
        mask,
        output["posterior_mean"],
        output["posterior_log_variance"],
        kl_weight=10.0,
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.action_head.weight.grad is not None


def test_inference_uses_deterministic_zero_latent() -> None:
    model = make_model().eval()
    images = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    state = torch.randn(1, 4)
    input_ids = torch.randint(0, 32, (1, 5))
    attention_mask = torch.ones(1, 5, dtype=torch.long)

    first = model(images, state, input_ids, attention_mask)["actions"]
    second = model(images, state, input_ids, attention_mask)["actions"]
    torch.testing.assert_close(first, second)


def test_model_accepts_the_default_128_pixel_training_input() -> None:
    model = make_model(image_size=128).eval()
    images = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8)
    state = torch.randn(1, 4)
    input_ids = torch.randint(0, 32, (1, 5))
    attention_mask = torch.ones(1, 5, dtype=torch.long)

    actions = model(images, state, input_ids, attention_mask)["actions"]

    assert actions.shape == (1, 4, 3)


def test_action_loss_ignores_padded_targets() -> None:
    predicted = torch.zeros(1, 3, 3)
    target = torch.zeros_like(predicted)
    target[:, 2] = 100.0
    mask = torch.tensor([[True, True, False]])
    latent = torch.zeros(1, 4)

    losses = act_loss(predicted, target, mask, latent, latent, kl_weight=10.0)
    assert losses["loss"].item() == 0.0
