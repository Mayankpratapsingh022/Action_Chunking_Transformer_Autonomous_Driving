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


def make_model(image_size: int = 64, *, policy_version: str = "v2") -> LanguageConditionedACT:
    config = ModelConfig(
        policy_version=policy_version,
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
        kl_weight=0.1,
        activity_logits=output["action_activity_logits"],
        action_magnitudes=output["action_magnitudes"],
        positive_weights=torch.tensor([2.0, 4.0]),
        sample_weights=torch.ones(2),
        steering_loss_weight=2.0,
        steering_active_weight=3.0,
        overlap_loss_weight=0.25,
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.action_activity_head.weight.grad is not None
    assert model.action_magnitude_head.weight.grad is not None
    assert model.steering_head.weight.grad is not None


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


def test_config_without_policy_version_loads_the_v1_architecture() -> None:
    config = ModelConfig.from_dict(
        {
            "image_size": 64,
            "chunk_size": 4,
            "d_model": 32,
            "nhead": 4,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "latent_dim": 8,
            "text_hidden_size": 16,
            "vision_channels": 16,
            "pretrained_vision": False,
        }
    )
    assert config.policy_version == "v1"

    model = make_model(policy_version=config.policy_version)
    assert hasattr(model, "action_head")
    assert not hasattr(model, "action_activity_head")


def test_balanced_objective_learns_nonzero_longitudinal_actions() -> None:
    torch.manual_seed(7)
    target = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.8, 0.0, 0.3], [0.0, 0.9, -0.4], [0.0, 0.0, 0.0]],
            [[0.7, 0.0, -0.2], [0.0, 0.8, 0.2], [0.0, 0.0, 0.0], [0.6, 0.0, 0.0]],
        ]
    )
    mask = torch.ones(2, 4, dtype=torch.bool)
    activity_logits = nn.Parameter(torch.full((2, 4, 2), -1.0))
    magnitude_logits = nn.Parameter(torch.zeros(2, 4, 2))
    steering_logits = nn.Parameter(torch.zeros(2, 4))
    optimizer = torch.optim.Adam((activity_logits, magnitude_logits, steering_logits), lr=0.15)
    latent = torch.zeros(2, 4)

    for _ in range(80):
        optimizer.zero_grad()
        magnitudes = magnitude_logits.sigmoid()
        steering = steering_logits.tanh()
        probabilities = activity_logits.sigmoid()
        actions = torch.cat(
            (torch.where(probabilities >= 0.5, magnitudes, 0.0), steering.unsqueeze(-1)),
            dim=-1,
        )
        losses = act_loss(
            actions,
            target,
            mask,
            latent,
            latent,
            kl_weight=0.0,
            activity_logits=activity_logits,
            action_magnitudes=magnitudes,
            positive_weights=torch.tensor([2.0, 4.0]),
            sample_weights=torch.ones(2),
            steering_loss_weight=2.0,
            steering_active_weight=3.0,
            overlap_loss_weight=0.25,
        )
        losses["loss"].backward()
        optimizer.step()

    predicted_active = activity_logits.sigmoid() >= 0.5
    target_active = target[..., :2] > 0.05
    assert torch.equal(predicted_active, target_active)
    assert magnitudes[target_active].mean() > 0.65
