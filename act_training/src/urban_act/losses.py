from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def act_loss(
    predicted_actions: Tensor,
    target_actions: Tensor,
    action_mask: Tensor,
    posterior_mean: Tensor,
    posterior_log_variance: Tensor,
    *,
    kl_weight: float,
    activity_logits: Tensor | None = None,
    action_magnitudes: Tensor | None = None,
    positive_weights: Tensor | None = None,
    sample_weights: Tensor | None = None,
    activity_threshold: float = 0.05,
    steering_active_threshold: float = 0.15,
    activity_loss_weight: float = 1.0,
    magnitude_loss_weight: float = 1.0,
    steering_loss_weight: float = 1.0,
    steering_active_weight: float = 1.0,
    overlap_loss_weight: float = 0.0,
) -> dict[str, Tensor]:
    if activity_logits is None or action_magnitudes is None:
        return _legacy_act_loss(
            predicted_actions,
            target_actions,
            action_mask,
            posterior_mean,
            posterior_log_variance,
            kl_weight=kl_weight,
        )

    valid = action_mask.unsqueeze(-1).to(predicted_actions.dtype)
    if sample_weights is None:
        sample_weights = torch.ones(
            predicted_actions.shape[0],
            device=predicted_actions.device,
            dtype=predicted_actions.dtype,
        )
    example_weights = sample_weights.to(predicted_actions.dtype).view(-1, 1, 1)
    weighted_valid = valid * example_weights

    activity_targets = (target_actions[..., :2] > activity_threshold).to(predicted_actions.dtype)
    if positive_weights is None:
        positive_weights = torch.ones(2, device=predicted_actions.device, dtype=predicted_actions.dtype)
    activity_errors = F.binary_cross_entropy_with_logits(
        activity_logits,
        activity_targets,
        pos_weight=positive_weights.to(device=predicted_actions.device, dtype=predicted_actions.dtype),
        reduction="none",
    )
    activity_loss = (activity_errors * weighted_valid).sum()
    activity_loss = activity_loss / (weighted_valid.sum() * 2).clamp_min(1.0)

    magnitude_errors = F.smooth_l1_loss(
        action_magnitudes,
        target_actions[..., :2],
        reduction="none",
        beta=0.05,
    )
    magnitude_weights = weighted_valid * activity_targets
    magnitude_loss = (magnitude_errors * magnitude_weights).sum()
    magnitude_loss = magnitude_loss / magnitude_weights.sum().clamp_min(1.0)

    steering_errors = F.smooth_l1_loss(
        predicted_actions[..., 2],
        target_actions[..., 2],
        reduction="none",
        beta=0.05,
    )
    steering_emphasis = torch.where(
        target_actions[..., 2].abs() > steering_active_threshold,
        steering_active_weight,
        1.0,
    )
    steering_weights = weighted_valid.squeeze(-1) * steering_emphasis
    steering_loss = (steering_errors * steering_weights).sum()
    steering_loss = steering_loss / steering_weights.sum().clamp_min(1.0)

    activity_probabilities = activity_logits.sigmoid()
    overlap_errors = activity_probabilities[..., 0] * activity_probabilities[..., 1]
    overlap_weights = weighted_valid.squeeze(-1)
    overlap_loss = (overlap_errors * overlap_weights).sum()
    overlap_loss = overlap_loss / overlap_weights.sum().clamp_min(1.0)

    reconstruction = (
        activity_loss_weight * activity_loss
        + magnitude_loss_weight * magnitude_loss
        + steering_loss_weight * steering_loss
        + overlap_loss_weight * overlap_loss
    )
    kl = _kl_divergence(posterior_mean, posterior_log_variance)
    total = reconstruction + kl_weight * kl
    return {
        "loss": total,
        "reconstruction_loss": reconstruction,
        "activity_loss": activity_loss,
        "magnitude_loss": magnitude_loss,
        "steering_loss": steering_loss,
        "overlap_loss": overlap_loss,
        "kl_loss": kl,
    }


def _legacy_act_loss(
    predicted_actions: Tensor,
    target_actions: Tensor,
    action_mask: Tensor,
    posterior_mean: Tensor,
    posterior_log_variance: Tensor,
    *,
    kl_weight: float,
) -> dict[str, Tensor]:
    valid = action_mask.unsqueeze(-1).to(predicted_actions.dtype)
    reconstruction = ((predicted_actions - target_actions).abs() * valid).sum()
    reconstruction = reconstruction / (valid.sum() * predicted_actions.shape[-1]).clamp_min(1.0)
    kl = _kl_divergence(posterior_mean, posterior_log_variance)
    total = reconstruction + kl_weight * kl
    return {"loss": total, "reconstruction_loss": reconstruction, "kl_loss": kl}


def _kl_divergence(posterior_mean: Tensor, posterior_log_variance: Tensor) -> Tensor:
    kl = -0.5 * (1.0 + posterior_log_variance - posterior_mean.square() - posterior_log_variance.exp())
    return kl.mean()
