from __future__ import annotations

from torch import Tensor


def act_loss(
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
    kl = -0.5 * (1.0 + posterior_log_variance - posterior_mean.square() - posterior_log_variance.exp())
    kl = kl.mean()
    total = reconstruction + kl_weight * kl
    return {"loss": total, "reconstruction_loss": reconstruction, "kl_loss": kl}

