from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from urban_act.losses import act_loss
from urban_act.metrics import ActionMetricAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that the ACT v2 objective rejects an all-zero policy.")
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(17)
    target, mask, state = _synthetic_batch()
    activity_logits = nn.Parameter(torch.full((*target.shape[:2], 2), -1.0))
    magnitude_logits = nn.Parameter(torch.zeros(*target.shape[:2], 2))
    steering_logits = nn.Parameter(torch.zeros(*target.shape[:2]))
    optimizer = torch.optim.Adam((activity_logits, magnitude_logits, steering_logits), lr=0.12)
    latent = torch.zeros(target.shape[0], 4)

    for _ in range(args.steps):
        optimizer.zero_grad()
        probabilities = activity_logits.sigmoid()
        magnitudes = magnitude_logits.sigmoid()
        steering = steering_logits.tanh()
        predicted = torch.cat(
            (torch.where(probabilities >= 0.5, magnitudes, 0.0), steering.unsqueeze(-1)),
            dim=-1,
        )
        losses = act_loss(
            predicted,
            target,
            mask,
            latent,
            latent,
            kl_weight=0.0,
            activity_logits=activity_logits,
            action_magnitudes=magnitudes,
            positive_weights=torch.tensor([2.0, 4.0]),
            sample_weights=torch.tensor([6.0, 4.0]),
            steering_loss_weight=2.0,
            steering_active_weight=3.0,
            overlap_loss_weight=0.25,
        )
        losses["loss"].backward()
        optimizer.step()

    with torch.no_grad():
        probabilities = activity_logits.sigmoid()
        magnitudes = magnitude_logits.sigmoid()
        predicted = torch.cat(
            (
                torch.where(probabilities >= 0.5, magnitudes, 0.0),
                steering_logits.tanh().unsqueeze(-1),
            ),
            dim=-1,
        )
    accumulator = ActionMetricAccumulator()
    accumulator.update(predicted, target, mask, ["launch", "yield"], states=state)
    metrics = accumulator.compute()
    summary = {
        "steps": args.steps,
        "selection_score": metrics["selection_score"],
        "collapse_detected": metrics["collapse_detected"],
        "quality_gates": metrics["quality_gates"],
        "activity": metrics["activity"],
        "startup_throttle": metrics["startup_throttle"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if metrics["collapse_detected"] or not metrics["quality_gates"]["all_passed"]:
        raise SystemExit(1)


def _synthetic_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.8, 0.0, 0.35], [0.0, 0.9, -0.4], [0.0, 0.0, 0.0]],
            [[0.7, 0.0, -0.25], [0.0, 0.8, 0.25], [0.0, 0.0, 0.0], [0.6, 0.0, 0.0]],
        ]
    )
    mask = torch.ones(2, 4, dtype=torch.bool)
    state = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    return target, mask, state


if __name__ == "__main__":
    main()
