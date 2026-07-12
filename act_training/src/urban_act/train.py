from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from torch import Tensor
from transformers import AutoTokenizer

from urban_act.checkpoints import (
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    save_inference_weights,
    training_state,
)
from urban_act.config import TrainConfig
from urban_act.data import UrbanEpisodeStream, compute_state_statistics, load_episode_records, make_dataloader
from urban_act.hub import publish_training_run, write_json
from urban_act.losses import act_loss
from urban_act.metrics import ActionMetricAccumulator, EvaluationSamples
from urban_act.model import LanguageConditionedACT, ModelConfig
from urban_act.plots import write_all_plots

TRANSIENT_HUB_STATUS_CODES = frozenset({408, 425, 500, 502, 503, 504})


def _download_dataset_snapshot(
    config: TrainConfig,
    *,
    token: str | None,
    attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    for attempt in range(1, attempts + 1):
        try:
            return Path(
                snapshot_download(
                    repo_id=config.dataset_repo,
                    repo_type="dataset",
                    revision=config.dataset_revision,
                    cache_dir=config.cache_dir,
                    token=token,
                    allow_patterns=("config.yaml", "schema.json", "manifests/*.jsonl", "raw/accepted/**"),
                )
            )
        except (HfHubHTTPError, LocalEntryNotFoundError) as error:
            response = _hub_error_response(error)
            status_code = response.status_code if response is not None else None
            if attempt == attempts or (status_code != 429 and status_code not in TRANSIENT_HUB_STATUS_CODES):
                raise
            delay_seconds = _rate_limit_delay(response) if status_code == 429 else min(2 ** (attempt - 1), 16)
            print(
                json.dumps(
                    {
                        "event": "hub_download_retry",
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "status_code": status_code,
                        "retry_in_seconds": delay_seconds,
                    }
                ),
                flush=True,
            )
            sleep(delay_seconds)

    raise RuntimeError("Dataset download retry loop exited unexpectedly")


def _hub_error_response(error: BaseException) -> Any | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, HfHubHTTPError) and current.response is not None:
            return current.response
        current = current.__cause__ or current.__context__
    return None


def _rate_limit_delay(response: Any | None) -> int:
    if response is not None:
        match = re.search(r"(?:^|[;,])\s*t=(\d+)", response.headers.get("RateLimit", ""))
        if match:
            return int(match.group(1)) + 1
    return 301


def run_training(
    config: TrainConfig,
    *,
    readme_template: str | Path,
    checkpoint_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    config.validate()
    _seed_everything(config.seed)
    run_dir = Path(config.artifact_root) / config.run_name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "training_config.json", config.to_dict())

    token = os.environ.get("HF_TOKEN")
    dataset_root = _download_dataset_snapshot(config, token=token)
    train_records = load_episode_records(dataset_root, "train")
    validation_records = load_episode_records(dataset_root, "validation")
    test_records = load_episode_records(dataset_root, "test")

    state_stats_path = run_dir / "state_statistics.json"
    if state_stats_path.exists():
        state_statistics = json.loads(state_stats_path.read_text())
    else:
        state_statistics = compute_state_statistics(train_records)
        write_json(state_stats_path, state_statistics)

    model_config = ModelConfig(
        image_size=config.image_size,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        chunk_size=config.chunk_size,
        d_model=config.d_model,
        nhead=config.nhead,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
        text_model_name=config.text_model_name,
        freeze_text_encoder=config.freeze_text_encoder,
        pretrained_vision=config.pretrained_vision,
        state_mean=tuple(state_statistics["mean"]),
        state_std=tuple(state_statistics["std"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ACT training requires a CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name, cache_dir=config.cache_dir)
    model = LanguageConditionedACT(model_config).to(device)
    optimizer = _make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(config))
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision == "fp16")
    amp_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    amp_enabled = config.mixed_precision != "none"

    train_dataset = UrbanEpisodeStream(
        train_records,
        image_size=config.image_size,
        chunk_size=config.chunk_size,
        stride=config.train_stride,
        shuffle=True,
        shuffle_buffer=config.shuffle_buffer,
        seed=config.seed,
    )
    validation_dataset = UrbanEpisodeStream(
        validation_records,
        image_size=config.image_size,
        chunk_size=config.chunk_size,
        stride=config.eval_stride,
        shuffle=True,
        shuffle_buffer=1,
        seed=config.seed + 1,
    )
    test_dataset = UrbanEpisodeStream(
        test_records,
        image_size=config.image_size,
        chunk_size=config.chunk_size,
        stride=config.eval_stride,
        shuffle=True,
        shuffle_buffer=1,
        seed=config.seed + 2,
    )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
    )
    validation_loader = make_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=False,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=False,
    )

    history: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    global_step = 0
    epoch = 0
    elapsed_offset = 0.0
    best_validation_mae = math.inf
    best_step = 0
    best_validation_metrics: dict[str, Any] | None = None
    resume_path = _resume_path(run_dir, config.resume)
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        global_step = int(checkpoint["global_step"])
        epoch = int(checkpoint["epoch"])
        elapsed_offset = float(checkpoint.get("elapsed_seconds", 0.0))
        best_validation_mae = float(checkpoint.get("best_validation_mae", math.inf))
        best_step = int(checkpoint.get("best_step", 0))
        best_validation_metrics = checkpoint.get("best_validation_metrics")
        history = checkpoint.get("history", history)
        restore_rng_state(checkpoint)
        _log_event(run_dir, {"event": "resumed", "checkpoint": str(resume_path), "step": global_step})

    started = time.monotonic()
    rolling = {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0, "count": 0}
    last_validation_samples = EvaluationSamples()

    while global_step < config.max_steps:
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            if global_step >= config.max_steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tensors = _move_batch(batch, device)
            encoded = _tokenize(tokenizer, batch["instruction"], device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                output = model(
                    tensors["image"],
                    tensors["state"],
                    encoded["input_ids"],
                    encoded["attention_mask"],
                    target_actions=tensors["actions"],
                    action_mask=tensors["action_mask"],
                )
                losses = act_loss(
                    output["actions"],
                    tensors["actions"],
                    tensors["action_mask"],
                    output["posterior_mean"],
                    output["posterior_log_variance"],
                    kl_weight=config.kl_weight,
                )

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            for name in ("loss", "reconstruction_loss", "kl_loss"):
                rolling[name] += float(losses[name].detach())
            rolling["count"] += 1

            if global_step % config.log_interval == 0:
                elapsed = elapsed_offset + time.monotonic() - started
                row = {
                    "step": global_step,
                    "loss": rolling["loss"] / rolling["count"],
                    "reconstruction_loss": rolling["reconstruction_loss"] / rolling["count"],
                    "kl_loss": rolling["kl_loss"] / rolling["count"],
                    "learning_rate": optimizer.param_groups[-1]["lr"],
                }
                history["train"].append(row)
                _log_event(run_dir, _progress_event(row, config.max_steps, elapsed))
                rolling = {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0, "count": 0}

            if global_step % config.eval_interval == 0:
                validation_metrics, last_validation_samples = evaluate(
                    model,
                    validation_loader,
                    validation_dataset,
                    tokenizer,
                    device,
                    max_batches=config.eval_batches,
                    mixed_precision=config.mixed_precision,
                )
                validation_row = {"step": global_step, **validation_metrics}
                history["validation"].append(validation_row)
                _log_event(run_dir, {"event": "validation", **validation_row})
                if validation_metrics["mean_action_mae"] < best_validation_mae:
                    best_validation_mae = validation_metrics["mean_action_mae"]
                    best_validation_metrics = validation_metrics
                    best_step = global_step
                    save_checkpoint(
                        run_dir / "best.pt",
                        _checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            global_step=global_step,
                            epoch=epoch,
                            elapsed_seconds=elapsed_offset + time.monotonic() - started,
                            best_validation_mae=best_validation_mae,
                            best_step=best_step,
                            best_validation_metrics=best_validation_metrics,
                            history=history,
                        ),
                    )
                    _after_checkpoint(checkpoint_callback)

            if global_step % config.checkpoint_interval == 0:
                save_checkpoint(
                    run_dir / "last.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        global_step=global_step,
                        epoch=epoch,
                        elapsed_seconds=elapsed_offset + time.monotonic() - started,
                        best_validation_mae=best_validation_mae,
                        best_step=best_step,
                        best_validation_metrics=best_validation_metrics,
                        history=history,
                    ),
                )
                _after_checkpoint(checkpoint_callback)
        epoch += 1

    if not history["validation"] or history["validation"][-1]["step"] != global_step:
        validation_metrics, last_validation_samples = evaluate(
            model,
            validation_loader,
            validation_dataset,
            tokenizer,
            device,
            max_batches=config.eval_batches,
            mixed_precision=config.mixed_precision,
        )
        history["validation"].append({"step": global_step, **validation_metrics})
        if validation_metrics["mean_action_mae"] < best_validation_mae:
            best_validation_mae = validation_metrics["mean_action_mae"]
            best_validation_metrics = validation_metrics
            best_step = global_step
            save_checkpoint(
                run_dir / "best.pt",
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                    epoch=epoch,
                    elapsed_seconds=elapsed_offset + time.monotonic() - started,
                    best_validation_mae=best_validation_mae,
                    best_step=best_step,
                    best_validation_metrics=best_validation_metrics,
                    history=history,
                ),
            )
            _after_checkpoint(checkpoint_callback)

    save_checkpoint(
        run_dir / "last.pt",
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            epoch=epoch,
            elapsed_seconds=elapsed_offset + time.monotonic() - started,
            best_validation_mae=best_validation_mae,
            best_step=best_step,
            best_validation_metrics=best_validation_metrics,
            history=history,
        ),
    )
    _after_checkpoint(checkpoint_callback)

    best_checkpoint = load_checkpoint(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics, test_samples = evaluate(
        model,
        test_loader,
        test_dataset,
        tokenizer,
        device,
        max_batches=config.test_batches,
        mixed_precision=config.mixed_precision,
    )
    metrics = {
        "best_step": best_step,
        "validation": best_validation_metrics,
        "test": test_metrics,
    }

    save_inference_weights(model, run_dir / "model.safetensors")
    model_payload = {"architectures": ["LanguageConditionedACT"], "model_type": "urban_language_act"}
    model_payload.update(model_config.to_dict())
    write_json(run_dir / "config.json", model_payload)
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "metrics.json", metrics)
    tokenizer.save_pretrained(run_dir / "tokenizer")
    write_all_plots(run_dir / "plots", history, test_metrics, test_samples or last_validation_samples)

    hub_url = None
    if config.push_to_hub:
        if not token:
            raise RuntimeError("HF_TOKEN is required when push_to_hub is enabled")
        hub_url = publish_training_run(
            repo_id=config.model_repo,
            run_dir=run_dir,
            run_name=config.run_name,
            metrics=metrics,
            readme_template=readme_template,
            private=config.hub_private,
            token=token,
        )
    result = {
        "run_name": config.run_name,
        "artifact_path": str(run_dir),
        "best_step": best_step,
        "metrics": metrics,
        "hub_url": hub_url,
    }
    _log_event(run_dir, {"event": "complete", **result})
    return result


@torch.inference_mode()
def evaluate(
    model: LanguageConditionedACT,
    loader: Any,
    dataset: UrbanEpisodeStream,
    tokenizer: Any,
    device: torch.device,
    *,
    max_batches: int,
    mixed_precision: str,
) -> tuple[dict[str, Any], EvaluationSamples]:
    model.eval()
    dataset.set_epoch(0)
    accumulator = ActionMetricAccumulator()
    amp_enabled = mixed_precision != "none"
    amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        tensors = _move_batch(batch, device)
        encoded = _tokenize(tokenizer, batch["instruction"], device)
        context = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
        with context:
            output = model(
                tensors["image"],
                tensors["state"],
                encoded["input_ids"],
                encoded["attention_mask"],
            )
        accumulator.update(output["actions"], tensors["actions"], tensors["action_mask"], batch["task_id"])
    return accumulator.compute(), accumulator.samples


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in ("image", "state", "actions", "action_mask")
    }


def _tokenize(tokenizer: Any, instructions: list[str], device: torch.device) -> dict[str, Tensor]:
    encoded = tokenizer(
        instructions,
        padding=True,
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )
    return {name: value.to(device, non_blocking=True) for name, value in encoded.items()}


def _make_optimizer(model: LanguageConditionedACT, config: TrainConfig) -> torch.optim.Optimizer:
    backbone_parameters = [parameter for parameter in model.vision_backbone.parameters() if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    policy_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        (
            {"params": backbone_parameters, "lr": config.backbone_learning_rate},
            {"params": policy_parameters, "lr": config.learning_rate},
        ),
        weight_decay=config.weight_decay,
        fused=torch.cuda.is_available(),
    )


def _schedule(config: TrainConfig) -> Any:
    def multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return max(step, 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return multiplier


def _resume_path(run_dir: Path, resume: str) -> Path | None:
    if resume in {"", "none", "false"}:
        return None
    candidate = run_dir / "last.pt" if resume == "auto" else Path(resume)
    return candidate if candidate.exists() else None


def _checkpoint_payload(**values: Any) -> dict[str, Any]:
    return training_state(**values)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _progress_event(row: dict[str, Any], max_steps: int, elapsed: float) -> dict[str, Any]:
    step = int(row["step"])
    steps_per_second = step / max(elapsed, 1e-6)
    eta = (max_steps - step) / max(steps_per_second, 1e-6)
    return {
        "event": "train_progress",
        **row,
        "percent": round(step / max_steps * 100.0, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1),
    }


def _log_event(run_dir: Path, event: dict[str, Any]) -> None:
    line = json.dumps(event, sort_keys=True, default=str)
    with (run_dir / "logs" / "train.jsonl").open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _after_checkpoint(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()
