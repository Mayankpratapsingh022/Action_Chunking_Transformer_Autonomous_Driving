from __future__ import annotations

import json
from pathlib import Path

from left_turn_vla.config import TrainConfig


def build_train_command(
    config: TrainConfig,
    *,
    executable: str = "lerobot-train",
    output_dir: str | Path,
    resume_config: str | Path | None = None,
    wandb_enabled: bool = False,
    wandb_project: str = "urban-left-turn-vla",
) -> list[str]:
    output_dir = Path(output_dir).expanduser().resolve()
    command = [executable]
    if resume_config is None:
        command.append(f"--policy.path={config.base_model}")
    else:
        command.extend(("--resume=true", f"--config_path={Path(resume_config).expanduser().resolve()}"))

    if config.rename_map:
        command.append(f"--rename_map={json.dumps(config.rename_map, separators=(',', ':'))}")

    command.extend(
        [
            f"--dataset.repo_id={config.dataset_repo}",
            f"--dataset.eval_split={config.eval_split}",
            f"--dataset.image_transforms.enable={_bool(config.image_augmentation)}",
            "--dataset.image_transforms.max_num_transforms=3",
            "--dataset.image_transforms.random_order=true",
            f"--output_dir={output_dir}",
            f"--job_name={config.run_name}",
            f"--seed={config.seed}",
            f"--batch_size={config.batch_size}",
            f"--num_workers={config.num_workers}",
            f"--steps={config.steps}",
            f"--eval_steps={config.eval_interval}",
            f"--max_eval_samples={config.max_eval_samples}",
            f"--save_freq={config.save_interval}",
            f"--log_freq={config.log_interval}",
            "--env_eval_freq=0",
            "--save_checkpoint=true",
            "--save_checkpoint_to_hub=false",
            "--policy.device=cuda",
            f"--policy.repo_id={config.model_repo}",
            f"--policy.push_to_hub={_bool(config.push_to_hub)}",
            f"--policy.chunk_size={config.chunk_size}",
            f"--policy.n_action_steps={config.action_steps}",
            f"--policy.freeze_vision_encoder={_bool(config.freeze_vision_encoder)}",
            f"--policy.train_expert_only={_bool(config.train_expert_only)}",
            f"--policy.scheduler_warmup_steps={config.warmup_steps}",
            f"--policy.scheduler_decay_steps={config.steps}",
            f"--wandb.enable={_bool(wandb_enabled)}",
            f"--wandb.project={wandb_project}",
        ]
    )
    return command


def find_resume_config(output_dir: str | Path) -> Path | None:
    root = Path(output_dir).expanduser().resolve()
    direct = root / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if direct.exists():
        return direct.resolve()
    candidates = sorted(root.glob("checkpoints/*/pretrained_model/train_config.json"))
    return candidates[-1].resolve() if candidates else None


def latest_pretrained_model(output_dir: str | Path) -> Path:
    config_path = find_resume_config(output_dir)
    if config_path is None:
        raise FileNotFoundError(f"No completed checkpoint found under {Path(output_dir)}")
    return config_path.parent


def _bool(value: bool) -> str:
    return "true" if value else "false"


__all__ = ["build_train_command", "find_resume_config", "latest_pretrained_model"]
