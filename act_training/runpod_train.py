from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from urban_act.config import TrainConfig
from urban_act.hub import write_json

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE_ROOT = Path(os.environ.get("ACT_WORKSPACE_ROOT", "/workspace/act-driving"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the language-conditioned ACT policy on a RunPod GPU Pod.")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs" / "base.json"))
    parser.add_argument("--run-name")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--artifact-root", default=str(DEFAULT_WORKSPACE_ROOT / "artifacts" / "runs"))
    parser.add_argument("--cache-dir", default=str(DEFAULT_WORKSPACE_ROOT / "cache" / "huggingface"))
    parser.add_argument("--no-push-to-hub", action="store_true")
    return parser.parse_args()


def load_config(
    config_path: str | Path,
    *,
    run_name: str | None = None,
    max_steps: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    resume: str | None = None,
    artifact_root: str | None = None,
    cache_dir: str | None = None,
    push_to_hub: bool | None = None,
) -> TrainConfig:
    values = json.loads(Path(config_path).read_text())
    overrides = {
        "run_name": run_name,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "resume": resume,
        "artifact_root": artifact_root,
        "cache_dir": cache_dir,
        "push_to_hub": push_to_hub,
    }
    values.update({name: value for name, value in overrides.items() if value is not None})
    config = TrainConfig.from_dict(values)
    config.validate()
    return config


def main() -> None:
    from urban_act.train import run_training

    args = parse_args()
    config = load_config(
        args.config,
        run_name=args.run_name,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=args.resume,
        artifact_root=args.artifact_root,
        cache_dir=args.cache_dir,
        push_to_hub=False if args.no_push_to_hub else None,
    )
    run_dir = Path(config.artifact_root) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(config.artifact_root).parent / "status" / f"{config.run_name}.json"
    started_at = _timestamp()
    runtime = _runtime_metadata()
    _write_status(
        status_path,
        status="running",
        started_at=started_at,
        updated_at=started_at,
        run_name=config.run_name,
        runtime=runtime,
    )
    print(json.dumps({"event": "run_started", "run_name": config.run_name, "runtime": runtime}), flush=True)

    try:
        result = run_training(config, readme_template=PROJECT_DIR / "README.md")
    except BaseException as error:
        failed_at = _timestamp()
        _write_status(
            status_path,
            status="failed",
            started_at=started_at,
            updated_at=failed_at,
            run_name=config.run_name,
            runtime=runtime,
            error={"type": type(error).__name__, "message": str(error)},
        )
        print(
            json.dumps(
                {
                    "event": "run_failed",
                    "run_name": config.run_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            flush=True,
        )
        raise

    completed_at = _timestamp()
    _write_status(
        status_path,
        status="completed",
        started_at=started_at,
        updated_at=completed_at,
        run_name=config.run_name,
        runtime=runtime,
        result=result,
    )
    print(json.dumps({"event": "run_completed", **result}, default=str), flush=True)


def _runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "runpod",
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(0)
        metadata["gpu_count"] = torch.cuda.device_count()
    return metadata


def _write_status(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, values)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
