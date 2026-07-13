from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal

PROJECT_DIR = Path(__file__).resolve().parent
APP_NAME = "urban-vla-language-act-training"
ARTIFACT_VOLUME_NAME = "urban-vla-act-artifacts"
CACHE_VOLUME_NAME = "urban-vla-act-cache"

RUNTIME_PACKAGES = (
    "av==15.0.0",
    "huggingface-hub==0.36.2",
    "matplotlib==3.9.2",
    "numpy==1.26.4",
    "safetensors==0.5.3",
    "torch==2.8.0",
    "torchvision==0.23.0",
    "tqdm==4.67.1",
    "transformers==4.52.3",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(*RUNTIME_PACKAGES)
    .env(
        {
            "PYTHONPATH": "/root/src",
            "HF_HOME": "/cache/huggingface",
            "TORCH_HOME": "/cache/torch",
            "MPLCONFIGDIR": "/cache/matplotlib",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(PROJECT_DIR / "src", remote_path="/root/src")
    .add_local_file(PROJECT_DIR / "README.md", remote_path="/root/act_training/README.md")
)

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
huggingface_secret = (
    modal.Secret.from_local_environ(["HF_TOKEN"])
    if os.environ.get("HF_TOKEN")
    else modal.Secret.from_name("huggingface")
)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32_768,
    timeout=86_400,
    max_containers=1,
    secrets=[huggingface_secret],
    volumes={"/artifacts": artifact_volume, "/cache": cache_volume},
)
def train_on_modal(config_values: dict[str, Any]) -> dict[str, Any]:
    from urban_act.config import TrainConfig
    from urban_act.train import run_training

    config = TrainConfig.from_dict(config_values)
    try:
        return run_training(
            config,
            readme_template="/root/act_training/README.md",
            checkpoint_callback=artifact_volume.commit,
        )
    finally:
        artifact_volume.commit()
        cache_volume.commit()


@app.local_entrypoint()
def main(
    run_name: str = "act-driving-v2",
    max_steps: int = 10_000,
    batch_size: int = 64,
    config_path: str = "configs/base.json",
    download: bool = True,
) -> None:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    config = json.loads(path.read_text())
    config.update({"run_name": run_name, "max_steps": max_steps, "batch_size": batch_size})
    result = train_on_modal.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))

    if download:
        destination = PROJECT_DIR / "artifacts" / run_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "modal",
                "volume",
                "get",
                "--force",
                ARTIFACT_VOLUME_NAME,
                f"runs/{run_name}",
                str(destination),
            ],
            check=True,
        )
        print(f"Downloaded artifacts to {destination}")
