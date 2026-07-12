#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from runpod_launcher import load_env_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    base_config = json.loads((PROJECT_DIR / "configs" / "base.json").read_text())
    parser = argparse.ArgumentParser(description="Download one completed ACT run from Hugging Face.")
    parser.add_argument("--run-name", default="act-driving-v1")
    parser.add_argument("--repo-id", default=base_config["model_repo"])
    parser.add_argument("--destination")
    parser.add_argument("--cache-dir", default=str(PROJECT_DIR / ".cache" / "huggingface"))
    parser.add_argument("--env-file", default=str(PROJECT_DIR / ".env"))
    return parser.parse_args()


def select_run_files(repository_files: list[str], run_name: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, dots, underscores, and hyphens")
    prefix = f"runs/{run_name}/"
    return sorted(path for path in repository_files if path.startswith(prefix) and path != prefix)


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    token = os.environ.get("HF_TOKEN") or None
    destination = Path(args.destination) if args.destination else PROJECT_DIR / "artifacts" / args.run_name
    api = HfApi(token=token)
    files = select_run_files(api.list_repo_files(args.repo_id, repo_type="model"), args.run_name)
    if not files:
        raise RuntimeError(f"No completed run named {args.run_name!r} exists in {args.repo_id}")

    prefix = f"runs/{args.run_name}/"
    for index, repository_path in enumerate(files, start=1):
        cached_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="model",
            filename=repository_path,
            token=token,
            cache_dir=args.cache_dir,
        )
        local_path = destination / repository_path.removeprefix(prefix)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, local_path)
        print(f"[{index}/{len(files)}] {local_path}")
    print(f"Downloaded {len(files)} files to {destination}")


if __name__ == "__main__":
    main()
