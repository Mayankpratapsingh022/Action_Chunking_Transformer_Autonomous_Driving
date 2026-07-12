#!/usr/bin/env python3
"""Publish the training source without local environments or run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Mayank022/urban-vla-language-act")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=root,
        commit_message="Add Modal ACT training code",
        ignore_patterns=[
            ".git/**",
            ".env",
            ".venv/**",
            "artifacts/**",
            "logs/**",
            "**/__pycache__/**",
            "**/*.pyc",
            ".pytest_cache/**",
            ".runpod/**",
            ".ruff_cache/**",
            ".cache/**",
        ],
    )
    print(result)


if __name__ == "__main__":
    main()
