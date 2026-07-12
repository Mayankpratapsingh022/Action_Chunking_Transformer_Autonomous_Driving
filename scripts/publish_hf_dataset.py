#!/usr/bin/env python3
"""Publish the validated Urban VLA dataset without collector working files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "Mayank022/urban-vla-expert-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/urban-vla-expert-v1")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    report_path = root / "reports" / "validation-summary.json"
    if not report_path.exists():
        raise SystemExit(f"Missing validation report: {report_path}")

    report = json.loads(report_path.read_text())
    if not report.get("valid"):
        raise SystemExit("Dataset validation did not pass; refusing to publish")
    if report.get("collectedEpisodes") != report.get("manifestEpisodes"):
        raise SystemExit("Dataset is incomplete; refusing to publish")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=root,
        commit_message="Publish validated Urban VLA Expert v1 dataset",
        allow_patterns=[
            "README.md",
            "config.yaml",
            "schema.json",
            "manifests/**",
            "raw/accepted/**",
            "raw/failures/**",
            "reports/validation-summary.json",
        ],
    )
    print(result)


if __name__ == "__main__":
    main()
