from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from left_turn_vla.env import load_env_file  # noqa: E402


def main() -> None:
    load_env_file(PROJECT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Download the fine-tuned SmolVLA checkpoint for local inference.")
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_MODEL_REPO", "Mayank022/urban-vla-left-turn-smolvla-v2"),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "artifacts" / "smolvla-left-turn-v2"))
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=output,
        token=os.environ.get("HF_TOKEN"),
        max_workers=4,
    )
    print(output)


if __name__ == "__main__":
    main()
