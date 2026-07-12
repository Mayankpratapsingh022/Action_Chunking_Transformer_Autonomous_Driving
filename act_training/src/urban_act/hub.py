from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi

RESULTS_START = "<!-- TRAINING_RESULTS_START -->"
RESULTS_END = "<!-- TRAINING_RESULTS_END -->"


def publish_training_run(
    *,
    repo_id: str,
    run_dir: str | Path,
    run_name: str,
    metrics: dict[str, Any],
    readme_template: str | Path,
    private: bool,
    token: str,
) -> str:
    root = Path(run_dir)
    rendered_readme = root / "README.hub.md"
    rendered_readme.write_text(_render_readme(Path(readme_template).read_text(), run_name, metrics))

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    operations: list[CommitOperationAdd] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == rendered_readme:
            continue
        repository_path = f"runs/{run_name}/{path.relative_to(root).as_posix()}"
        operations.append(
            CommitOperationAdd(path_in_repo=repository_path, path_or_fileobj=path)
        )

    top_level_files = ("model.safetensors", "config.json", "training_config.json", "metrics.json", "history.json")
    for name in top_level_files:
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=root / name))
    for directory in ("tokenizer", "plots"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                operations.append(
                    CommitOperationAdd(path_in_repo=path.relative_to(root).as_posix(), path_or_fileobj=path)
                )
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=rendered_readme))

    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Publish trained checkpoint {run_name}",
    )
    return commit.commit_url


def _render_readme(template: str, run_name: str, metrics: dict[str, Any]) -> str:
    validation = metrics["validation"]
    test = metrics["test"]
    action_mae = test["action_mae"]
    result = "\n".join(
        (
            f"Latest completed run: `{run_name}`.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Best validation mean action MAE | {validation['mean_action_mae']:.5f} |",
            f"| Test mean action MAE | {test['mean_action_mae']:.5f} |",
            f"| Test throttle MAE | {action_mae['throttle']:.5f} |",
            f"| Test brake MAE | {action_mae['brake']:.5f} |",
            f"| Test steering MAE | {action_mae['steering']:.5f} |",
            f"| Test brake accuracy | {test['brake_accuracy']:.2%} |",
            f"| Test steering-direction accuracy | {test['steering_direction_accuracy']:.2%} |",
            "",
            f"Full metrics and plots are in [`runs/{run_name}/`](./runs/{run_name}/).",
        )
    )
    before, remainder = template.split(RESULTS_START, maxsplit=1)
    _, after = remainder.split(RESULTS_END, maxsplit=1)
    return f"{before}{RESULTS_START}\n{result}\n{RESULTS_END}{after}"


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
