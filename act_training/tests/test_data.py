from __future__ import annotations

import json
from pathlib import Path

from urban_act.data import load_episode_records


def _add_episode(root: Path, episode_id: str) -> None:
    directory = root / "raw" / "accepted" / episode_id
    directory.mkdir(parents=True)
    for name in ("episode.json", "telemetry.jsonl", "front.mp4"):
        (directory / name).touch()


def test_manifest_split_is_preserved_and_failure_manifest_is_ignored(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    train = {
        "id": "nominal-turn-left-0000",
        "kind": "nominal",
        "split": "train",
        "taskId": "turn_left_intersection",
        "instruction": "Turn left at the intersection.",
    }
    validation = {
        "id": "recovery-turn-left-0000",
        "kind": "recovery",
        "split": "validation",
        "taskId": "turn_left_intersection",
        "instruction": "Take the left turn.",
    }
    failure = {
        "id": "failure-turn-left-0000",
        "kind": "failure",
        "split": "analysis",
        "taskId": "turn_left_intersection",
        "instruction": "Turn left.",
    }
    (manifests / "nominal.jsonl").write_text(json.dumps(train) + "\n")
    (manifests / "recovery.jsonl").write_text(json.dumps(validation) + "\n")
    (manifests / "failures.jsonl").write_text(json.dumps(failure) + "\n")
    _add_episode(tmp_path, train["id"])
    _add_episode(tmp_path, validation["id"])

    train_records = load_episode_records(tmp_path, "train")
    validation_records = load_episode_records(tmp_path, "validation")

    assert [record.episode_id for record in train_records] == [train["id"]]
    assert [record.episode_id for record in validation_records] == [validation["id"]]
    assert all(record.kind != "failure" for record in train_records + validation_records)

