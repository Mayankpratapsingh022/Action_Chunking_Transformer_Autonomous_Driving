from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np

from urban_act.data import (
    CriticalWindowConfig,
    EpisodeRecord,
    UrbanEpisodeStream,
    compute_action_statistics,
    load_episode_records,
)


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


def test_episode_stream_resizes_source_video_during_decode(tmp_path: Path) -> None:
    episode_directory = tmp_path / "episode"
    episode_directory.mkdir()
    telemetry = []
    for frame_index in range(3):
        telemetry.append(
            json.dumps(
                {
                    "observation": {"state": [float(frame_index), 0.0, 0.5, 0.0]},
                    "action": [0.5, 0.0, 0.1],
                }
            )
        )
    (episode_directory / "telemetry.jsonl").write_text("\n".join(telemetry) + "\n")
    _write_test_video(episode_directory / "front.mp4", size=256, frame_count=3)

    record = EpisodeRecord(
        episode_id="test-resize",
        kind="nominal",
        split="train",
        task_id="turn_left_intersection",
        instruction="Turn left.",
        directory=episode_directory,
    )
    dataset = UrbanEpisodeStream(
        [record],
        image_size=128,
        chunk_size=2,
        stride=1,
        shuffle=False,
        shuffle_buffer=1,
        seed=42,
        critical_windows=CriticalWindowConfig(
            activity_threshold=0.05,
            steering_threshold=0.15,
            startup_weight=6.0,
            throttle_weight=2.0,
            brake_weight=4.0,
            turn_weight=2.5,
            recovery_weight=2.0,
        ),
    )

    sample = next(iter(dataset))

    assert sample["image"].shape == (3, 128, 128)
    assert sample["actions"].shape == (2, 3)
    assert sample["action_mask"].tolist() == [True, True]
    assert sample["sample_weight"].item() == 6.0


def test_action_statistics_produce_balancing_weights_and_zero_baseline(tmp_path: Path) -> None:
    directory = tmp_path / "episode"
    directory.mkdir()
    actions = ([1.0, 0.0, 0.0], [0.5, 0.0, 0.2], [0.0, 1.0, -0.4], [0.0, 0.0, 0.0])
    rows = [{"observation": {"state": [0.0, 0.0, 0.0, 0.0]}, "action": action} for action in actions]
    (directory / "telemetry.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    record = EpisodeRecord(
        episode_id="statistics",
        kind="nominal",
        split="train",
        task_id="continue_straight_intersection",
        instruction="Continue straight.",
        directory=directory,
    )

    statistics = compute_action_statistics(
        [record],
        activity_threshold=0.05,
        positive_weight_cap=12.0,
    )

    assert statistics["active_fraction"] == {"throttle": 0.5, "brake": 0.25}
    assert statistics["positive_weights"] == {"throttle": 1.0, "brake": 3.0}
    assert statistics["zero_baseline_mae"]["throttle"] == 0.375


def _write_test_video(path: Path, *, size: int, frame_count: int) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for frame_index in range(frame_count):
            pixels = np.full((size, size, 3), 40 + frame_index * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
