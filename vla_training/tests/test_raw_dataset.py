import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from left_turn_vla.constants import LEFT_TURN_INSTRUCTION
from left_turn_vla.raw_dataset import analyze_directory, inspect_episode, iter_lerobot_frames, load_episode


def _image_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), (20, 40, 60)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _sample(index: int, *, collision: bool = False) -> dict:
    return {
        "seed": 42,
        "capture_resolution": 128,
        "image": _image_url(),
        "language_id": 0,
        "language_text": LEFT_TURN_INSTRUCTION,
        "control_target": {"targetSpeedMps": 18.0, "targetSteer": -(0.2 + index * 0.1), "mode": "turn_left"},
        "control": {"throttle": 0.4 + index * 0.1, "brake": 0.0, "steer": -0.2},
        "ego": {"speed": 0.5, "steering": -0.1},
        "route_state": {"progress": 0.5, "lateralErrorM": 0.1, "headingErrorRad": -0.05},
        "events": {"collision": collision, "offRoute": False, "redLightViolation": False},
        "task": {"routeProgress": 0.96},
    }


def _write_episode(path: Path, *, collision: bool = False) -> None:
    samples = [_sample(index % 2, collision=collision and index == 10) for index in range(60)]
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "vla-urban-4",
                    "action_encoding": "target_speed_steering",
                },
                "samples": samples,
            }
        )
    )


def test_analysis_ignores_cumulative_exports_and_rejects_collision(tmp_path: Path) -> None:
    _write_episode(tmp_path / "human-clean.json")
    _write_episode(tmp_path / "human-collision.json", collision=True)
    (tmp_path / "vla_urban_dataset_123.json").write_text("{}")
    inspections, report = analyze_directory(tmp_path)
    assert report["independent_files"] == 2
    assert report["old_cumulative_exports_ignored"] == 1
    assert report["accepted_episodes"] == 1
    assert sum(item.collision for item in inspections) == 1


def test_held_out_tail_is_seed_disjoint(tmp_path: Path) -> None:
    for seed in range(10):
        path = tmp_path / f"human-{seed}.json"
        _write_episode(path)
        payload = json.loads(path.read_text())
        for sample in payload["samples"]:
            sample["seed"] = seed
        path.write_text(json.dumps(payload))
    inspections, report = analyze_directory(tmp_path, eval_split=0.2)
    accepted = [item for item in inspections if item.accepted]
    held_out = accepted[-report["held_out_episodes"] :]
    training = accepted[: -report["held_out_episodes"]]
    assert {item.seed for item in training}.isdisjoint(item.seed for item in held_out)
    assert report["seed_disjoint_holdout"]


def test_frame_conversion_aligns_previous_targets_and_speed_units(tmp_path: Path) -> None:
    path = tmp_path / "human-clean.json"
    _write_episode(path)
    frames = list(iter_lerobot_frames(load_episode(path)))
    np.testing.assert_allclose(frames[0]["observation.state"], [12.0, -0.1, 0.0, 0.0])
    np.testing.assert_allclose(frames[1]["observation.state"], [12.0, -0.1, 18.0, -0.2])
    np.testing.assert_allclose(frames[1]["action"], [18.0, -0.3])
    assert frames[0]["observation.images.front"].shape == (128, 128, 3)


def test_legacy_raw_pedal_episode_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "human-legacy.json"
    _write_episode(path)
    payload = json.loads(path.read_text())
    payload["metadata"]["schema_version"] = "vla-urban-3"
    path.write_text(json.dumps(payload))
    inspections, report = analyze_directory(tmp_path)
    assert report["accepted_episodes"] == 0
    assert "vla-urban-4" in inspections[0].rejection_reason


def test_episode_quality_gates_reject_bad_alignment_and_long_stops(tmp_path: Path) -> None:
    aligned = tmp_path / "human-aligned.json"
    _write_episode(aligned)
    assert inspect_episode(aligned).accepted

    misaligned = tmp_path / "human-misaligned.json"
    _write_episode(misaligned)
    payload = json.loads(misaligned.read_text())
    payload["samples"][-1]["route_state"]["lateralErrorM"] = 2.1
    misaligned.write_text(json.dumps(payload))
    inspection = inspect_episode(misaligned)
    assert not inspection.accepted
    assert "final lateral error" in inspection.rejection_reason

    stopped = tmp_path / "human-stopped.json"
    _write_episode(stopped)
    payload = json.loads(stopped.read_text())
    for sample in payload["samples"][-21:]:
        sample["control_target"].update(targetSpeedMps=0.0, mode="stopped")
    stopped.write_text(json.dumps(payload))
    inspection = inspect_episode(stopped)
    assert not inspection.accepted
    assert inspection.stopped_frames == 21
    assert "stopped frames" in inspection.rejection_reason
