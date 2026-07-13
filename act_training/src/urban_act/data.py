from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

MANIFEST_FILES = ("nominal.jsonl", "recovery.jsonl")
SPLIT_SEEDS = {"train": 101, "validation": 211, "test": 307}


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    kind: str
    split: str
    task_id: str
    instruction: str
    directory: Path


@dataclass(frozen=True, slots=True)
class CriticalWindowConfig:
    activity_threshold: float
    steering_threshold: float
    startup_weight: float
    throttle_weight: float
    brake_weight: float
    turn_weight: float
    recovery_weight: float


def load_episode_records(dataset_root: str | Path, split: str) -> list[EpisodeRecord]:
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split: {split}")

    root = Path(dataset_root)
    records: list[EpisodeRecord] = []
    for manifest_name in MANIFEST_FILES:
        manifest_path = root / "manifests" / manifest_name
        for line in manifest_path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row["split"] != split:
                continue
            directory = root / "raw" / "accepted" / row["id"]
            for required in ("episode.json", "telemetry.jsonl", "front.mp4"):
                if not (directory / required).is_file():
                    raise FileNotFoundError(directory / required)
            records.append(
                EpisodeRecord(
                    episode_id=row["id"],
                    kind=row["kind"],
                    split=row["split"],
                    task_id=row["taskId"],
                    instruction=row["instruction"],
                    directory=directory,
                )
            )

    if not records:
        raise RuntimeError(f"No expert episodes found for split {split}")
    return records


def compute_state_statistics(records: Iterable[EpisodeRecord]) -> dict[str, list[float]]:
    count = 0
    total = np.zeros(4, dtype=np.float64)
    total_square = np.zeros(4, dtype=np.float64)
    for record in records:
        for line in (record.directory / "telemetry.jsonl").read_text().splitlines():
            if not line:
                continue
            state = np.asarray(json.loads(line)["observation"]["state"], dtype=np.float64)
            total += state
            total_square += state * state
            count += 1
    if count == 0:
        raise RuntimeError("Cannot compute state statistics from an empty dataset")
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 1e-8)
    return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist(), "count": count}


def compute_action_statistics(
    records: Iterable[EpisodeRecord],
    *,
    activity_threshold: float,
    positive_weight_cap: float,
) -> dict[str, Any]:
    count = 0
    active_count = np.zeros(2, dtype=np.int64)
    absolute_total = np.zeros(3, dtype=np.float64)
    for record in records:
        for line in (record.directory / "telemetry.jsonl").read_text().splitlines():
            if not line:
                continue
            action = np.asarray(json.loads(line)["action"], dtype=np.float64)
            active_count += action[:2] > activity_threshold
            absolute_total += np.abs(action)
            count += 1
    if count == 0:
        raise RuntimeError("Cannot compute action statistics from an empty dataset")
    inactive_count = count - active_count
    positive_weights = np.minimum(
        inactive_count / np.maximum(active_count, 1),
        positive_weight_cap,
    )
    return {
        "count": count,
        "activity_threshold": activity_threshold,
        "active_count": {
            "throttle": int(active_count[0]),
            "brake": int(active_count[1]),
        },
        "active_fraction": {
            "throttle": float(active_count[0] / count),
            "brake": float(active_count[1] / count),
        },
        "positive_weights": {
            "throttle": float(positive_weights[0]),
            "brake": float(positive_weights[1]),
        },
        "zero_baseline_mae": {
            "throttle": float(absolute_total[0] / count),
            "brake": float(absolute_total[1] / count),
            "steering": float(absolute_total[2] / count),
        },
    }


class UrbanEpisodeStream(IterableDataset[dict[str, Any]]):
    """Decode each MP4 once per epoch and emit frame-aligned future action chunks."""

    def __init__(
        self,
        records: list[EpisodeRecord],
        *,
        image_size: int,
        chunk_size: int,
        stride: int,
        shuffle: bool,
        shuffle_buffer: int,
        seed: int,
        critical_windows: CriticalWindowConfig | None = None,
    ) -> None:
        super().__init__()
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        self.records = records
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.stride = stride
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.critical_windows = critical_windows
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        rng = random.Random(self.seed + self.epoch * 10_007)

        records = list(self.records)
        if self.shuffle:
            rng.shuffle(records)
        records = records[worker_id::worker_count]

        samples = (sample for record in records for sample in self._episode_samples(record))
        if self.shuffle and self.shuffle_buffer > 1:
            yield from _buffered_shuffle(samples, rng, self.shuffle_buffer)
        else:
            yield from samples

    def _episode_samples(self, record: EpisodeRecord) -> Iterator[dict[str, Any]]:
        rows = [json.loads(line) for line in (record.directory / "telemetry.jsonl").read_text().splitlines() if line]
        actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
        states = np.asarray([row["observation"]["state"] for row in rows], dtype=np.float32)
        decoded = 0

        with av.open(str(record.directory / "front.mp4"), mode="r") as container:
            for decoded, frame in enumerate(container.decode(video=0), start=1):
                frame_index = decoded - 1
                if frame_index >= len(rows):
                    raise RuntimeError(f"Video has extra frames: {record.episode_id}")
                if frame_index % self.stride:
                    continue

                end = min(frame_index + self.chunk_size, len(rows))
                valid = end - frame_index
                action_chunk = np.zeros((self.chunk_size, actions.shape[1]), dtype=np.float32)
                action_mask = np.zeros(self.chunk_size, dtype=np.bool_)
                action_chunk[:valid] = actions[frame_index:end]
                action_mask[:valid] = True
                sample_weight = _critical_window_weight(
                    states[frame_index],
                    action_chunk[:valid],
                    record.kind,
                    self.critical_windows,
                )
                resized_frame = frame.reformat(
                    width=self.image_size,
                    height=self.image_size,
                    format="rgb24",
                )
                rgb = np.ascontiguousarray(resized_frame.to_ndarray())

                yield {
                    "image": torch.from_numpy(rgb).permute(2, 0, 1),
                    "state": torch.from_numpy(states[frame_index]),
                    "actions": torch.from_numpy(action_chunk),
                    "action_mask": torch.from_numpy(action_mask),
                    "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
                    "instruction": record.instruction,
                    "task_id": record.task_id,
                    "episode_kind": record.kind,
                    "episode_id": record.episode_id,
                    "frame_index": frame_index,
                }

        if decoded != len(rows):
            raise RuntimeError(
                f"Video/telemetry length mismatch for {record.episode_id}: video={decoded}, telemetry={len(rows)}"
            )


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_fields = ("image", "state", "actions", "action_mask", "sample_weight")
    batch: dict[str, Any] = {name: torch.stack([sample[name] for sample in samples]) for name in tensor_fields}
    for name in ("instruction", "task_id", "episode_kind", "episode_id", "frame_index"):
        batch[name] = [sample[name] for sample in samples]
    return batch


def make_dataloader(
    dataset: UrbanEpisodeStream,
    *,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
) -> DataLoader[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "collate_fn": collate_samples,
    }
    if num_workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _buffered_shuffle(
    samples: Iterable[dict[str, Any]], rng: random.Random, buffer_size: int
) -> Iterator[dict[str, Any]]:
    buffer: list[dict[str, Any]] = []
    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample
    rng.shuffle(buffer)
    yield from buffer


def _critical_window_weight(
    state: np.ndarray,
    actions: np.ndarray,
    episode_kind: str,
    config: CriticalWindowConfig | None,
) -> float:
    if config is None or len(actions) == 0:
        return 1.0
    weight = 1.0
    if abs(float(state[0])) < 0.5 and float(actions[0, 0]) > config.activity_threshold:
        weight = max(weight, config.startup_weight)
    if np.any(actions[:, 0] > config.activity_threshold):
        weight = max(weight, config.throttle_weight)
    if np.any(actions[:, 1] > config.activity_threshold):
        weight = max(weight, config.brake_weight)
    if np.any(np.abs(actions[:, 2]) > config.steering_threshold):
        weight = max(weight, config.turn_weight)
    if episode_kind == "recovery":
        weight = max(weight, config.recovery_weight)
    return weight
