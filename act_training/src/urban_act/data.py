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


class UrbanEpisodeStream(IterableDataset[dict[str, Any]]):
    """Decode each MP4 once per epoch and emit frame-aligned future action chunks."""

    def __init__(
        self,
        records: list[EpisodeRecord],
        *,
        chunk_size: int,
        stride: int,
        shuffle: bool,
        shuffle_buffer: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.records = records
        self.chunk_size = chunk_size
        self.stride = stride
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
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
        rows = [
            json.loads(line)
            for line in (record.directory / "telemetry.jsonl").read_text().splitlines()
            if line
        ]
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
                rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))

                yield {
                    "image": torch.from_numpy(rgb).permute(2, 0, 1),
                    "state": torch.from_numpy(states[frame_index]),
                    "actions": torch.from_numpy(action_chunk),
                    "action_mask": torch.from_numpy(action_mask),
                    "instruction": record.instruction,
                    "task_id": record.task_id,
                    "episode_id": record.episode_id,
                    "frame_index": frame_index,
                }

        if decoded != len(rows):
            raise RuntimeError(
                f"Video/telemetry length mismatch for {record.episode_id}: video={decoded}, telemetry={len(rows)}"
            )


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_fields = ("image", "state", "actions", "action_mask")
    batch: dict[str, Any] = {name: torch.stack([sample[name] for sample in samples]) for name in tensor_fields}
    for name in ("instruction", "task_id", "episode_id", "frame_index"):
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
