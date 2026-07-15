from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from left_turn_vla.constants import (
    ACTION_KEY,
    CAMERA_KEY,
    CAPTURE_FPS,
    IMAGE_SIZE,
    LEFT_TURN_INSTRUCTION,
    LEFT_TURN_INTENT_ID,
    MAX_SPEED_MPS,
    STATE_KEY,
)


@dataclass(slots=True)
class EpisodeInspection:
    source: str
    seed: int | None
    frames: int
    max_route_progress: float
    final_lateral_error_m: float
    final_heading_error_rad: float
    stopped_frames: int
    collision: bool
    off_route: bool
    red_light_violation: bool
    exact_duplicate: bool = False
    accepted: bool = False
    rejection_reason: str | None = None
    fingerprint: str = ""

    @property
    def has_safety_event(self) -> bool:
        return self.collision or self.off_route or self.red_light_violation

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("fingerprint")
        return payload


def discover_independent_episodes(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return sorted(path for path in root.glob("human-*.json") if path.is_file())


def load_episode(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {source.name}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"{source.name} is not a simulator episode export")
    return payload


def inspect_episode(
    path: str | Path,
    *,
    min_frames: int = 60,
    min_progress: float = 0.95,
    max_final_lateral_error_m: float = 2.0,
    max_final_heading_error_rad: float = 0.15,
    max_stopped_frames: int = 20,
) -> EpisodeInspection:
    source = Path(path)
    payload = load_episode(source)
    samples = payload["samples"]
    seed = samples[0].get("seed") if samples else None
    progress = [float(sample.get("task", {}).get("routeProgress", 0.0)) for sample in samples]
    collision = any(bool(sample.get("events", {}).get("collision")) for sample in samples)
    off_route = any(bool(sample.get("events", {}).get("offRoute")) for sample in samples)
    red_light = any(bool(sample.get("events", {}).get("redLightViolation")) for sample in samples)
    final_route_state = samples[-1].get("route_state", {}) if samples else {}
    try:
        final_lateral_error_m = abs(float(final_route_state.get("lateralErrorM", math.inf)))
        final_heading_error_rad = abs(float(final_route_state.get("headingErrorRad", math.inf)))
    except (TypeError, ValueError):
        final_lateral_error_m = math.inf
        final_heading_error_rad = math.inf
    stopped_frames = sum(
        _target_speed_mps(sample) <= 0.05
        for sample in samples
    )

    reasons: list[str] = []
    if payload.get("metadata", {}).get("schema_version") != "vla-urban-4":
        reasons.append("requires vla-urban-4 target-control schema")
    if payload.get("metadata", {}).get("action_encoding") != "target_speed_steering":
        reasons.append("requires target_speed_steering action encoding")
    if len(samples) < min_frames:
        reasons.append(f"fewer than {min_frames} frames")
    if max(progress, default=0.0) < min_progress:
        reasons.append(f"route progress below {min_progress:.2f}")
    if final_lateral_error_m > max_final_lateral_error_m:
        reasons.append(f"final lateral error above {max_final_lateral_error_m:.2f} m")
    if final_heading_error_rad > max_final_heading_error_rad:
        reasons.append(f"final heading error above {max_final_heading_error_rad:.3f} rad")
    if stopped_frames > max_stopped_frames:
        reasons.append(f"more than {max_stopped_frames} stopped frames")
    if samples:
        bad_intents = [
            sample
            for sample in samples
            if sample.get("language_id") != LEFT_TURN_INTENT_ID
            or sample.get("language_text") != LEFT_TURN_INSTRUCTION
        ]
        if bad_intents:
            reasons.append("contains a non-left-turn instruction")
        if any(sample.get("capture_resolution") != IMAGE_SIZE for sample in samples):
            reasons.append(f"contains images other than {IMAGE_SIZE} x {IMAGE_SIZE}")
        if any(not _valid_sample(sample) for sample in samples):
            reasons.append("contains an invalid observation or action")
    else:
        reasons.append("contains no frames")

    return EpisodeInspection(
        source=source.name,
        seed=int(seed) if isinstance(seed, (int, float)) else None,
        frames=len(samples),
        max_route_progress=max(progress, default=0.0),
        final_lateral_error_m=final_lateral_error_m,
        final_heading_error_rad=final_heading_error_rad,
        stopped_frames=stopped_frames,
        collision=collision,
        off_route=off_route,
        red_light_violation=red_light,
        accepted=not reasons and not (collision or off_route or red_light),
        rejection_reason="; ".join(reasons) if reasons else None,
        fingerprint=_episode_fingerprint(samples),
    )


def analyze_directory(
    input_dir: str | Path,
    *,
    include_recovery: bool = False,
    min_frames: int = 60,
    min_progress: float = 0.95,
    max_final_lateral_error_m: float = 2.0,
    max_final_heading_error_rad: float = 0.15,
    max_stopped_frames: int = 20,
    shuffle_seed: int = 42,
    eval_split: float = 0.1,
) -> tuple[list[EpisodeInspection], dict[str, Any]]:
    if not 0 < eval_split < 0.5:
        raise ValueError("eval_split must be greater than 0 and less than 0.5")
    root = Path(input_dir).expanduser().resolve()
    files = discover_independent_episodes(root)
    inspections: list[EpisodeInspection] = []
    seen: set[str] = set()

    for path in files:
        try:
            item = inspect_episode(
                path,
                min_frames=min_frames,
                min_progress=min_progress,
                max_final_lateral_error_m=max_final_lateral_error_m,
                max_final_heading_error_rad=max_final_heading_error_rad,
                max_stopped_frames=max_stopped_frames,
            )
        except ValueError as error:
            item = EpisodeInspection(
                source=path.name,
                seed=None,
                frames=0,
                max_route_progress=0.0,
                final_lateral_error_m=math.inf,
                final_heading_error_rad=math.inf,
                stopped_frames=0,
                collision=False,
                off_route=False,
                red_light_violation=False,
                accepted=False,
                rejection_reason=str(error),
            )
        if item.fingerprint and item.fingerprint in seen:
            item.exact_duplicate = True
            item.accepted = False
            item.rejection_reason = "exact duplicate episode"
        elif item.fingerprint:
            seen.add(item.fingerprint)
        if item.has_safety_event and include_recovery and item.rejection_reason is None and not item.exact_duplicate:
            item.accepted = True
        elif item.has_safety_event and item.rejection_reason is None:
            item.rejection_reason = "collision/off-route episode excluded from nominal imitation"
        inspections.append(item)

    accepted = [item for item in inspections if item.accepted]
    accepted, held_out = _seed_disjoint_order(accepted, eval_split=eval_split, shuffle_seed=shuffle_seed)
    rejected = [item for item in inspections if not item.accepted]
    ordered = accepted + rejected
    cumulative = len(list(root.glob("vla_urban_dataset_*.json")))
    report = {
        "input_dir": str(root),
        "instruction": LEFT_TURN_INSTRUCTION,
        "independent_files": len(files),
        "old_cumulative_exports_ignored": cumulative,
        "accepted_episodes": len(accepted),
        "rejected_episodes": len(rejected),
        "accepted_frames": sum(item.frames for item in accepted),
        "unique_seeds": len({item.seed for item in accepted if item.seed is not None}),
        "include_recovery": include_recovery,
        "min_frames": min_frames,
        "min_progress": min_progress,
        "max_final_lateral_error_m": max_final_lateral_error_m,
        "max_final_heading_error_rad": max_final_heading_error_rad,
        "max_stopped_frames": max_stopped_frames,
        "shuffle_seed": shuffle_seed,
        "eval_split": eval_split,
        "held_out_episodes": len(held_out),
        "held_out_seeds": sorted({item.seed for item in held_out if item.seed is not None}),
        "seed_disjoint_holdout": not (
            {item.seed for item in accepted[:-len(held_out)] if item.seed is not None}
            & {item.seed for item in held_out if item.seed is not None}
        ),
        "episodes": [item.public_dict() for item in ordered],
    }
    return ordered, report


def iter_lerobot_frames(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    previous_target_speed = 0.0
    previous_target_steering = 0.0
    for sample in payload["samples"]:
        control_target = sample["control_target"]
        ego = sample["ego"]
        yield {
            CAMERA_KEY: decode_image_data_url(sample["image"]),
            STATE_KEY: np.asarray(
                [
                    float(ego["speed"]) * MAX_SPEED_MPS,
                    float(ego["steering"]),
                    previous_target_speed,
                    previous_target_steering,
                ],
                dtype=np.float32,
            ),
            ACTION_KEY: np.asarray(
                [
                    float(control_target["targetSpeedMps"]),
                    float(control_target["targetSteer"]),
                ],
                dtype=np.float32,
            ),
            "task": LEFT_TURN_INSTRUCTION,
        }
        previous_target_speed = float(control_target["targetSpeedMps"])
        previous_target_steering = float(control_target["targetSteer"])


def decode_image_data_url(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("image must be a base64 data URL")
    encoded = value.split(",", maxsplit=1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            if rgb.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"expected {IMAGE_SIZE} x {IMAGE_SIZE}, got {rgb.width} x {rgb.height}")
            return np.asarray(rgb, dtype=np.uint8)
    except (binascii.Error, UnidentifiedImageError, OSError) as error:
        raise ValueError("image data is not a valid PNG or JPEG") from error


def accepted_sources(inspections: Iterable[EpisodeInspection]) -> list[str]:
    return [item.source for item in inspections if item.accepted]


def _seed_disjoint_order(
    accepted: list[EpisodeInspection], *, eval_split: float, shuffle_seed: int
) -> tuple[list[EpisodeInspection], list[EpisodeInspection]]:
    if len(accepted) < 2:
        return accepted, []
    target = math.ceil(len(accepted) * eval_split)
    groups: dict[tuple[str, int | str], list[EpisodeInspection]] = {}
    for item in accepted:
        key: tuple[str, int | str] = ("seed", item.seed) if item.seed is not None else ("source", item.source)
        groups.setdefault(key, []).append(item)

    rng = random.Random(shuffle_seed)
    keys = list(groups)
    rng.shuffle(keys)
    subsets: dict[int, tuple[tuple[str, int | str], ...]] = {0: ()}
    for key in keys:
        size = len(groups[key])
        for count, chosen in sorted(tuple(subsets.items()), reverse=True):
            next_count = count + size
            if next_count <= target and next_count not in subsets:
                subsets[next_count] = (*chosen, key)
    if target not in subsets:
        raise RuntimeError(
            f"Could not form a seed-disjoint held-out set of exactly {target} episodes; "
            "change --eval-split or remove duplicate-seed recordings"
        )

    held_out_keys = set(subsets[target])
    train: list[EpisodeInspection] = []
    held_out: list[EpisodeInspection] = []
    for key in keys:
        group = groups[key]
        rng.shuffle(group)
        (held_out if key in held_out_keys else train).extend(group)
    return train + held_out, held_out


def _episode_fingerprint(samples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(str(sample.get("image", "")).encode())
        digest.update(json.dumps(sample.get("control_target", {}), sort_keys=True, separators=(",", ":")).encode())
        digest.update(json.dumps(sample.get("ego", {}), sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest() if samples else ""


def _target_speed_mps(sample: dict[str, Any]) -> float:
    try:
        return float(sample.get("control_target", {}).get("targetSpeedMps", math.inf))
    except (TypeError, ValueError):
        return math.inf


def _valid_sample(sample: dict[str, Any]) -> bool:
    try:
        image = sample["image"]
        control = sample["control"]
        control_target = sample["control_target"]
        ego = sample["ego"]
        route_state = sample["route_state"]
        values = (
            float(control["throttle"]),
            float(control["brake"]),
            float(control["steer"]),
            float(ego["speed"]),
            float(ego["steering"]),
            float(control_target["targetSpeedMps"]),
            float(control_target["targetSteer"]),
            float(route_state["progress"]),
            float(route_state["lateralErrorM"]),
            float(route_state["headingErrorRad"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        isinstance(image, str)
        and image.startswith("data:image/")
        and all(np.isfinite(value) for value in values)
        and 0.0 <= values[0] <= 1.0
        and 0.0 <= values[1] <= 1.0
        and -1.0 <= values[2] <= 1.0
        and 0.0 <= values[5] <= MAX_SPEED_MPS
        and -1.0 <= values[6] <= 1.0
        and 0.0 <= values[7] <= 1.0
        and control_target.get("mode")
        in {"stopped", "cruise", "turn_left", "turn_right", "straighten", "emergency_stop"}
    )


def lerobot_features() -> dict[str, dict[str, Any]]:
    return {
        CAMERA_KEY: {
            "dtype": "video",
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (4,),
            "names": {
                "vehicle": [
                    "speed_mps",
                    "steering",
                    "previous_target_speed_mps",
                    "previous_target_steering",
                ]
            },
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (2,),
            "names": {"driving": ["target_speed_mps", "target_steering"]},
        },
    }


__all__ = [
    "CAPTURE_FPS",
    "EpisodeInspection",
    "accepted_sources",
    "analyze_directory",
    "decode_image_data_url",
    "discover_independent_episodes",
    "inspect_episode",
    "iter_lerobot_frames",
    "lerobot_features",
    "load_episode",
]
