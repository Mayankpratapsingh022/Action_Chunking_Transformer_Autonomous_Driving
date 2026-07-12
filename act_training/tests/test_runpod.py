from __future__ import annotations

import json
from pathlib import Path

import pytest

from runpod_launcher import (
    DEFAULT_IMAGE,
    LaunchSettings,
    RunPodAPIError,
    _ensure_no_active_pod,
    _resolve_pod_id,
    build_pod_payload,
    load_env_file,
)
from runpod_train import load_config
from scripts.download_hf_run import select_run_files


def _settings(**overrides: object) -> LaunchSettings:
    values = {
        "run_name": "act-driving-v1",
        "gpu_types": ("NVIDIA H100 80GB HBM3",),
        "network_volume_id": "volume-123",
        "hf_secret_name": "huggingface_token",
        "image": DEFAULT_IMAGE,
        "git_repo": "https://github.com/example/act-driving.git",
        "git_ref": "main",
    }
    values.update(overrides)
    return LaunchSettings(**values)


def test_runpod_payload_is_resumable_and_does_not_contain_secret_values() -> None:
    payload = build_pod_payload(_settings())

    assert payload["networkVolumeId"] == "volume-123"
    assert payload["volumeMountPath"] == "/workspace"
    assert payload["cloudType"] == "SECURE"
    assert payload["interruptible"] is False
    assert payload["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3"]
    assert payload["env"]["HF_TOKEN"] == "{{ RUNPOD_SECRET_huggingface_token }}"
    assert payload["env"]["ACT_BATCH_SIZE"] == "64"
    assert payload["env"]["ACT_MAX_STEPS"] == "10000"
    assert payload["env"]["ACT_TIMEOUT_SECONDS"] == "43200"
    assert "RUNPOD_API_KEY" not in json.dumps(payload)
    assert payload["dockerEntrypoint"] == ["/bin/bash", "-lc"]


def test_runpod_payload_requires_persistent_network_volume() -> None:
    with pytest.raises(ValueError, match="NETWORK_VOLUME"):
        build_pod_payload(_settings(network_volume_id=""))


def test_dotenv_loader_preserves_explicit_process_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RUNPOD_API_KEY=file-key\nRUNPOD_NETWORK_VOLUME_ID='volume-from-file'\n")
    monkeypatch.setenv("RUNPOD_API_KEY", "process-key")
    monkeypatch.delenv("RUNPOD_NETWORK_VOLUME_ID", raising=False)

    load_env_file(env_file)

    assert __import__("os").environ["RUNPOD_API_KEY"] == "process-key"
    assert __import__("os").environ["RUNPOD_NETWORK_VOLUME_ID"] == "volume-from-file"


def test_saved_pod_id_is_used_when_cli_id_is_absent(tmp_path: Path) -> None:
    state_path = tmp_path / "last_pod.json"
    state_path.write_text(json.dumps({"pod_id": "pod-123"}))

    assert _resolve_pod_id(None, state_path) == "pod-123"
    assert _resolve_pod_id("explicit-pod", state_path) == "explicit-pod"


def test_active_pod_prevents_accidental_duplicate_launch(tmp_path: Path) -> None:
    state_path = tmp_path / "last_pod.json"
    state_path.write_text(json.dumps({"pod_id": "pod-123"}))

    class Client:
        def get_pod(self, pod_id: str) -> dict[str, str]:
            assert pod_id == "pod-123"
            return {"desiredStatus": "RUNNING"}

    with pytest.raises(ValueError, match="still RUNNING"):
        _ensure_no_active_pod(Client(), state_path)  # type: ignore[arg-type]


def test_missing_previous_pod_does_not_block_launch(tmp_path: Path) -> None:
    state_path = tmp_path / "last_pod.json"
    state_path.write_text(json.dumps({"pod_id": "deleted-pod"}))

    class Client:
        def get_pod(self, pod_id: str) -> dict[str, str]:
            raise RunPodAPIError("not found", status=404)

    _ensure_no_active_pod(Client(), state_path)  # type: ignore[arg-type]


def test_runpod_training_config_uses_workspace_paths_and_cli_overrides() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "base.json",
        run_name="runpod-test",
        max_steps=25,
        batch_size=8,
        artifact_root="/workspace/act-driving/artifacts/runs",
        cache_dir="/workspace/act-driving/cache/huggingface",
    )

    assert config.run_name == "runpod-test"
    assert config.max_steps == 25
    assert config.batch_size == 8
    assert config.artifact_root.startswith("/workspace/")
    assert config.cache_dir.startswith("/workspace/")


def test_hugging_face_download_selects_only_requested_run() -> None:
    files = [
        "model.safetensors",
        "runs/act-driving-v1/model.safetensors",
        "runs/act-driving-v1/plots/training_curves.png",
        "runs/another-run/model.safetensors",
    ]

    assert select_run_files(files, "act-driving-v1") == [
        "runs/act-driving-v1/model.safetensors",
        "runs/act-driving-v1/plots/training_curves.png",
    ]
