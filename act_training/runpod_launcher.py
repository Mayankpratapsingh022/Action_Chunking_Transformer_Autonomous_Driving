from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = PROJECT_DIR / ".env"
DEFAULT_STATE_FILE = PROJECT_DIR / ".runpod" / "last_pod.json"
DEFAULT_API_BASE = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"
DEFAULT_GIT_REPO = (
    "https://github.com/Mayankpratapsingh022/Action_Chunking_Transformer_Autonomous_Driving.git"
)
FINAL_POD_STATES = {"EXITED", "TERMINATED"}

REMOTE_BOOTSTRAP = r"""set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates ffmpeg git
rm -rf /var/lib/apt/lists/*

: "${ACT_GIT_REPO:?ACT_GIT_REPO is required}"
: "${ACT_GIT_REF:?ACT_GIT_REF is required}"
: "${ACT_SOURCE_DIR:?ACT_SOURCE_DIR is required}"

mkdir -p "$(dirname "$ACT_SOURCE_DIR")"
if [[ -d "$ACT_SOURCE_DIR/.git" ]]; then
  git -C "$ACT_SOURCE_DIR" fetch --depth 1 origin "$ACT_GIT_REF"
  git -C "$ACT_SOURCE_DIR" checkout --detach FETCH_HEAD
elif [[ -e "$ACT_SOURCE_DIR" ]]; then
  echo "Refusing to replace non-Git path: $ACT_SOURCE_DIR" >&2
  exit 1
else
  git clone --depth 1 --single-branch --branch "$ACT_GIT_REF" "$ACT_GIT_REPO" "$ACT_SOURCE_DIR"
fi

exec bash "$ACT_SOURCE_DIR/act_training/scripts/runpod_bootstrap.sh"
"""


class RunPodAPIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RunPodClient:
    def __init__(self, api_key: str, *, api_base: str = DEFAULT_API_BASE, timeout: float = 60.0) -> None:
        if not api_key:
            raise ValueError("RUNPOD_API_KEY is required")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/pods", payload)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pods/{quote(pod_id, safe='')}")

    def start_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("POST", f"/pods/{quote(pod_id, safe='')}/start")

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("POST", f"/pods/{quote(pod_id, safe='')}/stop")

    def delete_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/pods/{quote(pod_id, safe='')}")

    def list_network_volumes(self) -> list[dict[str, Any]]:
        result = self._request("GET", "/networkvolumes")
        if not isinstance(result, list):
            raise RunPodAPIError("RunPod returned an invalid network-volume list")
        return result

    def create_network_volume(self, *, name: str, size: int, data_center_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/networkvolumes",
            {"name": name, "size": size, "dataCenterId": data_center_id},
        )

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.api_base}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            detail = _api_error_detail(response_body) or error.reason
            raise RunPodAPIError(f"RunPod API returned HTTP {error.code}: {detail}", status=error.code) from error
        except URLError as error:
            raise RunPodAPIError(f"Could not reach RunPod API: {error.reason}") from error

        if not response_body:
            return {}
        try:
            return json.loads(response_body)
        except json.JSONDecodeError as error:
            if method in {"POST", "DELETE"} and payload is None:
                return {"message": response_body.strip()}
            raise RunPodAPIError("RunPod API returned non-JSON data") from error


@dataclass(frozen=True, slots=True)
class LaunchSettings:
    run_name: str
    gpu_types: tuple[str, ...]
    network_volume_id: str
    hf_secret_name: str
    image: str
    git_repo: str
    git_ref: str
    max_steps: int = 10_000
    batch_size: int = 64
    num_workers: int = 4
    container_disk_gb: int = 30
    min_vcpu_per_gpu: int = 8
    min_ram_per_gpu: int = 32
    timeout_seconds: int = 43_200
    interruptible: bool = False

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.run_name):
            raise ValueError("run_name may contain only letters, numbers, dots, underscores, and hyphens")
        if not self.gpu_types or any(not gpu.strip() for gpu in self.gpu_types):
            raise ValueError("At least one RunPod GPU type ID is required")
        if not self.network_volume_id:
            raise ValueError("RUNPOD_NETWORK_VOLUME_ID is required for resumable training")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.hf_secret_name):
            raise ValueError("RunPod secret names may contain only letters, numbers, underscores, and hyphens")
        if not self.git_repo.startswith("https://"):
            raise ValueError("The automated bootstrap requires a public HTTPS Git repository")
        if not self.git_ref or not self.image:
            raise ValueError("git_ref and image must not be empty")
        if self.git_ref.startswith("-") or not re.fullmatch(r"[A-Za-z0-9._/-]+", self.git_ref):
            raise ValueError("git_ref contains unsupported characters")
        for name, value in (
            ("max_steps", self.max_steps),
            ("batch_size", self.batch_size),
            ("num_workers", self.num_workers),
            ("container_disk_gb", self.container_disk_gb),
            ("min_vcpu_per_gpu", self.min_vcpu_per_gpu),
            ("min_ram_per_gpu", self.min_ram_per_gpu),
            ("timeout_seconds", self.timeout_seconds),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")


def build_pod_payload(settings: LaunchSettings) -> dict[str, Any]:
    settings.validate()
    workspace_root = "/workspace/act-driving"
    source_dir = f"{workspace_root}/source"
    return {
        "name": f"act-{settings.run_name}",
        "computeType": "GPU",
        "cloudType": "SECURE",
        "gpuTypeIds": list(settings.gpu_types),
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "imageName": settings.image,
        "containerDiskInGb": settings.container_disk_gb,
        "networkVolumeId": settings.network_volume_id,
        "volumeMountPath": "/workspace",
        "interruptible": settings.interruptible,
        "minVCPUPerGPU": settings.min_vcpu_per_gpu,
        "minRAMPerGPU": settings.min_ram_per_gpu,
        "ports": [],
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [REMOTE_BOOTSTRAP],
        "env": {
            "HF_TOKEN": f"{{{{ RUNPOD_SECRET_{settings.hf_secret_name} }}}}",
            "ACT_WORKSPACE_ROOT": workspace_root,
            "ACT_SOURCE_DIR": source_dir,
            "ACT_GIT_REPO": settings.git_repo,
            "ACT_GIT_REF": settings.git_ref,
            "ACT_RUN_NAME": settings.run_name,
            "ACT_MAX_STEPS": str(settings.max_steps),
            "ACT_BATCH_SIZE": str(settings.batch_size),
            "ACT_NUM_WORKERS": str(settings.num_workers),
            "ACT_TIMEOUT_SECONDS": str(settings.timeout_seconds),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line_number, raw_line in enumerate(env_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE")
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid .env key on line {line_number}: {key}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and manage RunPod ACT training Pods.")
    parser.add_argument("--env-file", default=os.environ.get("ACT_ENV_FILE", str(DEFAULT_ENV_FILE)))
    parser.add_argument("--state-file", default=os.environ.get("RUNPOD_STATE_FILE", str(DEFAULT_STATE_FILE)))
    commands = parser.add_subparsers(dest="command", required=True)

    launch = commands.add_parser("launch", help="Create a billable GPU Pod and start training.")
    launch.add_argument("--run-name", default=os.environ.get("ACT_RUN_NAME", "act-driving-v2"))
    launch.add_argument("--gpu-type", action="append", dest="gpu_types")
    launch.add_argument("--network-volume-id", default=os.environ.get("RUNPOD_NETWORK_VOLUME_ID", ""))
    launch.add_argument("--hf-secret-name", default=os.environ.get("RUNPOD_HF_SECRET_NAME", "huggingface_token"))
    launch.add_argument("--image", default=os.environ.get("RUNPOD_IMAGE", DEFAULT_IMAGE))
    launch.add_argument("--git-repo", default=os.environ.get("RUNPOD_GIT_REPO", DEFAULT_GIT_REPO))
    launch.add_argument("--git-ref", default=os.environ.get("RUNPOD_GIT_REF", "main"))
    launch.add_argument("--max-steps", type=int, default=10_000)
    launch.add_argument("--batch-size", type=int, default=64)
    launch.add_argument("--num-workers", type=int, default=4)
    launch.add_argument("--container-disk-gb", type=int, default=30)
    launch.add_argument("--min-vcpu-per-gpu", type=int, default=8)
    launch.add_argument("--min-ram-per-gpu", type=int, default=32)
    launch.add_argument("--timeout-hours", type=float, default=12.0)
    launch.add_argument("--spot", action="store_true", help="Use interruptible capacity. Not recommended initially.")
    launch.add_argument("--allow-duplicate", action="store_true")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--yes", action="store_true", help="Confirm creation of a billable Pod.")

    for command in ("status", "watch", "start", "stop", "logs"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--pod-id")
        if command == "watch":
            subparser.add_argument("--interval", type=float, default=15.0)
        elif command == "logs":
            subparser.add_argument("--run-name", default=os.environ.get("ACT_RUN_NAME", "act-driving-v2"))

    terminate = commands.add_parser("terminate", help="Permanently delete a Pod, preserving its network volume.")
    terminate.add_argument("--pod-id")
    terminate.add_argument("--yes", action="store_true")

    create_volume = commands.add_parser("create-volume", help="Create persistent storage for checkpoints.")
    create_volume.add_argument("--name", default="act-driving-training")
    create_volume.add_argument("--size", type=int, default=50)
    create_volume.add_argument("--data-center-id", required=True)
    create_volume.add_argument("--yes", action="store_true")

    commands.add_parser("list-volumes")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", default=os.environ.get("ACT_ENV_FILE", str(DEFAULT_ENV_FILE)))
    known, _ = env_parser.parse_known_args(arguments)
    load_env_file(known.env_file)

    parser = build_parser()
    args = parser.parse_args(arguments)
    state_path = Path(args.state_file)

    try:
        if args.command == "launch":
            _launch(args, state_path)
        elif args.command == "create-volume":
            _create_volume(args)
        elif args.command == "list-volumes":
            _print_json(_client().list_network_volumes())
        else:
            pod_id = _resolve_pod_id(args.pod_id, state_path)
            _run_pod_command(args, pod_id)
    except (RunPodAPIError, ValueError) as error:
        parser.error(str(error))


def _launch(args: argparse.Namespace, state_path: Path) -> None:
    env_gpu_types = tuple(
        gpu.strip() for gpu in os.environ.get("RUNPOD_GPU_TYPE_IDS", "").split("|") if gpu.strip()
    )
    if args.timeout_hours <= 0:
        raise ValueError("timeout-hours must be positive")
    settings = LaunchSettings(
        run_name=args.run_name,
        gpu_types=tuple(args.gpu_types or env_gpu_types or (DEFAULT_GPU_TYPE,)),
        network_volume_id=args.network_volume_id,
        hf_secret_name=args.hf_secret_name,
        image=args.image,
        git_repo=args.git_repo,
        git_ref=args.git_ref,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        container_disk_gb=args.container_disk_gb,
        min_vcpu_per_gpu=args.min_vcpu_per_gpu,
        min_ram_per_gpu=args.min_ram_per_gpu,
        timeout_seconds=round(args.timeout_hours * 3_600),
        interruptible=args.spot,
    )
    payload = build_pod_payload(settings)
    if args.dry_run:
        _print_json(payload)
        return
    if not args.yes:
        raise ValueError("launch creates billable GPU compute; inspect --dry-run, then repeat with --yes")

    client = _client()
    if not args.allow_duplicate:
        _ensure_no_active_pod(client, state_path)
    response = client.create_pod(payload)
    pod_id = str(response.get("id", ""))
    if not pod_id:
        raise RunPodAPIError("RunPod created a Pod but did not return its ID")
    state = {
        "pod_id": pod_id,
        "run_name": settings.run_name,
        "network_volume_id": settings.network_volume_id,
        "gpu_types": list(settings.gpu_types),
        "created_at": _timestamp(),
        "cost_per_hour": response.get("adjustedCostPerHr", response.get("costPerHr")),
    }
    _write_state(state_path, state)
    print(f"Created RunPod Pod: {pod_id}")
    print(f"GPU: {', '.join(settings.gpu_types)}")
    if state["cost_per_hour"] is not None:
        print(f"Reported cost: {state['cost_per_hour']} credits/hour")
    print(f"Dashboard: {_dashboard_url(pod_id)}")
    print(f"Watch: python runpod_launcher.py watch --pod-id {pod_id}")


def _run_pod_command(args: argparse.Namespace, pod_id: str) -> None:
    if args.command == "logs":
        print(f"Container logs: {_dashboard_url(pod_id)}")
        print(f"Persistent log: /workspace/act-driving/logs/{args.run_name}.log")
        return

    client = _client()
    if args.command == "status":
        _print_pod_status(client.get_pod(pod_id))
    elif args.command == "watch":
        if args.interval < 1:
            raise ValueError("watch interval must be at least one second")
        _watch(client, pod_id, args.interval)
    elif args.command == "start":
        client.start_pod(pod_id)
        print(f"Start requested for Pod {pod_id}")
    elif args.command == "stop":
        client.stop_pod(pod_id)
        print(f"Stop requested for Pod {pod_id}")
    elif args.command == "terminate":
        if not args.yes:
            raise ValueError("terminate permanently deletes the Pod; repeat with --yes")
        client.delete_pod(pod_id)
        print(f"Terminated Pod {pod_id}. An attached network volume remains available.")


def _create_volume(args: argparse.Namespace) -> None:
    if args.size < 1:
        raise ValueError("network-volume size must be positive")
    if not args.yes:
        raise ValueError("network volumes incur storage charges; repeat with --yes")
    result = _client().create_network_volume(
        name=args.name,
        size=args.size,
        data_center_id=args.data_center_id,
    )
    _print_json(result)
    if result.get("id"):
        print(f"Add this to .env: RUNPOD_NETWORK_VOLUME_ID={result['id']}")


def _watch(client: RunPodClient, pod_id: str, interval: float) -> None:
    while True:
        pod = client.get_pod(pod_id)
        status = str(pod.get("desiredStatus", "UNKNOWN"))
        cost = pod.get("adjustedCostPerHr", pod.get("costPerHr"))
        print(f"{_timestamp()}  {status:10}  cost/hour={cost or 'unknown'}", flush=True)
        if status in FINAL_POD_STATES:
            return
        time.sleep(interval)


def _ensure_no_active_pod(client: RunPodClient, state_path: Path) -> None:
    if not state_path.is_file():
        return
    state = json.loads(state_path.read_text())
    pod_id = state.get("pod_id")
    if not pod_id:
        return
    try:
        pod = client.get_pod(str(pod_id))
    except RunPodAPIError as error:
        if error.status == 404:
            return
        raise
    if pod.get("desiredStatus") not in FINAL_POD_STATES:
        raise ValueError(
            f"Pod {pod_id} is still {pod.get('desiredStatus', 'active')}; stop or terminate it, "
            "or pass --allow-duplicate"
        )


def _resolve_pod_id(explicit_pod_id: str | None, state_path: Path) -> str:
    if explicit_pod_id:
        return explicit_pod_id
    if not state_path.is_file():
        raise ValueError("No saved Pod state. Supply --pod-id explicitly.")
    state = json.loads(state_path.read_text())
    pod_id = state.get("pod_id")
    if not pod_id:
        raise ValueError(f"No pod_id found in {state_path}")
    return str(pod_id)


def _client() -> RunPodClient:
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY is empty; add it to act_training/.env")
    return RunPodClient(api_key)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _print_pod_status(pod: dict[str, Any]) -> None:
    gpu = pod.get("gpu") or {}
    _print_json(
        {
            "id": pod.get("id"),
            "name": pod.get("name"),
            "status": pod.get("desiredStatus"),
            "gpu": gpu.get("displayName", gpu.get("id")),
            "gpu_count": gpu.get("count"),
            "cost_per_hour": pod.get("adjustedCostPerHr", pod.get("costPerHr")),
            "last_status_change": pod.get("lastStatusChange"),
            "network_volume": pod.get("networkVolume"),
        }
    )


def _api_error_detail(body: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()[:500]
    if isinstance(parsed, dict):
        for key in ("error", "message", "detail"):
            if parsed.get(key):
                return str(parsed[key])[:500]
    return str(parsed)[:500]


def _dashboard_url(pod_id: str) -> str:
    return f"https://www.runpod.io/console/pods?pod={quote(pod_id, safe='')}"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
