from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE_ROOT = Path(os.environ.get("ACT_WORKSPACE_ROOT", "/workspace/act-driving"))
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
REQUIRED_TORCH_PREFIX = "2.8."
REQUIRED_TORCHVISION = "0.23.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a manually created RunPod machine and train the ACT driving policy."
    )
    parser.add_argument("--run-name", default="act-driving-v2")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--env-file", default=str(PROJECT_DIR / ".env"))
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--skip-setup", action="store_true", help="Skip pip installation on later runs.")
    parser.add_argument(
        "--install-pytorch",
        action="store_true",
        help="Install pinned CUDA PyTorch when the selected Pod image does not already provide Torch 2.8.",
    )
    parser.add_argument("--no-push-to-hub", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    load_env_file(args.env_file)
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    environment = build_runtime_environment(workspace_root)
    training_command = build_training_command(args, workspace_root)
    log_path = workspace_root / "logs" / f"{args.run_name}.log"

    summary = {
        "run_name": args.run_name,
        "workspace_root": str(workspace_root),
        "artifact_root": str(workspace_root / "artifacts" / "runs"),
        "cache_dir": str(workspace_root / "cache" / "huggingface"),
        "log_path": str(log_path),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "resume": args.resume,
        "push_to_hub": not args.no_push_to_hub,
        "timeout_hours": args.timeout_hours,
        "command": training_command,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return

    if not args.no_push_to_hub and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required. Add it to the RunPod environment or act_training/.env.")
    python_version = tuple(int(part) for part in platform.python_version_tuple()[:2])
    if python_version < (3, 11):
        raise RuntimeError(f"Python 3.11 or newer is required; found {sys.version.split()[0]}")

    workspace_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_setup:
        prepare_environment(install_pytorch=args.install_pytorch)
    verify_cuda(environment)

    timeout_seconds = round(args.timeout_hours * 3_600)
    return_code = stream_process(
        training_command,
        environment=environment,
        log_path=log_path,
        timeout_seconds=timeout_seconds,
    )
    if return_code == 124:
        raise RuntimeError(f"Training exceeded the {args.timeout_hours:g}-hour safety timeout")
    if return_code:
        raise RuntimeError(f"Training exited with code {return_code}. See {log_path}")
    print(f"Training completed. Artifacts: {workspace_root / 'artifacts' / 'runs' / args.run_name}")


def prepare_environment(*, install_pytorch: bool) -> None:
    torch_version = installed_distribution_version("torch")
    if install_pytorch:
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--index-url",
                PYTORCH_INDEX_URL,
                "torch==2.8.0",
                f"torchvision=={REQUIRED_TORCHVISION}",
            ]
        )
    elif torch_version is None or not torch_version.startswith(REQUIRED_TORCH_PREFIX):
        found = torch_version or "not installed"
        raise RuntimeError(
            f"Expected CUDA Torch {REQUIRED_TORCH_PREFIX}x, found {found}. "
            "Use a PyTorch 2.8 / CUDA 12.8 Pod image or add --install-pytorch."
        )
    else:
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--index-url",
                PYTORCH_INDEX_URL,
                f"torchvision=={REQUIRED_TORCHVISION}",
            ]
        )

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(PROJECT_DIR / "requirements-runpod.txt"),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(PROJECT_DIR),
        ]
    )


def build_training_command(args: argparse.Namespace, workspace_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_DIR / "runpod_train.py"),
        "--config",
        str(PROJECT_DIR / "configs" / "base.json"),
        "--run-name",
        args.run_name,
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--resume",
        args.resume,
        "--artifact-root",
        str(workspace_root / "artifacts" / "runs"),
        "--cache-dir",
        str(workspace_root / "cache" / "huggingface"),
    ]
    if args.no_push_to_hub:
        command.append("--no-push-to-hub")
    return command


def build_runtime_environment(workspace_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "ACT_WORKSPACE_ROOT": str(workspace_root),
            "HF_HOME": str(workspace_root / "cache" / "huggingface"),
            "TORCH_HOME": str(workspace_root / "cache" / "torch"),
            "MPLCONFIGDIR": str(workspace_root / "cache" / "matplotlib"),
            "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(PROJECT_DIR / "src"),
        }
    )
    return environment


def verify_cuda(environment: dict[str, str]) -> None:
    code = (
        "import json, torch, torchvision; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "print(json.dumps({'torch': torch.__version__, 'torchvision': torchvision.__version__, "
        "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), "
        "'bf16': torch.cuda.is_bf16_supported()}))"
    )
    _run([sys.executable, "-c", code], environment=environment)


def stream_process(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> int:
    wrapped_command = command
    timeout_binary = shutil.which("timeout")
    if timeout_binary and timeout_seconds > 0:
        wrapped_command = [
            timeout_binary,
            "--signal=TERM",
            "--kill-after=120",
            str(timeout_seconds),
            *command,
        ]

    print(f"$ {shlex.join(wrapped_command)}", flush=True)
    with log_path.open("a", buffering=1) as log_file:
        process = subprocess.Popen(
            wrapped_command,
            cwd=PROJECT_DIR,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
        return process.wait()


def installed_distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def _validate_args(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_name):
        raise ValueError("run_name may contain only letters, numbers, dots, underscores, and hyphens")
    for name in ("max_steps", "batch_size", "num_workers"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.timeout_hours <= 0:
        raise ValueError("timeout_hours must be positive")


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, env=environment, check=True)


if __name__ == "__main__":
    main()
