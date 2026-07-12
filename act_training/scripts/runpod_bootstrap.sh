#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_ROOT="${ACT_WORKSPACE_ROOT:-/workspace/act-driving}"
SOURCE_DIR="${ACT_SOURCE_DIR:-$WORKSPACE_ROOT/source}"
TRAINING_DIR="$SOURCE_DIR/act_training"
VENV_DIR="$WORKSPACE_ROOT/venv"
RUN_NAME="${ACT_RUN_NAME:-act-driving-v1}"
MAX_STEPS="${ACT_MAX_STEPS:-10000}"
BATCH_SIZE="${ACT_BATCH_SIZE:-64}"
NUM_WORKERS="${ACT_NUM_WORKERS:-4}"
TIMEOUT_SECONDS="${ACT_TIMEOUT_SECONDS:-43200}"
LOG_DIR="$WORKSPACE_ROOT/logs"
LOG_FILE="$LOG_DIR/$RUN_NAME.log"

if [[ ! -f "$TRAINING_DIR/runpod_train.py" ]]; then
  echo "RunPod training entrypoint not found at $TRAINING_DIR/runpod_train.py" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is missing. Map a RunPod secret to the HF_TOKEN Pod environment variable." >&2
  exit 1
fi

mkdir -p "$WORKSPACE_ROOT/artifacts/runs" "$WORKSPACE_ROOT/cache" "$LOG_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python -m venv --system-site-packages "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check -r "$TRAINING_DIR/requirements-runpod.txt"
"$VENV_DIR/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 \
  torchvision==0.23.0
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --no-deps -e "$TRAINING_DIR"
"$VENV_DIR/bin/python" -c \
  'import torch, torchvision; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} torchvision={torchvision.__version__} gpu={torch.cuda.get_device_name(0)}")'

export HF_HOME="$WORKSPACE_ROOT/cache/huggingface"
export TORCH_HOME="$WORKSPACE_ROOT/cache/torch"
export MPLCONFIGDIR="$WORKSPACE_ROOT/cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Starting RunPod ACT training: run=$RUN_NAME steps=$MAX_STEPS batch=$BATCH_SIZE" | tee -a "$LOG_FILE"
set +e
timeout --signal=TERM --kill-after=120 "$TIMEOUT_SECONDS" \
  "$VENV_DIR/bin/python" "$TRAINING_DIR/runpod_train.py" \
  --config "$TRAINING_DIR/configs/base.json" \
  --run-name "$RUN_NAME" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --artifact-root "$WORKSPACE_ROOT/artifacts/runs" \
  --cache-dir "$WORKSPACE_ROOT/cache/huggingface" \
  2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -eq 0 ]]; then
  echo "RunPod ACT training completed successfully." | tee -a "$LOG_FILE"
elif [[ "$status" -eq 124 ]]; then
  echo "RunPod ACT training reached its $TIMEOUT_SECONDS-second safety limit." | tee -a "$LOG_FILE"
else
  echo "RunPod ACT training failed with exit code $status. Checkpoints remain on the network volume." | tee -a "$LOG_FILE"
fi
exit "$status"
