#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${TMUX_SESSION_NAME:-smolvla-left-turn}"
RUN_NAME="${VLA_RUN_NAME:-smolvla-left-turn-v2}"
MAX_STEPS="${VLA_MAX_STEPS:-20000}"
BATCH_SIZE="${VLA_BATCH_SIZE:-32}"
WORKSPACE_ROOT="${VLA_WORKSPACE_ROOT:-/workspace/vla-driving}"
LOG_DIR="$WORKSPACE_ROOT/logs"
LOG_FILE="$LOG_DIR/$RUN_NAME-launcher.log"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing $ROOT_DIR/.venv. Run scripts/setup_runpod.sh first." >&2
  exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
command=(
  "$ROOT_DIR/.venv/bin/python" -u "$ROOT_DIR/runpod_main.py"
  --run-name "$RUN_NAME"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  "$@"
)
printf -v quoted_command '%q ' "${command[@]}"
printf -v quoted_log '%q' "$LOG_FILE"

tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR" \
  "bash -lc 'set -o pipefail; ${quoted_command}2>&1 | tee -a ${quoted_log}'"

echo "Started: $SESSION_NAME"
echo "Attach:  tmux attach -t $SESSION_NAME"
echo "Log:     tail -F $LOG_FILE"
