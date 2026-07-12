#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${TMUX_SESSION_NAME:-act-h100}"
RUN_NAME="${RUN_NAME:-act-driving-v1}"
LOG_FILE="$TRAINING_DIR/logs/$RUN_NAME.log"

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$TRAINING_DIR/logs"
tmux new-session -d -s "$SESSION_NAME" -c "$TRAINING_DIR"
tmux send-keys -t "$SESSION_NAME" \
  "set -o pipefail; ./scripts/run_modal.sh --run-name $RUN_NAME --max-steps 10000 --batch-size 64 2>&1 | tee $LOG_FILE" \
  C-m

echo "Started H100 training in tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Log: tail -f $LOG_FILE"
