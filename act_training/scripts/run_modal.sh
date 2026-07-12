#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$TRAINING_DIR/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy .env.example to .env and add HF_TOKEN." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is empty in $ENV_FILE." >&2
  exit 1
fi

if [[ -n "${MODAL_TOKEN_ID:-}" || -n "${MODAL_TOKEN_SECRET:-}" ]]; then
  if [[ -z "${MODAL_TOKEN_ID:-}" || -z "${MODAL_TOKEN_SECRET:-}" ]]; then
    echo "Set both MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, or leave both blank." >&2
    exit 1
  fi
else
  unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
fi

MODAL_BIN="${MODAL_BIN:-$(command -v modal || true)}"
if [[ -z "$MODAL_BIN" ]]; then
  echo "Modal CLI not found. Install act_training/requirements-local.txt." >&2
  exit 1
fi

cd "$TRAINING_DIR"
exec "$MODAL_BIN" run modal_app.py "$@"
