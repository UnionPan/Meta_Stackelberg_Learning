#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
DEFAULT_LOGDIR="${ROOT_DIR}/attacker_sandbox/runs/clean_benchmark_100r"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # Keep the launcher self-contained for quick debugging sessions.
  source "${VENV_DIR}/bin/activate"
fi

LOGDIR="${1:-${DEFAULT_LOGDIR}}"
HOST="${TENSORBOARD_HOST:-localhost}"
PORT="${TENSORBOARD_PORT:-6006}"
DISPLAY_HOST="${HOST}"

echo "TensorBoard logdir: ${LOGDIR}"
echo "TensorBoard bind: http://${HOST}:${PORT}"
echo "Open in browser: http://${DISPLAY_HOST}:${PORT}"

exec tensorboard \
  --logdir "${LOGDIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --load_fast=false
