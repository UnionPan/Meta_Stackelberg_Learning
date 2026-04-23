#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
SCRIPT_PATH="${ROOT_DIR}/fl_sandbox/core/postprocess/postprocess.py"
OUTPUT_ROOT="${ROOT_DIR}/fl_sandbox/outputs"
RUN_ROOT="${ROOT_DIR}/fl_sandbox/runs"
CLEAN_DIR="${OUTPUT_ROOT}/mnist_clean_20c_30r"
TB_ROOT="${RUN_ROOT}/mnist_30r_all"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No python interpreter found in PATH." >&2
  exit 1
fi

if [[ ! -f "${CLEAN_DIR}/summary.json" ]]; then
  echo "Missing clean baseline summary: ${CLEAN_DIR}/summary.json" >&2
  exit 1
fi

rm -rf "${TB_ROOT}"
mkdir -p "${TB_ROOT}"

METHODS=(
  "ipm"
  "lmp"
  "bfl"
  "dba"
  "rl"
  "brl"
)

echo "Postprocessing MNIST 30-round attacks -> ${TB_ROOT}"
cmd=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --clean_input_dir "${CLEAN_DIR}"
  --attack_root "${OUTPUT_ROOT}"
  --methods "${METHODS[@]}"
  --tb_dir "${TB_ROOT}"
)

if ! "${cmd[@]}"; then
  echo "Postprocess failed. If the error mentions missing Python packages," >&2
  echo "set up or refresh the sandbox environment with: bash fl_sandbox/scripts/setup_env.sh cpu" >&2
  exit 1
fi

echo "Finished postprocessing all available MNIST 30-round attacks into ${TB_ROOT}."
