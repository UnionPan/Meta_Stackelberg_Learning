#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
SCRIPT_PATH="${ROOT_DIR}/attacker_sandbox/apps/postprocess/postprocess_sandbox.py"
OUTPUT_ROOT="${ROOT_DIR}/attacker_sandbox/outputs"
RUN_ROOT="${ROOT_DIR}/attacker_sandbox/runs"
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

ATTACK_DIRS=(
  "mnist_ipm_20c5a_30r"
  "mnist_lmp_20c5a_30r"
  "mnist_bfl_20c5a_30r"
  "mnist_dba_20c5a_30r"
  "mnist_rl_20c5a_30r"
  "mnist_brl_20c5a_30r"
)

first_attack=1

for attack_dir_name in "${ATTACK_DIRS[@]}"; do
  attack_dir="${OUTPUT_ROOT}/${attack_dir_name}"

  if [[ ! -f "${attack_dir}/summary.json" ]]; then
    echo "Skipping ${attack_dir_name}: missing ${attack_dir}/summary.json"
    continue
  fi

  echo "Postprocessing ${attack_dir_name} -> ${TB_ROOT}/${attack_dir_name#mnist_}"
  cmd=(
    "${PYTHON_BIN}" "${SCRIPT_PATH}"
    --clean_input_dir "${CLEAN_DIR}"
    --attack_input_dir "${attack_dir}"
    --tb_dir "${TB_ROOT}"
  )
  if [[ "${first_attack}" -eq 0 ]]; then
    cmd+=(--skip_clean_write)
  fi

  if ! "${cmd[@]}"; then
    echo "Postprocess failed for ${attack_dir_name}. If the error mentions missing Python packages," >&2
    echo "set up or refresh the sandbox environment with: bash attacker_sandbox/scripts/setup_env.sh cpu" >&2
    exit 1
  fi

  first_attack=0
done

echo "Finished postprocessing all available MNIST 30-round attacks into ${TB_ROOT}."
