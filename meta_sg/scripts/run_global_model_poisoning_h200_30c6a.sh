#!/usr/bin/env bash

set -euo pipefail

# Paper-scale Meta-SG training over the three global model-poisoning tasks:
# IPM, LMP, and adaptive RL. Stratified sampling guarantees that all three
# task types appear in every outer iteration when K covers the domain.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/fl_sandbox/runs/meta_sg_global_h200_t100_k10_c30_a6}"
DEVICE="${DEVICE:-cuda:0}"
RESUME_FROM="${RESUME_FROM:-}"
START_ITERATION="${START_ITERATION:-0}"

T="${T:-100}"
K="${K:-10}"
H="${H:-200}"
L="${L:-10}"
N_A="${N_A:-10}"
SUPPORT_EPISODES="${SUPPORT_EPISODES:-1}"
META_STEP="${META_STEP:-1.0}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-${T}}"

NUM_CLIENTS="${NUM_CLIENTS:-30}"
NUM_ATTACKERS="${NUM_ATTACKERS:-6}"
SUBSAMPLE_RATE="${SUBSAMPLE_RATE:-0.2}"
CLIENT_SAMPLES="${CLIENT_SAMPLES:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
FL_BATCH_SIZE="${FL_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
FL_PARALLEL_CLIENTS="${FL_PARALLEL_CLIENTS:-2}"
FL_NUM_WORKERS="${FL_NUM_WORKERS:-0}"

HIDDEN_DIM="${HIDDEN_DIM:-256}"
TD3_BATCH_SIZE="${TD3_BATCH_SIZE:-32}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SEED="${SEED:-42}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "${ROOT_DIR}/.venv/bin/python" \
    "${HOME}/anaconda3/bin/python"; do
    if [[ -x "${candidate}" ]] && \
      "${candidate}" -c 'import torch' >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" ]] || [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "No Python interpreter with PyTorch is available." >&2
  echo "Repair .venv or set PYTHON_BIN explicitly." >&2
  exit 1
fi

resume_args=(--start-iteration "${START_ITERATION}")
if [[ -n "${RESUME_FROM}" ]]; then
  resume_args+=(--resume-from "${RESUME_FROM}")
fi

mkdir -p "${RUN_ROOT}"

echo "[global-model-poisoning] start $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "[global-model-poisoning] tasks=ipm,lmp,rl sampler=stratified"
echo "[global-model-poisoning] T=${T} K=${K} H=${H} l=${L} N_A=${N_A}"
echo "[global-model-poisoning] clients=${NUM_CLIENTS} attackers=${NUM_ATTACKERS} subsample=${SUBSAMPLE_RATE}"
echo "[global-model-poisoning] output=${RUN_ROOT} device=${DEVICE}"

cd "${ROOT_DIR}"
PYTHONUNBUFFERED=1 PYTHONPATH=. "${PYTHON_BIN}" \
  meta_sg/scripts/run_meta_sg_pretraining.py \
  --backend fl_sandbox \
  --dataset mnist \
  --attack-domain model_poisoning \
  --T "${T}" \
  --K "${K}" \
  --H "${H}" \
  --l "${L}" \
  --N-A "${N_A}" \
  --support-episodes "${SUPPORT_EPISODES}" \
  --task-sampler stratified \
  --meta-objective reptile \
  --meta-step "${META_STEP}" \
  --query-diagnostics-horizon "${H}" \
  --defender-third-action neuroclip \
  --post-defense-mode model_aware_neuroclip \
  --neuroclip-eps-min 1.0 \
  --neuroclip-eps-max 10.0 \
  --num-clients "${NUM_CLIENTS}" \
  --num-attackers "${NUM_ATTACKERS}" \
  --subsample-rate "${SUBSAMPLE_RATE}" \
  --client-samples "${CLIENT_SAMPLES}" \
  --eval-samples "${EVAL_SAMPLES}" \
  --batch-size "${TD3_BATCH_SIZE}" \
  --buffer-capacity "${BUFFER_CAPACITY}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --fl-batch-size "${FL_BATCH_SIZE}" \
  --fl-parallel-clients "${FL_PARALLEL_CLIENTS}" \
  --fl-num-workers "${FL_NUM_WORKERS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --device "${DEVICE}" \
  --output-dir "${RUN_ROOT}" \
  --log-interval "${LOG_INTERVAL}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --total-iterations "${TOTAL_ITERATIONS}" \
  --tensorboard \
  --seed "${SEED}" \
  "${resume_args[@]}"

echo "[global-model-poisoning] done $(date '+%Y-%m-%dT%H:%M:%S%z')"
