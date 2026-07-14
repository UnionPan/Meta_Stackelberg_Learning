#!/usr/bin/env bash

set -Eeuo pipefail

# Paper-scale Meta-SG training over IPM, LMP, and adaptive RL, followed by
# clean/IPM/LMP/RL final evaluation. All defaults can be reduced for a smoke run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/fl_sandbox/runs/meta_sg_global_h200_t100_k10_c30_a6}"
RUN_ID="${RUN_ID:-$(date -u '+%Y%m%dT%H%M%SZ')-seed${MASTER_SEED:-${SEED:-42}}}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
EVALUATION_DIR="${RUN_DIR}/evaluation"
STATUS_JSON="${RUN_DIR}/status.json"
PROVENANCE_JSON="${RUN_DIR}/provenance.json"
RESOURCE_CSV="${RUN_DIR}/resource_metrics.csv"

BACKEND="${BACKEND:-fl_sandbox}"
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
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
MASTER_SEED="${MASTER_SEED:-${SEED:-42}}"
TRAINING_SEED="${TRAINING_SEED:-${MASTER_SEED}}"
EVALUATION_SEED="${EVALUATION_SEED:-$((MASTER_SEED + 10000))}"

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

# Artifact CLI contracts: experiment_artifacts.py provenance; experiment_artifacts.py status.
ARTIFACT_HELPER="${ROOT_DIR}/meta_sg/scripts/experiment_artifacts.py"
ACTIVE_PID=""
MONITOR_PID=""
FINAL_STAGE="initializing"
FAILURE_RECORDED=0

last_completed_iteration() {
  local metadata="${RUN_DIR}/checkpoints/latest/checkpoint.json"
  if [[ -f "${metadata}" ]]; then
    "${PYTHON_BIN}" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["completed_iteration"]))' "${metadata}" 2>/dev/null || echo "${START_ITERATION}"
  else
    echo "${START_ITERATION}"
  fi
}

write_status() {
  local stage="$1"
  local message="$2"
  local exit_code="${3:-}"
  local completed
  completed="$(last_completed_iteration)"
  local arguments=(
    status
    --output "${STATUS_JSON}"
    --stage "${stage}"
    --message "${message}"
    --last-completed-iteration "${completed}"
  )
  if [[ -n "${exit_code}" ]]; then
    arguments+=(--exit-code "${exit_code}")
  fi
  "${PYTHON_BIN}" "${ARTIFACT_HELPER}" "${arguments[@]}"
}

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  MONITOR_PID=""
}

on_signal() {
  local signal_name="$1"
  echo "[global-model-poisoning] received ${signal_name}; stopping child" >&2
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill -TERM "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  exit 130
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  set +e
  stop_monitor
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill -TERM "${ACTIVE_PID}" 2>/dev/null
    wait "${ACTIVE_PID}" 2>/dev/null
  fi
  if [[ "${FINAL_STAGE}" != "completed" ]] && [[ "${FAILURE_RECORDED}" -eq 0 ]]; then
    write_status failed "launcher exited during ${FINAL_STAGE}" "${exit_code}"
  fi
  exit "${exit_code}"
}

resource_monitor() {
  local watched_pid="$1"
  local gpu_index=""
  if [[ "${DEVICE}" == cuda:* ]]; then
    gpu_index="${DEVICE#cuda:}"
  elif [[ "${DEVICE}" == "cuda" ]]; then
    gpu_index="0"
  fi
  while kill -0 "${watched_pid}" 2>/dev/null; do
    local timestamp process_stats process_cpu process_rss gpu_metrics
    timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    process_stats="$(ps -p "${watched_pid}" -o %cpu=,rss= 2>/dev/null | awk '{$1=$1; print}' || true)"
    process_cpu="$(awk '{print $1}' <<<"${process_stats}")"
    process_rss="$(awk '{print $2}' <<<"${process_stats}")"
    gpu_metrics=",,,,"
    if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${gpu_index}" ]]; then
      gpu_metrics="$(nvidia-smi \
        --id="${gpu_index}" \
        --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
      gpu_metrics="${gpu_metrics:-,,,,}"
    fi
    echo "${timestamp},${watched_pid},${process_cpu},${process_rss},${gpu_metrics}" >>"${RESOURCE_CSV}"
    sleep "${MONITOR_INTERVAL_SECONDS}"
  done
}

run_logged_child() {
  local log_path="$1"
  shift
  set +e
  PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}" "$@" > >(tee -a "${log_path}") 2>&1 &
  ACTIVE_PID=$!
  resource_monitor "${ACTIVE_PID}" &
  MONITOR_PID=$!
  wait "${ACTIVE_PID}"
  local child_exit=$?
  ACTIVE_PID=""
  stop_monitor
  set -e
  return "${child_exit}"
}

trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

mkdir -p "${LOG_DIR}" "${EVALUATION_DIR}"
echo "timestamp,pid,process_cpu_percent,process_rss_kb,gpu_utilization_percent,gpu_memory_used_mb,gpu_memory_total_mb,gpu_temperature_c,gpu_power_w" >"${RESOURCE_CSV}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${ARTIFACT_HELPER}" provenance \
  --output "${PROVENANCE_JSON}" \
  --repo-root "${ROOT_DIR}" \
  --master-seed "${MASTER_SEED}" \
  --training-seed "${TRAINING_SEED}" \
  --evaluation-seed "${EVALUATION_SEED}" \
  --device "${DEVICE}" \
  --config "backend=${BACKEND}" \
  --config "run_id=${RUN_ID}" \
  --config "T=${T}" \
  --config "K=${K}" \
  --config "H=${H}" \
  --config "l=${L}" \
  --config "N_A=${N_A}" \
  --config "checkpoint_interval=${CHECKPOINT_INTERVAL}" \
  --config "num_clients=${NUM_CLIENTS}" \
  --config "num_attackers=${NUM_ATTACKERS}" \
  --config "subsample_rate=${SUBSAMPLE_RATE}"
write_status initializing "run directory and provenance created"

resume_args=(--start-iteration "${START_ITERATION}")
if [[ -n "${RESUME_FROM}" ]]; then
  resume_args+=(--resume-from "${RESUME_FROM}")
fi

training_command=(
  "${PYTHON_BIN}"
  meta_sg/scripts/run_meta_sg_pretraining.py
  --backend "${BACKEND}"
  --dataset mnist
  --attack-domain model_poisoning
  --T "${T}"
  --K "${K}"
  --H "${H}"
  --l "${L}"
  --N-A "${N_A}"
  --support-episodes "${SUPPORT_EPISODES}"
  --task-sampler stratified
  --meta-objective reptile
  --meta-step "${META_STEP}"
  --query-diagnostics-horizon "${H}"
  --defender-third-action neuroclip
  --post-defense-mode model_aware_neuroclip
  --neuroclip-eps-min 1.0
  --neuroclip-eps-max 10.0
  --num-clients "${NUM_CLIENTS}"
  --num-attackers "${NUM_ATTACKERS}"
  --subsample-rate "${SUBSAMPLE_RATE}"
  --client-samples "${CLIENT_SAMPLES}"
  --eval-samples "${EVAL_SAMPLES}"
  --batch-size "${TD3_BATCH_SIZE}"
  --buffer-capacity "${BUFFER_CAPACITY}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --fl-batch-size "${FL_BATCH_SIZE}"
  --fl-parallel-clients "${FL_PARALLEL_CLIENTS}"
  --fl-num-workers "${FL_NUM_WORKERS}"
  --hidden-dim "${HIDDEN_DIM}"
  --device "${DEVICE}"
  --output-dir "${RUN_ROOT}"
  --run-name "${RUN_ID}"
  --log-interval "${LOG_INTERVAL}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --latest-checkpoint-only
  --total-iterations "${TOTAL_ITERATIONS}"
  --tensorboard
  --seed "${TRAINING_SEED}"
  "${resume_args[@]}"
)

echo "[global-model-poisoning] start $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "[global-model-poisoning] run=${RUN_ID} backend=${BACKEND} device=${DEVICE}"
echo "[global-model-poisoning] T=${T} K=${K} H=${H} l=${L} N_A=${N_A} checkpoint=${CHECKPOINT_INTERVAL}"
FINAL_STAGE="training"
write_status training "started"
if run_logged_child "${LOG_DIR}/train.log" "${training_command[@]}"; then
  :
else
  training_exit=$?
  FAILURE_RECORDED=1
  write_status failed "training exited non-zero" "${training_exit}"
  exit "${training_exit}"
fi
write_status training "completed"

raw_evaluation_json="${EVALUATION_DIR}/final_model_poisoning_h${H}_seed_${EVALUATION_SEED}.json"
evaluation_summary_json="${EVALUATION_DIR}/summary.json"
evaluation_command=(
  "${PYTHON_BIN}"
  meta_sg/scripts/evaluate_meta_sg_direct.py
  --checkpoint "${RUN_DIR}/final"
  --output-json "${raw_evaluation_json}"
  --summary-json "${evaluation_summary_json}"
  --master-seed "${MASTER_SEED}"
  --scenario-set model_poisoning
  --H "${H}"
  --num-clients "${NUM_CLIENTS}"
  --num-attackers "${NUM_ATTACKERS}"
  --subsample-rate "${SUBSAMPLE_RATE}"
  --client-samples "${CLIENT_SAMPLES}"
  --eval-samples "${EVAL_SAMPLES}"
  --batch-size "${FL_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --hidden-dim "${HIDDEN_DIM}"
  --device "${DEVICE}"
  --seed "${EVALUATION_SEED}"
  --rl-seed "${EVALUATION_SEED}"
  --defender-third-action neuroclip
  --post-defense-mode model_aware_neuroclip
  --neuroclip-eps-min 1.0
  --neuroclip-eps-max 10.0
)

FINAL_STAGE="evaluating"
write_status evaluating "started"
if run_logged_child "${LOG_DIR}/final_eval.log" "${evaluation_command[@]}"; then
  :
else
  evaluation_exit=$?
  FAILURE_RECORDED=1
  write_status failed "evaluation exited non-zero" "${evaluation_exit}"
  exit "${evaluation_exit}"
fi

FINAL_STAGE="completed"
write_status completed "training and final evaluation completed" 0
echo "[global-model-poisoning] completed $(date '+%Y-%m-%dT%H:%M:%S%z') output=${RUN_DIR}"
