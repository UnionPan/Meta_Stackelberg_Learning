#!/usr/bin/env bash
set -euo pipefail

export RUN_ROOT="${RUN_ROOT:-runs/meta_sg_goal/clean_backdoor_mixed_paperlike_t100_k10_h200_l10}"
export T="${T:-100}"
export K="${K:-10}"
export H="${H:-200}"
export L="${L:-10}"
export QUERY_HORIZON="${QUERY_HORIZON:-200}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"

export NUM_CLIENTS="${NUM_CLIENTS:-30}"
export NUM_ATTACKERS="${NUM_ATTACKERS:-6}"
export SUBSAMPLE_RATE="${SUBSAMPLE_RATE:-0.2}"
export CLIENT_SAMPLES="${CLIENT_SAMPLES:-64}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
export BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"

export FL_PARALLEL_CLIENTS="${FL_PARALLEL_CLIENTS:-2}"
export FL_NUM_WORKERS="${FL_NUM_WORKERS:-0}"
export DEVICE="${DEVICE:-cuda:0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export RESUME_FROM="${RESUME_FROM:-}"
export START_ITERATION="${START_ITERATION:-0}"
export TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-}"

echo "[job] start $(date -Is)"
echo "[job] RUN_ROOT=$RUN_ROOT"
echo "[job] T=$T K=$K H=$H L=$L QUERY_HORIZON=$QUERY_HORIZON"
echo "[job] clients=$NUM_CLIENTS attackers=$NUM_ATTACKERS subsample=$SUBSAMPLE_RATE"
echo "[job] checkpoint_interval=$CHECKPOINT_INTERVAL fl_parallel_clients=$FL_PARALLEL_CLIENTS device=$DEVICE"
if [[ -n "$RESUME_FROM" ]]; then
  echo "[job] resume_from=$RESUME_FROM start_iteration=$START_ITERATION total_iterations=${TOTAL_ITERATIONS:-$((START_ITERATION + T))}"
fi

meta_sg/scripts/launch_clean_backdoor_mixed_h200_paperlike.sh all

echo "[job] done $(date -Is)"
