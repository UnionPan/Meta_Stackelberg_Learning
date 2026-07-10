#!/usr/bin/env bash
set -euo pipefail

# Aligned 4D version of the H=200 paper-like clean/backdoor/mixed-backdoor run.
# This keeps the task mix and FL scale aligned with
# launch_clean_backdoor_mixed_h200_paperlike.sh, but uses the 4D defender action:
#   alpha, beta, neuroclip_epsilon, server_lr

RUN_ROOT="${RUN_ROOT:-runs/meta_sg_goal/4d_both_clean_backdoor_mixed_h200_t100_k10_l10_c30_a6_aligned}"
DEVICE="${DEVICE:-cuda:0}"
RESUME_FROM="${RESUME_FROM:-}"
START_ITERATION="${START_ITERATION:-0}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-}"

T="${T:-100}"
K="${K:-10}"
H="${H:-200}"
L="${L:-10}"
SUPPORT_EPISODES="${SUPPORT_EPISODES:-1}"
QUERY_HORIZON="${QUERY_HORIZON:-200}"
META_STEP="${META_STEP:-0.25}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

NUM_CLIENTS="${NUM_CLIENTS:-30}"
NUM_ATTACKERS="${NUM_ATTACKERS:-6}"
SUBSAMPLE_RATE="${SUBSAMPLE_RATE:-0.2}"
CLIENT_SAMPLES="${CLIENT_SAMPLES:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
FL_PARALLEL_CLIENTS="${FL_PARALLEL_CLIENTS:-2}"
FL_NUM_WORKERS="${FL_NUM_WORKERS:-0}"

SERVER_LR_MIN="${SERVER_LR_MIN:-0.0}"
SERVER_LR_MAX="${SERVER_LR_MAX:-1.0}"
SERVER_LR_PENALTY_WEIGHT="${SERVER_LR_PENALTY_WEIGHT:-0.0}"
NEUROCLIP_EPS_MIN="${NEUROCLIP_EPS_MIN:-1.0}"
NEUROCLIP_EPS_MAX="${NEUROCLIP_EPS_MAX:-10.0}"

resume_args=()
if [[ -n "$RESUME_FROM" ]]; then
  resume_args+=(--resume-from "$RESUME_FROM" --start-iteration "$START_ITERATION")
fi
if [[ -n "$TOTAL_ITERATIONS" ]]; then
  resume_args+=(--total-iterations "$TOTAL_ITERATIONS")
fi

echo "[job] start $(date -Is)"
echo "[job] RUN_ROOT=$RUN_ROOT"
echo "[job] aligned 4D: attack_domain=clean_backdoor_mixed T=$T K=$K H=$H L=$L"
echo "[job] clients=$NUM_CLIENTS attackers=$NUM_ATTACKERS subsample=$SUBSAMPLE_RATE"
echo "[job] server_lr=[${SERVER_LR_MIN}, ${SERVER_LR_MAX}] penalty=$SERVER_LR_PENALTY_WEIGHT"

PYTHONPATH=. .venv/bin/python meta_sg/scripts/run_meta_sg_pretraining.py \
  --backend fl_sandbox \
  --attack-domain clean_backdoor_mixed \
  --T "$T" \
  --K "$K" \
  --H "$H" \
  --l "$L" \
  --support-episodes "$SUPPORT_EPISODES" \
  --task-sampler stratified \
  --meta-step "$META_STEP" \
  --meta-objective query_targeted_reptile \
  --query-horizon "$QUERY_HORIZON" \
  --query-targeted-asr-reduction-margin 0.0 \
  --query-targeted-min-base-backdoor 0.0 \
  --defender-third-action both \
  --post-defense-mode model_aware_neuroclip \
  --neuroclip-eps-min "$NEUROCLIP_EPS_MIN" \
  --neuroclip-eps-max "$NEUROCLIP_EPS_MAX" \
  --server-lr-min "$SERVER_LR_MIN" \
  --server-lr-max "$SERVER_LR_MAX" \
  --server-lr-penalty-weight "$SERVER_LR_PENALTY_WEIGHT" \
  --num-clients "$NUM_CLIENTS" \
  --num-attackers "$NUM_ATTACKERS" \
  --subsample-rate "$SUBSAMPLE_RATE" \
  --client-samples "$CLIENT_SAMPLES" \
  --eval-samples "$EVAL_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --buffer-capacity "$BUFFER_CAPACITY" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --fl-batch-size "$BATCH_SIZE" \
  --fl-parallel-clients "$FL_PARALLEL_CLIENTS" \
  --fl-num-workers "$FL_NUM_WORKERS" \
  --hidden-dim 256 \
  --device "$DEVICE" \
  --output-dir "$RUN_ROOT" \
  --log-interval "$LOG_INTERVAL" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --seed 42 \
  "${resume_args[@]}"

echo "[job] done $(date -Is)"
