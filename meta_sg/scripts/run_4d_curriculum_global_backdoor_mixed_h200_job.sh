#!/usr/bin/env bash
set -euo pipefail

# Three-stage curriculum for the 4D defender action:
#   1. clean + global poisoning: clean, ipm, lmp, rl
#   2. clean + backdoor family: clean, bfl, dba, rl_backdoor, mixed_backdoor
#   3. full mixed domain: clean, ipm, lmp, rl, bfl, dba, rl_backdoor, mixed_backdoor
#
# Keep ATTACK_CONTEXT disabled for this script. Each stage has a different task
# set, and the current --attack-context implementation changes obs_dim with the
# task set, which makes checkpoint chaining incompatible.

RUN_ROOT="${RUN_ROOT:-runs/meta_sg_goal/curriculum_global_backdoor_mixed_h200_t50_c30_a6}"
DEVICE="${DEVICE:-cuda:0}"
INITIAL_RESUME_FROM="${INITIAL_RESUME_FROM:-}"
INITIAL_START_ITERATION="${INITIAL_START_ITERATION:-0}"

STAGE1_DOMAIN="${STAGE1_DOMAIN:-clean_global}"
STAGE2_DOMAIN="${STAGE2_DOMAIN:-clean_backdoor_mixed}"
STAGE3_DOMAIN="${STAGE3_DOMAIN:-clean_global_backdoor_mixed}"

STAGE1_T="${STAGE1_T:-10}"
STAGE2_T="${STAGE2_T:-10}"
STAGE3_T="${STAGE3_T:-30}"
STAGE1_K="${STAGE1_K:-4}"
STAGE2_K="${STAGE2_K:-5}"
STAGE3_K="${STAGE3_K:-8}"
STAGE1_META_STEP="${STAGE1_META_STEP:-0.25}"
STAGE2_META_STEP="${STAGE2_META_STEP:-0.25}"
STAGE3_META_STEP="${STAGE3_META_STEP:-0.1}"

H="${H:-200}"
L="${L:-10}"
SUPPORT_EPISODES="${SUPPORT_EPISODES:-1}"
QUERY_HORIZON="${QUERY_HORIZON:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
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

mkdir -p "$RUN_ROOT"

TOTAL_ITERATIONS="$((INITIAL_START_ITERATION + STAGE1_T + STAGE2_T + STAGE3_T))"

echo "[curriculum] start $(date -Is)"
echo "[curriculum] RUN_ROOT=$RUN_ROOT"
echo "[curriculum] stages:"
echo "[curriculum]   stage1 domain=$STAGE1_DOMAIN T=$STAGE1_T K=$STAGE1_K meta_step=$STAGE1_META_STEP"
echo "[curriculum]   stage2 domain=$STAGE2_DOMAIN T=$STAGE2_T K=$STAGE2_K meta_step=$STAGE2_META_STEP"
echo "[curriculum]   stage3 domain=$STAGE3_DOMAIN T=$STAGE3_T K=$STAGE3_K meta_step=$STAGE3_META_STEP"
echo "[curriculum] total_iterations=$TOTAL_ITERATIONS H=$H L=$L"
echo "[curriculum] clients=$NUM_CLIENTS attackers=$NUM_ATTACKERS subsample=$SUBSAMPLE_RATE"

run_stage() {
  local stage_name="$1"
  local domain="$2"
  local stage_t="$3"
  local stage_k="$4"
  local meta_step="$5"
  local start_iteration="$6"
  local resume_from="$7"
  local stage_root="$RUN_ROOT/$stage_name"
  local resume_args=()

  if [[ -n "$resume_from" ]]; then
    resume_args+=(--resume-from "$resume_from" --start-iteration "$start_iteration")
  elif [[ "$start_iteration" -gt 0 ]]; then
    echo "[curriculum] missing resume checkpoint for nonzero start_iteration=$start_iteration" >&2
    exit 1
  fi

  echo "[curriculum] stage=$stage_name start $(date -Is)"
  echo "[curriculum] stage=$stage_name domain=$domain T=$stage_t K=$stage_k start_iteration=$start_iteration resume_from=${resume_from:-<none>}"

  PYTHONPATH=. .venv/bin/python meta_sg/scripts/run_meta_sg_pretraining.py \
    --backend fl_sandbox \
    --attack-domain "$domain" \
    --T "$stage_t" \
    --K "$stage_k" \
    --H "$H" \
    --l "$L" \
    --support-episodes "$SUPPORT_EPISODES" \
    --task-sampler stratified \
    --meta-step "$meta_step" \
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
    --output-dir "$stage_root" \
    --log-interval "$LOG_INTERVAL" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --seed 42 \
    --total-iterations "$TOTAL_ITERATIONS" \
    "${resume_args[@]}"

  local run_dir
  run_dir="$(ls -td "$stage_root"/20* 2>/dev/null | head -n 1)"
  if [[ -z "$run_dir" || ! -f "$run_dir/final/defender_meta.pt" ]]; then
    echo "[curriculum] stage=$stage_name did not produce final checkpoint under $stage_root" >&2
    exit 1
  fi
  echo "[curriculum] stage=$stage_name done $(date -Is) final=$run_dir/final"
  printf '%s\n' "$run_dir/final"
}

start_iteration="$INITIAL_START_ITERATION"
resume_from="$INITIAL_RESUME_FROM"

resume_from="$(run_stage stage1_global "$STAGE1_DOMAIN" "$STAGE1_T" "$STAGE1_K" "$STAGE1_META_STEP" "$start_iteration" "$resume_from" | tee /dev/stderr | tail -n 1)"
start_iteration="$((start_iteration + STAGE1_T))"

resume_from="$(run_stage stage2_backdoor "$STAGE2_DOMAIN" "$STAGE2_T" "$STAGE2_K" "$STAGE2_META_STEP" "$start_iteration" "$resume_from" | tee /dev/stderr | tail -n 1)"
start_iteration="$((start_iteration + STAGE2_T))"

resume_from="$(run_stage stage3_full "$STAGE3_DOMAIN" "$STAGE3_T" "$STAGE3_K" "$STAGE3_META_STEP" "$start_iteration" "$resume_from" | tee /dev/stderr | tail -n 1)"

echo "[curriculum] done $(date -Is) final=$resume_from"
