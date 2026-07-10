#!/usr/bin/env bash
set -euo pipefail

# Paper-like round configuration for our clean + single-backdoor + mixed-backdoor domain.
#
# Core rounds follow the paper's MNIST pre-training scale:
#   T=100 outer meta iterations, K=10 tasks per iteration, H=200 FL rounds,
#   l=10 defender TD3 updates per support trajectory, query_horizon=200.
#
# Client scale defaults to our current mixed-backdoor experiment so the result
# is directly comparable. Override NUM_CLIENTS/NUM_ATTACKERS/SUBSAMPLE_RATE to
# run a stricter paper-scale variant.

MODE="${1:-train}"

RUN_ROOT="${RUN_ROOT:-runs/meta_sg_goal/clean_backdoor_mixed_paperlike_t100_k10_h200_l10}"
CHECKPOINT="${CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda:0}"
RESUME_FROM="${RESUME_FROM:-}"
START_ITERATION="${START_ITERATION:-0}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-}"

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

T="${T:-100}"
K="${K:-10}"
H="${H:-200}"
L="${L:-10}"
SUPPORT_EPISODES="${SUPPORT_EPISODES:-1}"
QUERY_HORIZON="${QUERY_HORIZON:-200}"
META_STEP="${META_STEP:-0.25}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

ADAPTATION_UPDATES="${ADAPTATION_UPDATES:-10}"
SELECTION_MARGIN="${SELECTION_MARGIN:-0.01}"

common_eval_args=(
  --lambda-bd 1.0
  --num-clients "$NUM_CLIENTS"
  --num-attackers "$NUM_ATTACKERS"
  --subsample-rate "$SUBSAMPLE_RATE"
  --client-samples "$CLIENT_SAMPLES"
  --eval-samples "$EVAL_SAMPLES"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --hidden-dim 256
  --defender-third-action server_lr
  --post-defense-mode weight_copy
  --device "$DEVICE"
  --seed 42
  --rl-seed 506
  --few-shot
  --few-shot-method td3
  --few-shot-selection guarded
  --selection-margin "$SELECTION_MARGIN"
  --adaptation-episodes 2
  --adaptation-updates "$ADAPTATION_UPDATES"
  --adaptation-warmup-steps 5
  --adaptation-lr-scale 0.25
  --adaptation-noise 0.05
)

run_train() {
  local resume_args=()
  if [[ -n "$RESUME_FROM" ]]; then
    resume_args+=(--resume-from "$RESUME_FROM" --start-iteration "$START_ITERATION")
  fi
  if [[ -n "$TOTAL_ITERATIONS" ]]; then
    resume_args+=(--total-iterations "$TOTAL_ITERATIONS")
  fi

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
    --defender-third-action server_lr \
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
    --device "$DEVICE" \
    --output-dir "$RUN_ROOT" \
    --log-interval "$LOG_INTERVAL" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    "${resume_args[@]}"
}

resolve_checkpoint() {
  if [[ -n "$CHECKPOINT" ]]; then
    printf '%s\n' "$CHECKPOINT"
    return
  fi
  local latest
  latest="$(find "$RUN_ROOT" -maxdepth 2 -type d -name final -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$latest" ]]; then
    echo "No final checkpoint found under $RUN_ROOT. Set CHECKPOINT=/path/to/final." >&2
    exit 2
  fi
  printf '%s\n' "$latest"
}

run_eval_h200() {
  local ckpt
  ckpt="$(resolve_checkpoint)"
  local out_dir
  out_dir="$(dirname "$ckpt")"

  PYTHONPATH=. .venv/bin/python meta_sg/scripts/evaluate_meta_sg_direct.py \
    --checkpoint "$ckpt" \
    --output-json "$out_dir/eval_h200_backdoor_td3_guarded_margin001_u10.json" \
    --scenario-set backdoor \
    --H 200 \
    --adaptation-horizon 200 \
    --selection-horizon 200 \
    "${common_eval_args[@]}"

  PYTHONPATH=. .venv/bin/python meta_sg/scripts/evaluate_meta_sg_direct.py \
    --checkpoint "$ckpt" \
    --output-json "$out_dir/eval_h200_clean_mixed_backdoor_td3_guarded_margin001_u10.json" \
    --scenario-set clean_mixed_backdoor \
    --H 200 \
    --adaptation-horizon 200 \
    --selection-horizon 200 \
    "${common_eval_args[@]}"
}

case "$MODE" in
  train)
    run_train
    ;;
  eval-h200)
    run_eval_h200
    ;;
  all)
    run_train
    run_eval_h200
    ;;
  *)
    echo "Usage: $0 [train|eval-h200|all]" >&2
    exit 2
    ;;
esac
