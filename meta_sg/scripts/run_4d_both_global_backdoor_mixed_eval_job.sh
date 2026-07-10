#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-runs/meta_sg_goal/4d_both_clean_global_backdoor_mixed_h200_t100_k8_l10_c30_a6}"
DEVICE="${DEVICE:-cuda:0}"
RUN_DIR="${RUN_DIR:-}"
CHECKPOINT="${CHECKPOINT:-}"

H="${H:-200}"
NUM_CLIENTS="${NUM_CLIENTS:-30}"
NUM_ATTACKERS="${NUM_ATTACKERS:-6}"
SUBSAMPLE_RATE="${SUBSAMPLE_RATE:-0.2}"
CLIENT_SAMPLES="${CLIENT_SAMPLES:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
SEED="${SEED:-502}"
RL_SEED="${RL_SEED:-506}"

ADAPT_WINDOWS="${ADAPT_WINDOWS:-10}"
ADAPT_WINDOW_HORIZON="${ADAPT_WINDOW_HORIZON:-20}"
ADAPT_UPDATES_PER_WINDOW="${ADAPT_UPDATES_PER_WINDOW:-10}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-32}"
PROXY_REWARD_MODE="${PROXY_REWARD_MODE:-clean_update_anomaly}"
PROXY_SERVER_LR_OFFSET_STEP="${PROXY_SERVER_LR_OFFSET_STEP:-0.5}"
PROXY_SERVER_LR_OFFSET_MAX_STEPS="${PROXY_SERVER_LR_OFFSET_MAX_STEPS:-4}"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -td "$RUN_ROOT"/20* 2>/dev/null | head -n 1)"
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "[eval-job] could not locate a run directory under $RUN_ROOT" >&2
  exit 1
fi
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$RUN_DIR/final"
fi
if [[ ! -f "$CHECKPOINT/defender_meta.pt" && ! -f "$CHECKPOINT" ]]; then
  echo "[eval-job] checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

EVAL_DIR="${EVAL_DIR:-$RUN_DIR/eval}"
mkdir -p "$EVAL_DIR"

DIRECT_JSON="$EVAL_DIR/direct_h200_clean_global_backdoor_mixed.json"
ADAPT_JSON="$EVAL_DIR/adapt_proxy_h200_clean_global_backdoor_mixed.json"

common_args=(
  --checkpoint "$CHECKPOINT"
  --scenario-set clean_global_backdoor_mixed
  --H "$H"
  --num-clients "$NUM_CLIENTS"
  --num-attackers "$NUM_ATTACKERS"
  --subsample-rate "$SUBSAMPLE_RATE"
  --client-samples "$CLIENT_SAMPLES"
  --eval-samples "$EVAL_SAMPLES"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --hidden-dim "$HIDDEN_DIM"
  --device "$DEVICE"
  --seed "$SEED"
  --rl-seed "$RL_SEED"
  --defender-third-action both
  --post-defense-mode model_aware_neuroclip
  --server-lr-min 0.0
  --server-lr-max 1.0
  --lambda-bd 1.0
  --attacker-source native
)

echo "[eval-job] start $(date -Is)"
echo "[eval-job] RUN_DIR=$RUN_DIR"
echo "[eval-job] CHECKPOINT=$CHECKPOINT"
echo "[eval-job] direct=$DIRECT_JSON"
PYTHONPATH=. .venv/bin/python meta_sg/scripts/evaluate_meta_sg_direct.py \
  "${common_args[@]}" \
  --output-json "$DIRECT_JSON"

echo "[eval-job] adaptation=$ADAPT_JSON"
PYTHONPATH=. .venv/bin/python meta_sg/scripts/evaluate_meta_sg_direct.py \
  "${common_args[@]}" \
  --output-json "$ADAPT_JSON" \
  --few-shot \
  --few-shot-method paper_online_proxy_td3 \
  --adaptation-attacker-source native \
  --paper-online-windows "$ADAPT_WINDOWS" \
  --paper-online-window-horizon "$ADAPT_WINDOW_HORIZON" \
  --paper-online-updates-per-window "$ADAPT_UPDATES_PER_WINDOW" \
  --adaptation-batch-size "$ADAPT_BATCH_SIZE" \
  --proxy-reward-mode "$PROXY_REWARD_MODE" \
  --proxy-server-lr-offset-step "$PROXY_SERVER_LR_OFFSET_STEP" \
  --proxy-server-lr-offset-max-steps "$PROXY_SERVER_LR_OFFSET_MAX_STEPS"

echo "[eval-job] done $(date -Is)"
