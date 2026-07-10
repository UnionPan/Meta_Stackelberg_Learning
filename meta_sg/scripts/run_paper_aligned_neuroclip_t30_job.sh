#!/usr/bin/env bash
set -euo pipefail

# Small-scale paper-aligned NeuroClip Meta-SG run.
#
# Trains a 3D defender action policy:
#   (alpha norm bound, beta trimmed mean ratio, neuroclip_eps)
# with model-aware NeuroClip reward/evaluation, then runs direct and
# paper-online adaptation evaluations.

RUN_ROOT="${RUN_ROOT:-runs/meta_sg_goal/paper_aligned_neuroclip_t30_k7_h200_l10_c20_a4}"
DEVICE="${DEVICE:-cuda:0}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-30}"

T="${T:-30}"
K="${K:-7}"
H="${H:-200}"
L="${L:-10}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
NUM_ATTACKERS="${NUM_ATTACKERS:-4}"
SUBSAMPLE_RATE="${SUBSAMPLE_RATE:-0.2}"
CLIENT_SAMPLES="${CLIENT_SAMPLES:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-20000}"
FL_PARALLEL_CLIENTS="${FL_PARALLEL_CLIENTS:-2}"
FL_NUM_WORKERS="${FL_NUM_WORKERS:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"

DEFENDER_THIRD_ACTION="${DEFENDER_THIRD_ACTION:-neuroclip}"
POST_DEFENSE_MODE="${POST_DEFENSE_MODE:-model_aware_neuroclip}"
NEUROCLIP_EPS_MIN="${NEUROCLIP_EPS_MIN:-1.0}"
NEUROCLIP_EPS_MAX="${NEUROCLIP_EPS_MAX:-10.0}"
NEUROCLIP_LOG_SCALE="${NEUROCLIP_LOG_SCALE:-0}"

SEED="${SEED:-42}"
RL_SEED="${RL_SEED:-506}"
LAMBDA_BD="${LAMBDA_BD:-1.0}"
META_STEP="${META_STEP:-0.25}"
SUPPORT_EPISODES="${SUPPORT_EPISODES:-1}"
QUERY_HORIZON="${QUERY_HORIZON:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

ADAPT_CONFIGS="${ADAPT_CONFIGS:-5:40:10 10:20:10}"
ADAPT_SCENARIOS="${ADAPT_SCENARIOS:-backdoor mixed}"

mkdir -p "$RUN_ROOT/job_logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$RUN_ROOT/job_logs/paper_aligned_neuroclip_t30_${STAMP}.log"

neuroclip_scale_args=()
if [[ "$NEUROCLIP_LOG_SCALE" == "1" ]]; then
  neuroclip_scale_args+=(--neuroclip-log-scale)
fi

print_header() {
  cat <<EOF
[paper-aligned-neuroclip-t30]
  run_root:   $RUN_ROOT
  log:        $LOG
  device:     $DEVICE
  train:      T=$T K=$K H=$H l=$L support_episodes=$SUPPORT_EPISODES query_horizon=$QUERY_HORIZON
  clients:    $NUM_CLIENTS attackers=$NUM_ATTACKERS subsample=$SUBSAMPLE_RATE
  action:     alpha,beta,neuroclip_eps
  post_def:   $POST_DEFENSE_MODE
  eps_range:  [$NEUROCLIP_EPS_MIN, $NEUROCLIP_EPS_MAX] log_scale=$NEUROCLIP_LOG_SCALE
  adapt:      scenarios="$ADAPT_SCENARIOS" configs="$ADAPT_CONFIGS" selections="always guarded"
EOF
}

heartbeat_wait() {
  local pid="$1"
  local label="$2"
  local start
  start="$(date +%s)"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$HEARTBEAT_INTERVAL"
    if kill -0 "$pid" 2>/dev/null; then
      local now elapsed proc
      now="$(date +%s)"
      elapsed=$((now - start))
      proc="$(ps -o %cpu=,%mem=,rss= -p "$pid" 2>/dev/null | awk '{printf "cpu=%s%% mem=%s%% rss=%sKB", $1, $2, $3}')"
      echo "[heartbeat] $label elapsed=${elapsed}s pid=$pid ${proc:-running}" | tee -a "$LOG"
    fi
  done
}

run_logged() {
  local label="$1"
  shift
  echo
  echo "[start] $label $(date -Is)" | tee -a "$LOG"
  "$@" > >(tee -a "$LOG") 2>&1 &
  local pid=$!
  echo "[pid] $label pid=$pid" | tee -a "$LOG"
  if [[ "$HEARTBEAT_INTERVAL" -gt 0 ]]; then
    heartbeat_wait "$pid" "$label"
  fi
  set +e
  wait "$pid"
  local code=$?
  set -e
  echo "[done] $label exit_code=$code $(date -Is)" | tee -a "$LOG"
  if [[ "$code" -ne 0 ]]; then
    exit "$code"
  fi
}

latest_checkpoint() {
  local ckpt
  ckpt="$(find "$RUN_ROOT" -mindepth 2 -maxdepth 3 -type d \( -name final -o -name 'iter_*' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$ckpt" ]]; then
    echo "[error] no checkpoint directory found under $RUN_ROOT" >&2
    exit 3
  fi
  printf '%s\n' "$ckpt"
}

parse_json() {
  local path="$1"
  local label="$2"
  python - "$path" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
data = json.loads(path.read_text())
print(f"[result] {label}")
for row in data:
    fs = row.get("few_shot_adaptation") or {}
    ev = fs.get("evaluation") or fs.get("adapted_evaluation") or {}
    clean0 = float(row.get("final_clean_acc") or 0.0)
    clean1 = float(ev.get("final_clean_acc", clean0) or 0.0)
    asr0 = float(row.get("final_backdoor_acc") or 0.0)
    asr1 = float(ev.get("final_backdoor_acc", asr0) or 0.0)
    print(
        f"  {row['scenario']:<14} clean {clean0:.3f}->{clean1:.3f} "
        f"({clean1-clean0:+.3f}) | ASR {asr0:.3f}->{asr1:.3f} ({asr1-asr0:+.3f})"
    )
PY
}

run_direct_eval() {
  local ckpt="$1"
  local out_dir
  out_dir="$(dirname "$ckpt")"
  local scenario
  for scenario in backdoor mixed; do
    local out="$out_dir/eval_h${H}_${scenario}_direct_paper_aligned_neuroclip.json"
    run_logged "direct-$scenario" \
      env PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python meta_sg/scripts/evaluate_meta_sg_direct.py \
        --checkpoint "$ckpt" \
        --output-json "$out" \
        --scenario-set "$scenario" \
        --lambda-bd "$LAMBDA_BD" \
        --H "$H" \
        --num-clients "$NUM_CLIENTS" \
        --num-attackers "$NUM_ATTACKERS" \
        --subsample-rate "$SUBSAMPLE_RATE" \
        --client-samples "$CLIENT_SAMPLES" \
        --eval-samples "$EVAL_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --eval-batch-size "$EVAL_BATCH_SIZE" \
        --hidden-dim "$HIDDEN_DIM" \
        --defender-third-action "$DEFENDER_THIRD_ACTION" \
        --post-defense-mode "$POST_DEFENSE_MODE" \
        --neuroclip-eps-min "$NEUROCLIP_EPS_MIN" \
        --neuroclip-eps-max "$NEUROCLIP_EPS_MAX" \
        "${neuroclip_scale_args[@]}" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --rl-seed "$RL_SEED"
    parse_json "$out" "direct/$scenario" | tee -a "$LOG"
  done
}

run_adaptation_eval() {
  local ckpt="$1"
  local out_dir
  out_dir="$(dirname "$ckpt")"
  local scenario selection config
  for scenario in $ADAPT_SCENARIOS; do
    for selection in always guarded; do
      for config in $ADAPT_CONFIGS; do
        IFS=: read -r windows window_horizon updates <<<"$config"
        local tag="paper_online_${selection}_w${windows}_h${window_horizon}_u${updates}"
        local out="$out_dir/eval_h${H}_${scenario}_${tag}_paper_aligned_neuroclip.json"
        run_logged "adapt-$scenario-$tag" \
          env PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python meta_sg/scripts/evaluate_meta_sg_direct.py \
            --checkpoint "$ckpt" \
            --output-json "$out" \
            --scenario-set "$scenario" \
            --lambda-bd "$LAMBDA_BD" \
            --H "$H" \
            --num-clients "$NUM_CLIENTS" \
            --num-attackers "$NUM_ATTACKERS" \
            --subsample-rate "$SUBSAMPLE_RATE" \
            --client-samples "$CLIENT_SAMPLES" \
            --eval-samples "$EVAL_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --eval-batch-size "$EVAL_BATCH_SIZE" \
            --hidden-dim "$HIDDEN_DIM" \
            --defender-third-action "$DEFENDER_THIRD_ACTION" \
            --post-defense-mode "$POST_DEFENSE_MODE" \
            --neuroclip-eps-min "$NEUROCLIP_EPS_MIN" \
            --neuroclip-eps-max "$NEUROCLIP_EPS_MAX" \
            "${neuroclip_scale_args[@]}" \
            --device "$DEVICE" \
            --seed "$SEED" \
            --rl-seed "$RL_SEED" \
            --few-shot \
            --few-shot-method paper_online_td3 \
            --few-shot-selection "$selection" \
            --selection-margin 0.01 \
            --selection-horizon "$H" \
            --paper-online-windows "$windows" \
            --paper-online-window-horizon "$window_horizon" \
            --paper-online-updates-per-window "$updates" \
            --adaptation-warmup-steps 5 \
            --adaptation-lr-scale 0.25 \
            --adaptation-noise 0.05
        parse_json "$out" "adapt/$scenario/$tag" | tee -a "$LOG"
      done
    done
  done
}

main() {
  print_header | tee "$LOG"

  run_logged "train" \
    env PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python meta_sg/scripts/run_meta_sg_pretraining.py \
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
      --defender-third-action "$DEFENDER_THIRD_ACTION" \
      --post-defense-mode "$POST_DEFENSE_MODE" \
      --neuroclip-eps-min "$NEUROCLIP_EPS_MIN" \
      --neuroclip-eps-max "$NEUROCLIP_EPS_MAX" \
      "${neuroclip_scale_args[@]}" \
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
      --hidden-dim "$HIDDEN_DIM" \
      --device "$DEVICE" \
      --output-dir "$RUN_ROOT" \
      --log-interval "$LOG_INTERVAL" \
      --checkpoint-interval "$CHECKPOINT_INTERVAL"

  local ckpt
  ckpt="$(latest_checkpoint)"
  echo "[checkpoint] $ckpt" | tee -a "$LOG"
  run_direct_eval "$ckpt"
  run_adaptation_eval "$ckpt"
  echo "[all-done] $LOG" | tee -a "$LOG"
}

main "$@"
