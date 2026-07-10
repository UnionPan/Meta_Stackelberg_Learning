#!/usr/bin/env bash
set -euo pipefail

# Run paper-style online TD3 adaptation ablations and print a live, compact
# progress view. Defaults match the current H=200 paper-like mixed-backdoor run.
#
# Example:
#   CHECKPOINT=runs/.../checkpoints/iter_0050 \
#     bash meta_sg/scripts/run_paper_online_ablation.sh
#
# Useful overrides:
#   SCENARIO_SETS="backdoor" CONFIGS="5:40:10 10:20:10" bash ...
#   FORCE=1 DEVICE=cuda:0 bash ...

RUN_DIR="${RUN_DIR:-runs/meta_sg_goal/clean_backdoor_mixed_paperlike_t100_k10_h200_l10/20260704-100848}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/checkpoints/iter_0050}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/job_logs}"

SCENARIO_SETS="${SCENARIO_SETS:-backdoor clean_mixed_backdoor}"
# Format: windows:window_horizon:updates_per_window
CONFIGS="${CONFIGS:-5:40:10 10:20:10}"
SELECTION="${SELECTION:-always}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-30}"

DEVICE="${DEVICE:-cuda:0}"
H="${H:-200}"
NUM_CLIENTS="${NUM_CLIENTS:-30}"
NUM_ATTACKERS="${NUM_ATTACKERS:-6}"
SUBSAMPLE_RATE="${SUBSAMPLE_RATE:-0.2}"
CLIENT_SAMPLES="${CLIENT_SAMPLES:-64}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
DEFENDER_THIRD_ACTION="${DEFENDER_THIRD_ACTION:-server_lr}"
POST_DEFENSE_MODE="${POST_DEFENSE_MODE:-weight_copy}"
SEED="${SEED:-42}"
RL_SEED="${RL_SEED:-506}"
LAMBDA_BD="${LAMBDA_BD:-1.0}"
SELECTION_MARGIN="${SELECTION_MARGIN:-0.01}"
ADAPTATION_WARMUP_STEPS="${ADAPTATION_WARMUP_STEPS:-5}"
ADAPTATION_LR_SCALE="${ADAPTATION_LR_SCALE:-0.25}"
ADAPTATION_NOISE="${ADAPTATION_NOISE:-0.05}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_LOG="$LOG_DIR/paper_online_ablation_${STAMP}.log"
SUMMARY_TSV="$OUT_DIR/paper_online_ablation_${STAMP}.tsv"

if [[ ! -e "$CHECKPOINT" ]]; then
  echo "[error] checkpoint not found: $CHECKPOINT" >&2
  echo "Set CHECKPOINT=/path/to/checkpoint_dir_or_defender_meta.pt" >&2
  exit 2
fi

print_config() {
  cat <<EOF
[paper-online-ablation]
  checkpoint: $CHECKPOINT
  out_dir:    $OUT_DIR
  run_log:    $RUN_LOG
  summary:    $SUMMARY_TSV
  scenarios:  $SCENARIO_SETS
  configs:    $CONFIGS
  selection:  $SELECTION
  heartbeat:  ${HEARTBEAT_INTERVAL}s
  H:          $H
  clients:    $NUM_CLIENTS, attackers: $NUM_ATTACKERS, subsample: $SUBSAMPLE_RATE
EOF
}

parse_json() {
  local json_path="$1"
  local label="$2"
  python - "$json_path" "$label" "$SUMMARY_TSV" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
summary = Path(sys.argv[3])

data = json.loads(path.read_text())
if not summary.exists():
    summary.write_text(
        "label\tscenario\tbase_clean\tadapt_clean\tdelta_clean\t"
        "base_asr\tadapt_asr\tdelta_asr\twindows\twindow_horizon\tupdates\n"
    )

print(f"[result] {label}")
for row in data:
    fs = row.get("few_shot_adaptation") or {}
    ev = fs.get("evaluation") or fs.get("adapted_evaluation") or {}
    style = fs.get("paper_style") or {}
    base_clean = float(row.get("final_clean_acc") or 0.0)
    adapt_clean = float(ev.get("final_clean_acc") or 0.0)
    base_asr = float(row.get("final_backdoor_acc") or 0.0)
    adapt_asr = float(ev.get("final_backdoor_acc") or 0.0)
    windows = style.get("online_windows", "")
    window_horizon = style.get("window_horizon", "")
    updates = style.get("updates_per_window", "")
    print(
        f"  {row['scenario']:<14} clean {base_clean:.3f}->{adapt_clean:.3f} "
        f"({adapt_clean - base_clean:+.3f}) | "
        f"ASR {base_asr:.3f}->{adapt_asr:.3f} ({adapt_asr - base_asr:+.3f})"
    )
    with summary.open("a") as fh:
        fh.write(
            f"{label}\t{row['scenario']}\t{base_clean:.6f}\t{adapt_clean:.6f}\t"
            f"{adapt_clean - base_clean:.6f}\t{base_asr:.6f}\t{adapt_asr:.6f}\t"
            f"{adapt_asr - base_asr:.6f}\t{windows}\t{window_horizon}\t{updates}\n"
        )
PY
}

run_one() {
  local scenario_set="$1"
  local windows="$2"
  local window_horizon="$3"
  local updates="$4"
  local idx="$5"
  local total="$6"
  local tag="paper_online_${SELECTION}_w${windows}_h${window_horizon}_u${updates}"
  local out_json="$OUT_DIR/eval_h${H}_${scenario_set}_${tag}.json"
  local label="${scenario_set}/${tag}"

  echo
  echo "[$idx/$total] start $label $(date -Is)"
  echo "  output: $out_json"

  if [[ -s "$out_json" && "$FORCE" != "1" ]]; then
    echo "  skip existing output; set FORCE=1 to rerun"
    parse_json "$out_json" "$label"
    return
  fi

  local cmd=(
    env PYTHONUNBUFFERED=1 PYTHONPATH=.
    .venv/bin/python meta_sg/scripts/evaluate_meta_sg_direct.py
    --checkpoint "$CHECKPOINT"
    --output-json "$out_json"
    --scenario-set "$scenario_set"
    --lambda-bd "$LAMBDA_BD"
    --H "$H"
    --num-clients "$NUM_CLIENTS"
    --num-attackers "$NUM_ATTACKERS"
    --subsample-rate "$SUBSAMPLE_RATE"
    --client-samples "$CLIENT_SAMPLES"
    --eval-samples "$EVAL_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --hidden-dim "$HIDDEN_DIM"
    --defender-third-action "$DEFENDER_THIRD_ACTION"
    --post-defense-mode "$POST_DEFENSE_MODE"
    --device "$DEVICE"
    --seed "$SEED"
    --rl-seed "$RL_SEED"
    --few-shot
    --few-shot-method paper_online_td3
    --few-shot-selection "$SELECTION"
    --selection-margin "$SELECTION_MARGIN"
    --selection-horizon "$H"
    --paper-online-windows "$windows"
    --paper-online-window-horizon "$window_horizon"
    --paper-online-updates-per-window "$updates"
    --adaptation-warmup-steps "$ADAPTATION_WARMUP_STEPS"
    --adaptation-lr-scale "$ADAPTATION_LR_SCALE"
    --adaptation-noise "$ADAPTATION_NOISE"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  dry-run:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return
  fi

  local start_epoch
  start_epoch="$(date +%s)"
  "${cmd[@]}" > >(tee -a "$RUN_LOG") 2>&1 &
  local pid=$!
  echo "  pid: $pid"

  if [[ "$HEARTBEAT_INTERVAL" -gt 0 ]]; then
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$HEARTBEAT_INTERVAL"
      if kill -0 "$pid" 2>/dev/null; then
        local now elapsed size proc
        now="$(date +%s)"
        elapsed=$((now - start_epoch))
        if [[ -e "$out_json" ]]; then
          size="$(du -h "$out_json" 2>/dev/null | awk '{print $1}')"
        else
          size="pending"
        fi
        proc="$(ps -o %cpu=,%mem=,rss= -p "$pid" 2>/dev/null | awk '{printf "cpu=%s%% mem=%s%% rss=%sKB", $1, $2, $3}')"
        echo "[heartbeat] $label elapsed=${elapsed}s pid=$pid ${proc:-running} output=$size" | tee -a "$RUN_LOG"
      fi
    done
  fi

  local code
  set +e
  wait "$pid"
  code=$?
  set -e
  echo "[$idx/$total] exit_code=$code $label $(date -Is)" | tee -a "$RUN_LOG"
  if [[ "$code" -ne 0 ]]; then
    exit "$code"
  fi

  parse_json "$out_json" "$label" | tee -a "$RUN_LOG"
}

main() {
  print_config | tee "$RUN_LOG"
  local total=0
  local scenario_set config
  for scenario_set in $SCENARIO_SETS; do
    for config in $CONFIGS; do
      total=$((total + 1))
    done
  done

  local idx=0
  for scenario_set in $SCENARIO_SETS; do
    for config in $CONFIGS; do
      IFS=: read -r windows window_horizon updates <<<"$config"
      if [[ -z "${windows:-}" || -z "${window_horizon:-}" || -z "${updates:-}" ]]; then
        echo "[error] bad CONFIGS entry: $config; expected windows:window_horizon:updates" >&2
        exit 2
      fi
      idx=$((idx + 1))
      run_one "$scenario_set" "$windows" "$window_horizon" "$updates" "$idx" "$total"
    done
  done

  echo
  echo "[done] summary TSV: $SUMMARY_TSV"
  echo "[done] run log:     $RUN_LOG"
}

main "$@"
