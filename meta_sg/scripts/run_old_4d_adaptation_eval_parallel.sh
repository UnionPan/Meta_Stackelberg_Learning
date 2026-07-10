#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-runs/meta_sg_goal/4d_both_clean_global_backdoor_mixed_h200_t100_k8_l10_c30_a6}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/20260707-000755}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/final}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
EVAL_DIR="${EVAL_DIR:-$RUN_DIR/eval/adaptation_parallel_${STAMP}}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
WAIT_FOR_UNIT="${WAIT_FOR_UNIT:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-120}"

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
SEED="${SEED:-502}"
RL_SEED="${RL_SEED:-506}"

ADAPT_WINDOWS="${ADAPT_WINDOWS:-10}"
ADAPT_WINDOW_HORIZON="${ADAPT_WINDOW_HORIZON:-20}"
ADAPT_UPDATES_PER_WINDOW="${ADAPT_UPDATES_PER_WINDOW:-10}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-32}"
PROXY_REWARD_MODE="${PROXY_REWARD_MODE:-clean_update_anomaly}"
PROXY_SERVER_LR_OFFSET_STEP="${PROXY_SERVER_LR_OFFSET_STEP:-0.5}"
PROXY_SERVER_LR_OFFSET_MAX_STEPS="${PROXY_SERVER_LR_OFFSET_MAX_STEPS:-4}"

SCENARIOS_CSV="${SCENARIOS_CSV:-clean,ipm,lmp,rl,bfl,dba,rl_backdoor,mixed_backdoor}"
IFS=',' read -r -a SCENARIOS <<< "$SCENARIOS_CSV"

mkdir -p "$EVAL_DIR"/{json,logs}

if [[ -n "$WAIT_FOR_UNIT" ]]; then
  echo "[adapt-eval] waiting for $WAIT_FOR_UNIT to become inactive"
  while systemctl --user is-active --quiet "$WAIT_FOR_UNIT"; do
    echo "[adapt-eval] $(date '+%F %T %Z') $WAIT_FOR_UNIT active; sleeping ${WAIT_POLL_SECONDS}s"
    sleep "$WAIT_POLL_SECONDS"
  done
fi

if [[ ! -f "$CHECKPOINT/defender_meta.pt" && ! -f "$CHECKPOINT" ]]; then
  echo "[adapt-eval] checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

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
  --few-shot
  --few-shot-method paper_online_proxy_td3
  --adaptation-attacker-source native
  --paper-online-windows "$ADAPT_WINDOWS"
  --paper-online-window-horizon "$ADAPT_WINDOW_HORIZON"
  --paper-online-updates-per-window "$ADAPT_UPDATES_PER_WINDOW"
  --adaptation-batch-size "$ADAPT_BATCH_SIZE"
  --proxy-reward-mode "$PROXY_REWARD_MODE"
  --proxy-server-lr-offset-step "$PROXY_SERVER_LR_OFFSET_STEP"
  --proxy-server-lr-offset-max-steps "$PROXY_SERVER_LR_OFFSET_MAX_STEPS"
)

echo "[adapt-eval] start $(date -Is)"
echo "[adapt-eval] checkpoint=$CHECKPOINT"
echo "[adapt-eval] eval_dir=$EVAL_DIR"
echo "[adapt-eval] scenarios=${SCENARIOS[*]}"
echo "[adapt-eval] max_parallel=$MAX_PARALLEL device=$DEVICE"

running=0
pids=()
for scenario in "${SCENARIOS[@]}"; do
  scenario="$(echo "$scenario" | xargs)"
  [[ -n "$scenario" ]] || continue
  json="$EVAL_DIR/json/${scenario}.json"
  log="$EVAL_DIR/logs/${scenario}.log"
  (
    echo "[adapt-eval:$scenario] start $(date -Is)"
    PYTHONPATH=. .venv/bin/python meta_sg/scripts/evaluate_meta_sg_direct.py \
      "${common_args[@]}" \
      --scenario-filter "$scenario" \
      --output-json "$json"
    echo "[adapt-eval:$scenario] done $(date -Is)"
  ) > "$log" 2>&1 &
  pids+=("$!")
  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n
    running=$((running - 1))
  fi
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "[adapt-eval] at least one scenario failed; see $EVAL_DIR/logs" >&2
  exit 1
fi

EVAL_DIR="$EVAL_DIR" .venv/bin/python - <<'PY'
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

eval_dir = Path(os.environ["EVAL_DIR"])
json_dir = eval_dir / "json"
records = []
for path in sorted(json_dir.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"expected list in {path}")
    for record in payload:
        record["_source_json"] = str(path)
        records.append(record)

combined_json = eval_dir / "combined_adaptation_eval.json"
combined_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def get(mapping, path, default=""):
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def f(value):
    if value == "":
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(value):
        return ""
    return value

rows = []
for record in sorted(records, key=lambda item: str(item.get("scenario", ""))):
    few = record.get("few_shot_adaptation", {}) or {}
    adapted = few.get("adapted_evaluation", {}) or {}
    selected = few.get("evaluation", {}) or {}
    transition = few.get("transition", {}) or {}
    to_action = transition.get("to", {}) or {}
    selection = few.get("selection", {}) or {}
    window_records = few.get("window_records", []) or []
    last_window = window_records[-1] if window_records else {}
    rows.append(
        {
            "scenario": record.get("scenario", ""),
            "attack_type": record.get("attack_type", ""),
            "base_final_clean_acc": f(record.get("final_clean_acc", "")),
            "base_final_asr": f(record.get("final_backdoor_acc", "")),
            "base_final_defense_score": f(record.get("final_defense_score", "")),
            "base_mean_defender_reward": f(record.get("mean_defender_reward", "")),
            "adapted_final_clean_acc": f(adapted.get("final_clean_acc", "")),
            "adapted_final_asr": f(adapted.get("final_backdoor_acc", "")),
            "adapted_final_defense_score": f(adapted.get("final_defense_score", "")),
            "adapted_mean_defender_reward": f(adapted.get("mean_defender_reward", "")),
            "selected_final_clean_acc": f(selected.get("final_clean_acc", "")),
            "selected_final_asr": f(selected.get("final_backdoor_acc", "")),
            "selected_final_defense_score": f(selected.get("final_defense_score", "")),
            "gain_clean_acc": f(record.get("few_shot_gain_clean_acc", "")),
            "gain_asr": f(record.get("few_shot_gain_backdoor_acc", "")),
            "gain_defense_score": f(record.get("few_shot_gain_defense_score", "")),
            "selection": selection.get("selected", ""),
            "selection_accepted": selection.get("accepted", ""),
            "method": few.get("method", ""),
            "online_windows": get(few, ["paper_style", "online_windows"]),
            "window_horizon": get(few, ["paper_style", "window_horizon"]),
            "updates_per_window": get(few, ["paper_style", "updates_per_window"]),
            "num_transitions": few.get("num_transitions", ""),
            "num_updates": few.get("num_updates", ""),
            "adapted_alpha": f(to_action.get("alpha", "")),
            "adapted_beta": f(to_action.get("beta", "")),
            "adapted_neuroclip": f(to_action.get("neuroclip", to_action.get("post_param", ""))),
            "adapted_server_lr": f(to_action.get("server_lr", "")),
            "last_window_clean_acc": f(last_window.get("clean_acc", "")),
            "last_window_asr": f(last_window.get("backdoor_acc", "")),
            "last_window_mean_env_reward": f(last_window.get("mean_environment_reward", "")),
            "source_json": record.get("_source_json", ""),
        }
    )

summary_csv = eval_dir / "adaptation_eval_summary.csv"
fieldnames = list(rows[0].keys()) if rows else []
with summary_csv.open("w", encoding="utf-8", newline="") as fobj:
    writer = csv.DictWriter(fobj, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"combined_json={combined_json}")
print(f"summary_csv={summary_csv}")
print(f"num_records={len(rows)}")
PY

echo "[adapt-eval] done $(date -Is)"
