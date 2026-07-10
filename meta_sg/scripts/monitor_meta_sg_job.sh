#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RUN_ROOT JOB_PID MON_LOG JOB_LOG" >&2
  exit 2
fi

RUN_ROOT="$1"
JOB_PID="$2"
MON_LOG="$3"
JOB_LOG="$4"

descendant_pids() {
  local root="$1"
  local children child
  children="$(pgrep -P "$root" 2>/dev/null || true)"
  for child in $children; do
    printf '%s\n' "$child"
    descendant_pids "$child"
  done
}

log_process_snapshot() {
  local pids
  pids="$(printf '%s\n' "$JOB_PID"; descendant_pids "$JOB_PID")"
  echo "--- memory ---"
  free -h || true
  echo "--- process tree ---"
  if command -v pstree >/dev/null 2>&1; then
    pstree -ap "$JOB_PID" || true
  fi
  echo "--- process rss ---"
  if [[ -n "$pids" ]]; then
    ps -o pid,ppid,stat,etime,%cpu,%mem,rss,vsz,args -p $(echo "$pids") || true
  fi
  echo "--- python processes ---"
  ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,vsz,args | rg 'python|run_meta_sg|evaluate_meta_sg' || true
}

while kill -0 "$JOB_PID" 2>/dev/null; do
  {
    echo "===== $(date -Is) ====="
    echo "job_pid=$JOB_PID"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits || true
    log_process_snapshot
    df -h /home/antik/rl || true
    du -sh "$RUN_ROOT" 2>/dev/null || true
    printf "checkpoint_dirs="
    find "$RUN_ROOT" -type d -path "*/checkpoints/iter_*" 2>/dev/null | wc -l
    printf "latest_final_dirs="
    find "$RUN_ROOT" -maxdepth 3 -type d \( -name latest -o -name final \) 2>/dev/null | wc -l
    echo "--- job tail ---"
    tail -n 12 "$JOB_LOG" 2>/dev/null || true
    echo
  } >> "$MON_LOG"
  sleep "${MONITOR_INTERVAL:-300}"
done

{
  echo "===== $(date -Is) job_finished ====="
  ps -p "$JOB_PID" -o pid,stat,etime,args || true
  log_process_snapshot
  df -h /home/antik/rl || true
  du -sh "$RUN_ROOT" 2>/dev/null || true
  find "$RUN_ROOT" -maxdepth 3 -type d \( -name final -o -path "*/checkpoints/iter_*" \) 2>/dev/null | sort | tail -40
  echo "--- final job tail ---"
  tail -n 80 "$JOB_LOG" 2>/dev/null || true
} >> "$MON_LOG"
