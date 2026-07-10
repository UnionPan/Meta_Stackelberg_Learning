#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PREVIOUS_UNIT="${PREVIOUS_UNIT:-meta-sg-resume50-iter25-20260707-000752.service}"
RUN_ROOT="${RUN_ROOT:-runs/meta_sg_goal/curriculum_global_backdoor_mixed_h200_t50_c30_a6}"
POLL_SECONDS="${POLL_SECONDS:-120}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/job_logs"

stamp="$(date +%Y%m%d-%H%M%S)"
log="$RUN_ROOT/job_logs/train_curriculum_after_current_${stamp}.log"
export RUN_ROOT

{
  echo "[$(date '+%F %T %Z')] queued curriculum service started"
  echo "waiting for ${PREVIOUS_UNIT} to become inactive"

  while systemctl --user is-active --quiet "$PREVIOUS_UNIT"; do
    echo "[$(date '+%F %T %Z')] previous experiment still active; sleeping ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done

  echo "[$(date '+%F %T %Z')] previous experiment inactive; launching curriculum"
  exec "$SCRIPT_DIR/run_4d_curriculum_global_backdoor_mixed_h200_job.sh"
} >> "$log" 2>&1
