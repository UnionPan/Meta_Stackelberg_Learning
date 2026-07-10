#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  meta_sg/scripts/submit_meta_sg_job.sh [options] -- JOB [ARG...]

Options:
  --unit NAME       systemd unit name. Defaults to meta-sg-YYYYmmdd-HHMMSS-PID.
  --log PATH        stdout/stderr log path. Defaults to RUN_ROOT/job_logs/UNIT.log.
  --pid-file PATH   pid file used by the setsid fallback.
  --dry-run         print the command that would be launched.
  -h, --help        show this help.

Environment:
  META_SG_SUBMIT_DISABLE_SYSTEMD=1 forces the setsid fallback.
  META_SG_SUBMIT_EXTRA_ENV="VAR1 VAR2" passes additional environment variables.
EOF
}

quote_args() {
  local quoted out=()
  for arg in "$@"; do
    printf -v quoted "%q" "$arg"
    out+=("$quoted")
  done
  printf '%s' "${out[*]}"
}

print_command() {
  local quoted out=()
  for arg in "$@"; do
    printf -v quoted "%q" "$arg"
    out+=("$quoted")
  done
  printf '%s\n' "${out[*]}"
}

add_setenv_arg() {
  local name="$1"
  local value
  if [[ ! "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    return
  fi
  if [[ -n "${!name+x}" ]]; then
    value="${!name}"
    systemd_args+=("--setenv=$name=$value")
  fi
}

sanitize_unit_part() {
  local value="$1"
  value="${value//[^A-Za-z0-9_.-]/-}"
  printf '%s' "$value"
}

unit=""
log_path=""
pid_file=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit)
      [[ $# -ge 2 ]] || { echo "--unit requires a value" >&2; exit 2; }
      unit="$2"
      shift 2
      ;;
    --log)
      [[ $# -ge 2 ]] || { echo "--log requires a value" >&2; exit 2; }
      log_path="$2"
      shift 2
      ;;
    --pid-file)
      [[ $# -ge 2 ]] || { echo "--pid-file requires a value" >&2; exit 2; }
      pid_file="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "missing JOB command after --" >&2
  usage
  exit 2
fi

stamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "$unit" ]]; then
  unit="meta-sg-${stamp}-$$"
else
  unit="$(sanitize_unit_part "$unit")"
fi

run_root="${RUN_ROOT:-runs/meta_sg_goal}"
if [[ -z "$log_path" ]]; then
  log_path="$run_root/job_logs/${unit}.log"
fi
if [[ -z "$pid_file" ]]; then
  pid_file="$run_root/job_logs/${unit}.pid"
fi

log_dir="${log_path%/*}"
pid_dir="${pid_file%/*}"
if [[ "$log_dir" == "$log_path" ]]; then
  log_dir="."
fi
if [[ "$pid_dir" == "$pid_file" ]]; then
  pid_dir="."
fi
mkdir -p "$log_dir" "$pid_dir"

job_command="$(quote_args "$@")"
printf -v quoted_log "%q" "$log_path"
job_shell="exec $job_command >> $quoted_log 2>&1"

use_systemd=0
if [[ "${META_SG_SUBMIT_DISABLE_SYSTEMD:-0}" != "1" ]] && command -v systemd-run >/dev/null 2>&1; then
  use_systemd=1
fi

if [[ "$use_systemd" -eq 1 ]]; then
  systemd_args=(systemd-run --user "--unit=$unit" --same-dir --collect)
  default_env_vars=(
    HOME USER LOGNAME SHELL PATH LANG LC_ALL LC_CTYPE LC_NUMERIC
    CUDA_VISIBLE_DEVICES OMP_NUM_THREADS MKL_NUM_THREADS PYTHONUNBUFFERED
    PYTHONPATH
    RUN_ROOT RUN_DIR CHECKPOINT EVAL_DIR DEVICE
    RESUME_FROM START_ITERATION TOTAL_ITERATIONS
    T K H L SUPPORT_EPISODES QUERY_HORIZON META_STEP
    CHECKPOINT_INTERVAL LOG_INTERVAL
    NUM_CLIENTS NUM_ATTACKERS SUBSAMPLE_RATE CLIENT_SAMPLES EVAL_SAMPLES
    BATCH_SIZE EVAL_BATCH_SIZE BUFFER_CAPACITY
    FL_PARALLEL_CLIENTS FL_NUM_WORKERS HIDDEN_DIM
    SERVER_LR_MIN SERVER_LR_MAX SERVER_LR_PENALTY_WEIGHT
    NEUROCLIP_EPS_MIN NEUROCLIP_EPS_MAX
    ADAPT_WINDOWS ADAPT_WINDOW_HORIZON ADAPT_UPDATES_PER_WINDOW ADAPT_BATCH_SIZE
    PROXY_REWARD_MODE PROXY_SERVER_LR_OFFSET_STEP PROXY_SERVER_LR_OFFSET_MAX_STEPS
  )
  for env_name in "${default_env_vars[@]}"; do
    add_setenv_arg "$env_name"
  done
  normalized_extra_env="${META_SG_SUBMIT_EXTRA_ENV:-}"
  normalized_extra_env="${normalized_extra_env//,/ }"
  for env_name in $normalized_extra_env; do
    add_setenv_arg "$env_name"
  done
  bash_path="$(command -v bash)"
  systemd_args+=("$bash_path" -lc "$job_shell")

  if [[ "$dry_run" -eq 1 ]]; then
    print_command "${systemd_args[@]}"
    exit 0
  fi

  "${systemd_args[@]}"
  printf '%s\n' "$unit" > "$pid_file"
  echo "[submit] systemd unit=$unit"
  echo "[submit] log=$log_path"
  echo "[submit] unit-name-file=$pid_file"
  exit 0
fi

echo "[submit] systemd-run --user unavailable or disabled; falling back to setsid" >&2
fallback_args=(setsid bash -lc "$job_shell")
if [[ "$dry_run" -eq 1 ]]; then
  print_command "${fallback_args[@]}"
  exit 0
fi

"${fallback_args[@]}" >/dev/null 2>&1 < /dev/null &
fallback_pid="$!"
printf '%s\n' "$fallback_pid" > "$pid_file"
echo "[submit] fallback pid=$fallback_pid"
echo "[submit] log=$log_path"
echo "[submit] pid-file=$pid_file"
