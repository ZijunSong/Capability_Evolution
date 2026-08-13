#!/usr/bin/env bash
# Monitor learnability audit; relaunch failed GPU queues.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/learnability_audit}"
PID_DIR="${OUT_ROOT}/pids"
LAUNCH="${SCAPE_ROOT}/scripts/launch_learnability_audit_8gpu.sh"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"

check_gpu_done() {
  local g="$1"
  [[ -f "${OUT_ROOT}/gpu${g}/ALL_DONE" ]]
}

relaunch_failed() {
  for g in 0 1 2 3 4 5 6 7; do
    if check_gpu_done "$g"; then
      continue
    fi
    pf="${PID_DIR}/gpu${g}.pid"
    if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
      continue
    fi
    echo "[monitor] relaunch gpu${g}"
    GPU_ONLY="${g}" OUT_ROOT="${OUT_ROOT}" bash "${LAUNCH}" &
    echo $! >"${PID_DIR}/gpu${g}.pid"
  done
}

all_done() {
  for g in 0 1 2 3 4 5 6 7; do
    if ! check_gpu_done "$g"; then
      return 1
    fi
  done
  return 0
}

while true; do
  echo "[$(date -Iseconds)] monitor tick"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
  relaunch_failed
  if all_done; then
    echo "[monitor] ALL GPU queues done — aggregating"
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_learnability_audit.py" \
      --out-dir "${OUT_ROOT}"
    break
  fi
  sleep 120
done
