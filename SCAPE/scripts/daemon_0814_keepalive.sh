#!/usr/bin/env bash
# Session-independent keepalive for 0814 Clean Mechanism.
# Restarts monitor if it dies. Does not touch zyt processes.
set -euo pipefail
trap '' HUP
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/0814_clean_mechanism}"
PID_DIR="${OUT_ROOT}/pids"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${PID_DIR}" "${LOG_DIR}"

alive_pid() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

start_monitor() {
  local phase="${1:-c0}"
  PHASE="${phase}" OUT_ROOT="${OUT_ROOT}" \
    setsid bash "${SCAPE_ROOT}/scripts/monitor_0814_clean.sh" \
    </dev/null >>"${LOG_DIR}/monitor.log" 2>&1 &
  local pid=$!
  echo "${pid}" >"${PID_DIR}/monitor.pid"
  echo "[$(date -Iseconds)] keepalive started monitor pid=${pid} phase=${phase}"
}

echo "[$(date -Iseconds)] keepalive start pid=$$ sid=$(ps -o sid= -p $$)"
echo $$ >"${PID_DIR}/keepalive.pid"

while true; do
  mp=$(cat "${PID_DIR}/monitor.pid" 2>/dev/null || true)
  if ! alive_pid "${mp}"; then
    phase=c0
    if [[ -f "${OUT_ROOT}/sft/gpu0/ALL_DONE" && -f "${OUT_ROOT}/sft/gpu2/ALL_DONE" ]]; then
      # C0 trainers done; if C2 not finished, monitor should be in c2
      if [[ ! -f "${OUT_ROOT}/micro/gpu0/ALL_DONE" ]]; then
        phase=c2
      fi
    fi
    echo "[$(date -Iseconds)] monitor dead — restart phase=${phase}"
    start_monitor "${phase}"
  fi
  n_sft=$(pgrep -fc "run_clean_sft_cell.py" || true)
  n_q=$(pgrep -fc "launch_0814_clean_c0.sh" || true)
  echo "[$(date -Iseconds)] keepalive tick sft_py=${n_sft} c0_queues=${n_q} monitor=$(cat "${PID_DIR}/monitor.pid" 2>/dev/null || echo none)"
  sleep 120
done
