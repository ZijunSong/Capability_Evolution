#!/usr/bin/env bash
# Keepalive for H20 0817 clean AUTO OPD monitor.
set -euo pipefail
trap '' HUP
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/h20_clean_auto_0817}"
PID_DIR="${OUT_ROOT}/pids"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${PID_DIR}" "${LOG_DIR}"

alive_pid() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

start_monitor() {
  setsid bash "${SCAPE_ROOT}/scripts/monitor_h20_clean_auto_0817.sh" \
    </dev/null >>"${LOG_DIR}/monitor.log" 2>&1 &
  local pid=$!
  echo "${pid}" >"${PID_DIR}/monitor.pid"
  echo "[$(date -Iseconds)] keepalive started monitor pid=${pid}"
}

echo "[$(date -Iseconds)] keepalive start pid=$$"
echo $$ >"${PID_DIR}/keepalive.pid"

while true; do
  phase="$(tr -d '[:space:]' < "${OUT_ROOT}/PHASE" 2>/dev/null || echo A)"
  if [[ "${phase}" == "STOP" || "${phase}" == "DONE" ]]; then
    echo "[$(date -Iseconds)] terminal ${phase} — keepalive exit"
    exit 0
  fi
  mp=$(cat "${PID_DIR}/monitor.pid" 2>/dev/null || true)
  if ! alive_pid "${mp}"; then
    echo "[$(date -Iseconds)] monitor dead — restart"
    start_monitor
  fi
  echo "[$(date -Iseconds)] keepalive tick phase=${phase} monitor=$(cat "${PID_DIR}/monitor.pid" 2>/dev/null || echo none)"
  sleep 120
done
