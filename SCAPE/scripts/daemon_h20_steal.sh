#!/usr/bin/env bash
# Poll free GPUs and dispatch stolen Phase-G jobs without blocking the monitor.
set -euo pipefail
trap '' HUP
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/h20_clean_auto_0817}"
LOG="${OUT_ROOT}/logs/steal_daemon.log"
PIDF="${OUT_ROOT}/pids/steal_daemon.pid"
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids"
echo $$ >"$PIDF"
echo "[$(date -Iseconds)] steal daemon start pid=$$" >>"$LOG"
while true; do
  phase="$(tr -d '[:space:]' < "${OUT_ROOT}/PHASE" 2>/dev/null || echo G)"
  if [[ "${phase}" == "DONE" || "${phase}" == "STOP" ]]; then
    echo "[$(date -Iseconds)] terminal ${phase} — steal daemon exit" >>"$LOG"
    exit 0
  fi
  DISPATCH_ONCE=1 OUT_ROOT="${OUT_ROOT}" bash "${SCAPE_ROOT}/scripts/steal_h20_phase_g_jobs.sh" >>"$LOG" 2>&1 || true
  sleep 45
done
