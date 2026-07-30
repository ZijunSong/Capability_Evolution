#!/usr/bin/env bash
# Wait for Barrier 1 offline eval, then auto-launch Barrier 2+3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/outputs/scope_round4/logs"
PID_FILE="${LOG_DIR}/barrier1_offline_eval.pid"

if [[ -f "${PID_FILE}" ]]; then
  OFFLINE_PID="$(cat "${PID_FILE}")"
  echo "[wait] polling offline eval PID=${OFFLINE_PID}"
  while kill -0 "${OFFLINE_PID}" 2>/dev/null; do
    sleep 60
    echo "[wait] $(date -Is) still running..."
    tail -1 "${LOG_DIR}/barrier1_offline_eval.log" 2>/dev/null || true
  done
  echo "[wait] offline eval done"
fi

cd "${REPO_ROOT}"
nohup bash scripts/scope_round4/run_barrier2_onward.sh \
  > "${LOG_DIR}/round4_barrier2_onward_nohup.log" 2>&1 &
echo $! > "${LOG_DIR}/barrier2_onward.pid"
echo "[auto] launched Barrier 2+3 PID=$(cat "${LOG_DIR}/barrier2_onward.pid")"
