#!/usr/bin/env bash
# Resume incomplete Round 8 jobs
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup
for G in 0 1 2 3 4 5 6 7; do
  if [[ ! -f "${PID_DIR}/gpu${G}.pid" ]] || ! kill -0 "$(cat "${PID_DIR}/gpu${G}.pid")" 2>/dev/null; then
    scope8_log "Resuming GPU${G}"
    nohup bash "${REPO_ROOT}/scripts/scope_round8/run_gpu${G}_queue.sh" \
      >> "${LOG_DIR}/gpu${G}_resume.log" 2>&1 &
    echo $! > "${PID_DIR}/gpu${G}.pid"
  fi
done
