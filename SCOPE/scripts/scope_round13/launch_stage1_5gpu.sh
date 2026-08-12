#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

r13_log "Launch Stage1 training on GPU0-4"
for gpu in 0 1 2 3 4; do
  variant="${STAGE1_VARIANTS[$gpu]}"
  nohup bash "$(dirname "$0")/run_stage1_gpu.sh" "${gpu}" "${variant}" \
    >> "${LOG_DIR}/stage1_${variant}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/stage1_gpu${gpu}.pid"
  r13_log "started ${variant} on GPU${gpu} pid=$!"
  sleep 5
done
