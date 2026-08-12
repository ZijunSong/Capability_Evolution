#!/usr/bin/env bash
# Launch all 8 GPU queues (background supervisors)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

r14_log "launch_all_8gpu: starting supervisors"

for gpu in 0 1 2 3 4 5 6 7; do
  script="$(dirname "$0")/run_gpu${gpu}_queue.sh"
  if [[ ! -x "${script}" ]]; then
    chmod +x "${script}"
  fi
  r14_stop_recorded "gpu${gpu}"
  nohup bash "${script}" "${gpu}" \
    >> "${LOG_DIR}/gpu${gpu}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/gpu${gpu}.pid"
  r14_log "launched GPU${gpu} pid=$(cat "${PID_DIR}/gpu${gpu}.pid")"
  sleep 3
done

# Start guardian + monitor
nohup bash "$(dirname "$0")/guardian.sh" >> "${LOG_DIR}/guardian.log" 2>&1 &
echo $! > "${PID_DIR}/guardian.pid"
nohup bash "$(dirname "$0")/monitor_loop.sh" >> "${LOG_DIR}/monitor.log" 2>&1 &
echo $! > "${PID_DIR}/monitor.pid"

r14_log "launch_all_8gpu: all supervisors started"
