#!/usr/bin/env bash
# Round14 continuum:
#   Phase A: wave0 Dup calibrate on GPU0-4 + GPU5/6/7 capability queues
#   Phase B: after wave0, run GPU0 post (aggregate/830) + GPU1-4 capability queues
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

r14_log "continuum: start"

# Guardian / monitor
r14_stop_recorded guardian
r14_stop_recorded monitor
nohup bash "$(dirname "$0")/guardian.sh" >> "${LOG_DIR}/guardian.log" 2>&1 &
echo $! > "${PID_DIR}/guardian.pid"
nohup bash "$(dirname "$0")/monitor_loop.sh" >> "${LOG_DIR}/monitor.log" 2>&1 &
echo $! > "${PID_DIR}/monitor.pid"

# Phase A — wave0 uses GPU0-4
r14_stop_recorded wave0
nohup bash "$(dirname "$0")/launch_wave0_dup_calibrate.sh" \
  >> "${LOG_DIR}/wave0_supervisor.log" 2>&1 &
echo $! > "${PID_DIR}/wave0.pid"
r14_log "continuum: launched wave0 pid=$(cat "${PID_DIR}/wave0.pid")"

# Phase A — free GPUs 5/6/7 start their assigned work immediately
for gpu in 5 6 7; do
  script="$(dirname "$0")/run_gpu${gpu}_queue.sh"
  chmod +x "${script}"
  r14_stop_recorded "gpu${gpu}"
  nohup bash "${script}" "${gpu}" >> "${LOG_DIR}/gpu${gpu}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/gpu${gpu}.pid"
  r14_log "continuum: launched GPU${gpu} pid=$(cat "${PID_DIR}/gpu${gpu}.pid")"
  sleep 2
done

# Wait for wave0 completion (DONE markers or wave0 script exit)
r14_log "continuum: waiting for wave0..."
while true; do
  if r14_wave0_complete; then
    r14_log "continuum: wave0 markers complete"
    break
  fi
  if [[ -f "${PID_DIR}/wave0.pid" ]]; then
    wpid="$(cat "${PID_DIR}/wave0.pid")"
    if ! kill -0 "${wpid}" 2>/dev/null; then
      r14_log "continuum: wave0 process exited; checking markers"
      if r14_wave0_complete; then
        break
      fi
      r14_log "continuum: wave0 exited incomplete — guardian may restart; sleep"
    fi
  fi
  sleep 60
done

# Phase B — GPU0 post + GPU1-4 capability queues
for gpu in 0 1 2 3 4; do
  script="$(dirname "$0")/run_gpu${gpu}_queue.sh"
  chmod +x "${script}"
  # Avoid double-start if already running
  if [[ -f "${PID_DIR}/gpu${gpu}.pid" ]] && kill -0 "$(cat "${PID_DIR}/gpu${gpu}.pid")" 2>/dev/null; then
    r14_log "continuum: GPU${gpu} already running — skip"
    continue
  fi
  r14_stop_recorded "gpu${gpu}"
  nohup bash "${script}" "${gpu}" >> "${LOG_DIR}/gpu${gpu}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/gpu${gpu}.pid"
  r14_log "continuum: launched GPU${gpu} pid=$(cat "${PID_DIR}/gpu${gpu}.pid")"
  sleep 5
done

r14_log "continuum: all phases launched"
echo "CONTINUUM_LAUNCHED $(date -Is)" > "${MARKER_DIR}/continuum_launched"
