#!/usr/bin/env bash
# Monitor Round 10 jobs; restart stuck GPU workers.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope10_setup

STALL_SEC="${STALL_SEC:-1800}"
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"

scope10_log "Watch started (stall=${STALL_SEC}s interval=${CHECK_INTERVAL}s)"

while true; do
  # Continuum supervisor alive?
  if ! pgrep -f 'round10_continuum.sh' >/dev/null; then
    if [[ ! -f "${MARKER_DIR}/barrier5_train.DONE" ]]; then
      scope10_log "WARN: continuum dead, restarting"
      nohup bash "$(dirname "$0")/round10_continuum.sh" \
        >> "${LOG_DIR}/continuum_restart.log" 2>&1 &
    fi
  fi

  for gpu in 0 1 2 3 4 5 6 7; do
    variant="${TRAINING_VARIANTS[$gpu]}"
    marker="${OUT}/training/${variant}/DONE"
    [[ -f "${marker}" ]] && continue

    worker_log="${LOG_DIR}/${variant}_train.log"
    if [[ ! -f "${worker_log}" ]]; then
      continue
    fi

    age=$(( $(date +%s) - $(stat -c %Y "${worker_log}" 2>/dev/null || echo 0) ))
    if [[ "${age}" -gt "${STALL_SEC}" ]]; then
      # Check if GPU worker process exists
      if ! pgrep -f "run_gpu.sh ${gpu}" >/dev/null && ! pgrep -f "run_training.py --variant ${variant}" >/dev/null; then
        scope10_log "RESTART gpu${gpu} ${variant} (stalled ${age}s)"
        bash "$(dirname "$0")/run_gpu.sh" "${gpu}" \
          >> "${LOG_DIR}/training_supervisor.log" 2>&1 &
      fi
    fi
  done

  sleep "${CHECK_INTERVAL}"
done
