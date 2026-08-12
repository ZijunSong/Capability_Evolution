#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

SKIP_BUILD="${SKIP_BUILD:-0}"
if [[ "${SKIP_BUILD}" != "1" ]]; then
  python training/scope_round9/build_frozen_replay.py 2>&1 | tee "${LOG_DIR}/build_frozen_replay.log"
fi

# Touch a wave-level start marker so watchdog ignores previous-run logs.
date -Is > "${MARKER_DIR}/wave_a_STARTED"
scope9_log "Wave A launching 8 GPU workers"

for gpu in 0 1 2 3 4 5 6 7; do
  nohup bash "$(dirname "$0")/run_wave_a_gpu.sh" "${gpu}" \
    >> "${LOG_DIR}/wave_a_${WAVE_A_VARIANTS[$gpu]}_worker.log" 2>&1 &
  scope9_log "Wave A started gpu=${gpu} variant=${WAVE_A_VARIANTS[$gpu]} pid=$!"
  sleep 2
done
wait
done_n=$(find "${MARKER_DIR}" -maxdepth 1 -name 'wave_a_*.DONE' 2>/dev/null | wc -l | tr -d ' ')
if [[ "${done_n}" -ne 8 ]]; then
  scope9_log "ERROR: Wave A incomplete markers=${done_n}/8"
  exit 1
fi
scope9_log "Wave A all GPUs complete"
