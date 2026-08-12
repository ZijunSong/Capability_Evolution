#!/usr/bin/env bash
# Launch Phase A views on GPUs 0-4 (A0-A4). GPUs 5-7 idle / reserved for rebuild.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r11_setup

r11_log "Building Phase A view datasets"
python training/scope_round11/build_phase_a_datasets.py \
  >> "${LOG_DIR}/phase_a_build_datasets.log" 2>&1

r11_log "Launch Phase A views on GPUs 0-4"
PIDS=()
for i in 0 1 2 3 4; do
  VIEW="${PHASE_A_VIEWS[$i]}"
  log="${LOG_DIR}/phase_a_gpu${i}_${VIEW}_supervisor.log"
  (
    bash "$(dirname "$0")/run_phase_a_view.sh" "${i}" "${VIEW}"
  ) >> "${log}" 2>&1 &
  PIDS+=($!)
  echo $! > "${PID_DIR}/phase_a_gpu${i}.pid"
  r11_log "started Phase A GPU${i} view=${VIEW} pid=${!}"
  sleep 3
done

fail=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    r11_log "ERROR: Phase A GPU${i} exited non-zero"
    fail=1
  fi
done

python training/scope_round11/aggregate_phase_a.py \
  >> "${LOG_DIR}/phase_a_aggregate.log" 2>&1 || fail=1

if [[ "${fail}" -ne 0 ]]; then
  r11_log "Phase A finished with failures"
  exit 1
fi
r11_log "Phase A complete"
cat "${OUT}/phase_a_state_factorization/PHASE_A_DECISION.json"
