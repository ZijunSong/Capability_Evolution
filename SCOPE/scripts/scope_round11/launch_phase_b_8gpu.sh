#!/usr/bin/env bash
# Launch Round11 Phase B on 8 GPUs.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r11_setup

DECISION="${OUT}/phase_a_state_factorization/PHASE_A_DECISION.json"
if [[ ! -f "${DECISION}" ]]; then
  r11_log "ERROR: run Phase A first"
  exit 2
fi

r11_log "Launch Phase B on GPUs 0-7"
PIDS=()
for gpu in 0 1 2 3 4 5 6 7; do
  log="${LOG_DIR}/phase_b_gpu${gpu}_supervisor.log"
  (
    bash "$(dirname "$0")/run_phase_b_gpu.sh" "${gpu}"
  ) >> "${log}" 2>&1 &
  PIDS+=($!)
  echo $! > "${PID_DIR}/phase_b_gpu${gpu}.pid"
  r11_log "started Phase B GPU${gpu} variant=${PHASE_B_VARIANTS[$gpu]} pid=${!}"
  sleep 3
done

fail=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    r11_log "ERROR: Phase B GPU${i} exited non-zero"
    fail=1
  fi
done

python training/scope_round11/aggregate_phase_b_gate.py \
  >> "${LOG_DIR}/phase_b_aggregate.log" 2>&1 || true

if [[ "${fail}" -ne 0 ]]; then
  r11_log "Phase B finished with failures — see logs"
  exit 1
fi
r11_log "Phase B all GPU queues finished"
cat "${OUT}/FROZEN_LIVE_GATE.json" | head -c 2000 || true
