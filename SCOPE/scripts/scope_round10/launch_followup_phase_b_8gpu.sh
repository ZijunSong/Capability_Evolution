#!/usr/bin/env bash
# Launch followup Phase B on 8 GPUs (0808-todo1.md §3).
set -euo pipefail
source "$(dirname "$0")/_common_followup.sh"
followup_setup

GATE="${OUT}/CANONICAL_BACKEND_GATE.json"
if [[ ! -f "${GATE}" ]]; then
  followup_log "ERROR: run Phase A2 first (missing ${GATE})"
  exit 2
fi
PASS=$(python -c "import json; print(json.load(open('${GATE}')).get('pass', False))")
if [[ "${PASS}" != "True" ]]; then
  followup_log "CANONICAL_BACKEND_GATE.pass=false — refuse Phase B"
  exit 3
fi

followup_log "Launch followup Phase B on GPUs 0-7"
PIDS=()
for gpu in 0 1 2 3 4 5 6 7; do
  log="${LOG_DIR}/phase_b_gpu${gpu}_supervisor.log"
  (
    bash "$(dirname "$0")/run_followup_phase_b_gpu.sh" "${gpu}"
  ) >> "${log}" 2>&1 &
  PIDS+=($!)
  echo $! > "${PID_DIR}/phase_b_gpu${gpu}.pid"
  followup_log "started GPU${gpu} pid=${!}"
  sleep 2
done

fail=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    followup_log "ERROR: GPU${i} exited non-zero"
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  followup_log "Phase B finished with failures — see logs"
  exit 1
fi
followup_log "Phase B all GPU queues finished"
python training/scope_round10/aggregate_followup_phase_b_gate.py \
  >> "${LOG_DIR}/phase_b_aggregate_final.log" 2>&1 || true
