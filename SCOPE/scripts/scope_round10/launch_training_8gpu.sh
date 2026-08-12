#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope10_setup

scope10_log "Launch Round 10 training on 8 GPUs"
for gpu in 0 1 2 3 4 5 6 7; do
  bash "$(dirname "$0")/run_gpu.sh" "${gpu}" \
    >> "${LOG_DIR}/training_supervisor.log" 2>&1 &
  sleep 3
done
wait

done_n=0
for v in "${TRAINING_VARIANTS[@]}"; do
  [[ -f "${OUT}/training/${v}/DONE" ]] && done_n=$((done_n + 1))
done
if [[ "${done_n}" -lt 8 ]]; then
  scope10_log "ERROR: training incomplete (${done_n}/8)"
  exit 1
fi

python training/scope_round10/check_offline_gate.py \
  --output "${OUT}/ROUND10_OFFLINE_GATE.json" \
  >> "${LOG_DIR}/offline_gate.log" 2>&1 || true

scope10_log "Training phase complete"
