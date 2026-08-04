#!/usr/bin/env bash
# GPU7: sentinel + parity + tests + report
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

GPU=7
PORT=9207

scope7_log "GPU7 queue start (sentinel + tests)"

# Run unit tests first
pytest -q tests/scope/ >> "${LOG_DIR}/gpu7_tests.log" 2>&1 || true

# Sentinel runs on small query subset
for THRESH in inf neginf zero; do
  OUT_S="${OUT}/sentinel/threshold_${THRESH}"
  mkdir -p "${OUT_S}"
  TAU=0
  [[ "${THRESH}" == "inf" ]] && TAU=999999
  [[ "${THRESH}" == "neginf" ]] && TAU=-999999
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round7/sentinel_run.py \
    --model-path "${R5}/merged/o7_r64_seed42" \
    --output-dir "${OUT_S}" \
    --threshold "${TAU}" \
    --vllm-port "${PORT}" \
    --n-queries 5 \
    >> "${LOG_DIR}/sentinel_${THRESH}.log" 2>&1 || true
done

# Parity audit across seeds
for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round6/run_parity_audit.py \
    --mode adapter_merged --scorer "o7_${SEED}" --gpu cuda:0 \
    --output-dir "${OUT}/sentinel/parity_seed${SEED}" \
    >> "${LOG_DIR}/parity_seed${SEED}.log" 2>&1 || true
done

# Wait for main queues then build report
for i in $(seq 1 1440); do
  n=$(ls "${MARKER_DIR}"/*.json 2>/dev/null | wc -l)
  if [[ "${n}" -ge 4 ]]; then break; fi
  sleep 120
done

python training/scope_round7/build_round7_report.py >> "${LOG_DIR}/gpu7_report.log" 2>&1 || true

scope7_write_marker "gpu7_sentinel" "complete" "${OUT}/sentinel" 0
scope7_log "GPU7 queue complete"
