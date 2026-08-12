#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

for gpu in 0 1 2 3 4 5 6 7; do
  variant="${WAVE_B_VARIANTS[$gpu]}"
  port="$(scope9_port_for_gpu "${gpu}")"
  model="$(scope9_merged_model "${variant}")"
  for shard in 0 1 2 3; do
    out="${OUT}/wave_c/final100/${variant}/shard${shard}"
    marker="${MARKER_DIR}/final100_${variant}_shard${shard}.DONE"
    if [[ -f "${marker}" ]]; then continue; fi
    mkdir -p "${out}"
    CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round8/rollback_closed_loop_rollout.py \
      --output-dir "${out}" \
      --manifest "${MANIFEST_100}" \
      --shard "shard${shard}" --n-shards 4 \
      --model-path "${model}" \
      --variant "${variant}" \
      --vllm-port "${port}" \
      --parallel "${PARALLEL}" \
      --resume \
      >> "${LOG_DIR}/final100_${variant}_shard${shard}.log" 2>&1
    touch "${marker}"
  done &
done
wait
python training/scope_round9/aggregate_final_gate.py --mode final100
scope9_log "Wave C final100 complete"
