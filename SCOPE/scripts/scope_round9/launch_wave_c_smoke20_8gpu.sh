#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

python training/scope_round9/create_smoke_manifest.py --n 20 --output "${MANIFEST_20}"

for gpu in 0 1 2 3 4 5 6 7; do
  variant="${WAVE_B_VARIANTS[$gpu]}"
  port="$(scope9_port_for_gpu "${gpu}")"
  model="$(scope9_merged_model "${variant}")"
  out="${OUT}/wave_c/smoke20/${variant}"
  mkdir -p "${out}"
  if [[ -f "${MARKER_DIR}/smoke20_${variant}.DONE" ]]; then continue; fi
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round8/rollback_closed_loop_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST_20}" \
    --shard shard0 --n-shards 1 \
    --model-path "${model}" \
    --variant "${variant}" \
    --vllm-port "${port}" \
    --parallel "${PARALLEL}" \
    --resume \
    >> "${LOG_DIR}/smoke20_${variant}.log" 2>&1 &
done
wait
python training/scope_round9/aggregate_final_gate.py --mode smoke20
scope9_log "Wave C smoke20 complete"
