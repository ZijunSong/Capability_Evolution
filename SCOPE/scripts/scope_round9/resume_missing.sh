#!/usr/bin/env bash
# Resume missing Wave C shards only
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

for gpu in 0 1 2 3 4 5 6 7; do
  variant="${WAVE_B_VARIANTS[$gpu]}"
  for shard in 0 1 2 3; do
    out="${OUT}/wave_c/final100/${variant}/shard${shard}"
    ep="${out}/episodes.jsonl"
    n=$(scope9_count_jsonl "${ep}")
    if [[ "${n}" -lt 25 ]]; then
      scope9_log "Resume ${variant} shard${shard} (${n}/25)"
      port="$(scope9_port_for_gpu "${gpu}")"
      model="$(scope9_merged_model "${variant}")"
      CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round8/rollback_closed_loop_rollout.py \
        --output-dir "${out}" \
        --manifest "${MANIFEST_100}" \
        --shard "shard${shard}" --n-shards 4 \
        --model-path "${model}" \
        --variant "${variant}" \
        --vllm-port "${port}" \
        --parallel "${PARALLEL}" \
        --resume \
        >> "${LOG_DIR}/resume_${variant}_shard${shard}.log" 2>&1 || true
    fi
  done
done
