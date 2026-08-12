#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

OUT830="${OUT}/gpu0_dup_anchor/confirm_830_seed42/B_OFF"
MANIFEST="${MANIFEST_DIR}/R14_HOLD_830.json"
HARNESS_OFF="${OUT}/gpu0_dup_anchor/B_OFF/harness_module_off.yaml"

run_one() {
  local gpu="$1" shard="$2"
  local out="${OUT830}/${shard}"
  local expected=104
  [[ "${shard}" == "shard6" || "${shard}" == "shard7" ]] && expected=103
  local n=0
  [[ -f "${out}/episodes.jsonl" ]] && n=$(wc -l < "${out}/episodes.jsonl" | tr -d ' ')
  if [[ "${n}" -ge "${expected}" && -f "${out}/summary.json" ]]; then
    r14_log "skip ${shard} already complete (${n})"
    return 0
  fi
  mkdir -p "${out}"
  local port
  port="$(r14_port_for_gpu "${gpu}")"
  r14_log "fill B_OFF ${shard} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" \
    --n-shards 8 \
    --model-path "${BASE_MODEL}" \
    --harness-config "${HARNESS_OFF}" \
    --temperature 0.0 \
    --vllm-port "${port}" \
    --dup-seed 42 \
    --checkpoint-label B_OFF \
    --parallel "${R14_PARALLEL:-16}" \
    --decision-threshold 0 \
    --resume \
    --collect-states-only \
    >> "${LOG_DIR}/dup830_B_OFF_${shard}.log" 2>&1 &
  echo $! > "${PID_DIR}/dup830_b_off_${shard}.pid"
}

run_one 0 shard5
sleep 5
run_one 1 shard6
r14_log "fill_b_off_shards56 launched"
