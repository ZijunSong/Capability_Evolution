#!/usr/bin/env bash
# Resume rollback_correct_only Phase 3 shards (GPU6)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

VARIANT="rollback_correct_only"
GPU=6
MERGED="${OUT}/merged/${VARIANT}"
PHASE3_OUT="${OUT}/phase3_closed_loop/${VARIANT}"
LOG="${LOG_DIR}/phase3"
PARALLEL_PHASE3="${PARALLEL_PHASE3:-8}"

scope8_kill_vllm_ports() {
  local ports=("$@")
  for port in "${ports[@]}"; do
    if fuser -n tcp "${port}" >/dev/null 2>&1; then
      scope8_log "Kill stale vLLM on port ${port}"
      fuser -k -n tcp "${port}" >/dev/null 2>&1 || true
      sleep 3
    fi
  done
}

run_shard() {
  local shard="$1" port="$2"
  local out="${PHASE3_OUT}/${shard}"
  local logfile="${LOG}/${VARIANT}_${shard}.log"
  local n
  n=$(scope8_count_episodes "${out}/episodes.jsonl")
  if [[ "${n}" -ge 25 ]] && [[ -f "${out}/summary.json" ]]; then
    scope8_log "Skip ${VARIANT} ${shard} (${n}/25)"
    return 0
  fi
  scope8_log "Resume ${VARIANT} GPU${GPU} ${shard} (${n}/25)"
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round8/rollback_closed_loop_rollout.py \
    --output-dir "${out}" \
    --variant "${VARIANT}" \
    --manifest "${MANIFEST_100}" \
    --shard "${shard}" --n-shards 4 \
    --model-path "${BASE_MODEL}" \
    --merged-path "${MERGED}" \
    --harness-config "${REPO_ROOT}/harness/configs/agent_core_recovery.yaml" \
    --vllm-port "${port}" \
    --parallel "${PARALLEL_PHASE3}" \
    --rollback-operation \
    --resume \
    >> "${logfile}" 2>&1
}

scope8_log "Resuming ${VARIANT} shards on GPU${GPU}"
scope8_kill_vllm_ports 9461 9462 9463
run_shard shard2 9462
run_shard shard3 9463

scope8_log "Re-aggregating Phase 3 gate"
python training/scope_round8/aggregate_phase3_gate.py
scope8_log "${VARIANT} resume complete"
