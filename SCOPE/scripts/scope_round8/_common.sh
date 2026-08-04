#!/usr/bin/env bash
# Round 8 shared helpers
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round8"
R5="${REPO_ROOT}/outputs/scope_round5"
LOG_DIR="${OUT}/logs"
MARKER_DIR="${OUT}/markers"
PID_DIR="${OUT}/pids"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
MANIFEST_100="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
MANIFEST_830="${REPO_ROOT}/artifacts/datasets/scope_round8/query_manifest_830.json"
PARALLEL="${PARALLEL:-48}"
EXEC_LOG="${OUT}/GPU_EXECUTION_LOG.md"

scope8_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" \
    "${OUT}/dup_retention_830" \
    "${OUT}/agent_core_diagnostic" \
    "${OUT}/rollback_collection" \
    "${OUT}/preflight" \
    "${REPO_ROOT}/artifacts/datasets/scope_round8"
  if [[ ! -f "${EXEC_LOG}" ]]; then
    echo "# Round 8 GPU Execution Log" > "${EXEC_LOG}"
  fi
}

scope8_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/round8_supervisor.log"
}

scope8_log_gpu() {
  local gpu="$1" task="$2" cmd="$3" out="$4"
  {
    echo ""
    echo "## GPU${gpu} — ${task}"
    echo "- Start: $(date -Is)"
    echo "- Command: \`${cmd}\`"
    echo "- Output: ${out}"
  } >> "${EXEC_LOG}"
}

scope8_count_episodes() {
  local ep="$1"
  if [[ -f "${ep}" ]]; then
    wc -l < "${ep}" | tr -d ' '
  else
    echo 0
  fi
}

scope8_run_dup_retention() {
  local gpu="$1" shard="$2" model="$3" port="$4" seed="$5" label="$6"
  local out="${OUT}/dup_retention_830/${label}/shard${shard#shard}"
  local expected=207
  if [[ "${shard}" == "shard3" ]]; then expected=209; fi
  local n
  n=$(scope8_count_episodes "${out}/episodes.jsonl")
  if [[ "${n}" -ge "${expected}" ]] && [[ -f "${out}/summary.json" ]]; then
    scope8_log "Skip complete dup ${label} ${shard} (${n}/${expected})"
    return 0
  fi
  mkdir -p "${out}"
  scope8_log "Dup retention GPU${gpu} ${label} ${shard} -> ${out}"
  local cmd="CUDA_VISIBLE_DEVICES=${gpu} python training/scope_round3/hmin_v2_dup_rollout.py ..."
  scope8_log_gpu "${gpu}" "dup_${label}_${shard}" "${cmd}" "${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST_830}" \
    --shard "${shard}" --n-shards 4 \
    --model-path "${model}" \
    --harness-config "${REPO_ROOT}/harness/configs/modules_minimal_v2.yaml" \
    --vllm-port "${port}" \
    --dup-operation \
    --decision-threshold 0 \
    --dup-seed "${seed}" \
    --checkpoint-label "${label}" \
    --parallel "${PARALLEL}" \
    --resume \
    >> "${LOG_DIR}/dup_${label}_${shard}.log" 2>&1
}

scope8_run_agent_core() {
  local gpu="$1" shard="$2" model="$3" port="$4" harness_cfg="$5" label="$6"
  local out="${OUT}/agent_core_diagnostic/${label}/${shard}"
  local n
  n=$(scope8_count_episodes "${out}/episodes.jsonl")
  if [[ "${n}" -ge 25 ]] && [[ -f "${out}/summary.json" ]]; then
    scope8_log "Skip complete agent_core ${label} ${shard}"
    return 0
  fi
  mkdir -p "${out}"
  scope8_log "AgentCore GPU${gpu} ${label} ${shard}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round8/agent_core_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST_100}" \
    --shard "${shard}" --n-shards 4 \
    --model-path "${model}" \
    --harness-config "${harness_cfg}" \
    --label "${label}" \
    --vllm-port "${port}" \
    --parallel "${PARALLEL}" \
    --resume \
    >> "${LOG_DIR}/agent_${label}_${shard}.log" 2>&1
}

scope8_count_rollback_events() {
  local ep="$1"
  if [[ -f "${ep}" ]]; then
    wc -l < "${ep}" | tr -d ' '
  else
    echo 0
  fi
}

scope8_collect_rollback() {
  local gpu="$1" shard="$2" model="$3" port="$4" mode="$5"
  local out="${OUT}/rollback_collection/${mode}/${shard}"
  local events="${out}/rollback_events.jsonl"
  local n
  n=$(scope8_count_rollback_events "${events}")
  # 25q × ~15 turns ≈ 375 events/shard; require minimum coverage before skip
  if [[ "${n}" -ge 100 ]]; then
    scope8_log "Skip complete rollback ${mode} ${shard} (${n} events)"
    return 0
  fi
  if [[ "${n}" -gt 0 ]]; then
    scope8_log "WARN: rollback ${mode} ${shard} partial (${n} events), rerunning"
    rm -f "${events}" "${out}/collection_stats.json"
  fi
  mkdir -p "${out}"
  scope8_log "Rollback collect GPU${gpu} ${mode} ${shard}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round8/collect_rollback_states.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST_100}" \
    --shard "${shard}" --n-shards 4 \
    --mode "${mode}" \
    --model-path "${model}" \
    --vllm-port "${port}" \
    --parallel 16 \
    --resume \
    >> "${LOG_DIR}/rollback_${mode}_${shard}.log" 2>&1
}
