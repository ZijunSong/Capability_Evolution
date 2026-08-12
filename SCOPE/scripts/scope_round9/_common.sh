#!/usr/bin/env bash
# Round 9 shared helpers
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round9"
R8_OUT="${REPO_ROOT}/outputs/scope_round8"
LOG_DIR="${OUT}/logs"
MARKER_DIR="${OUT}/markers"
PID_DIR="${OUT}/pids"
DATA_DIR="${REPO_ROOT}/artifacts/datasets/scope_round9"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
MANIFEST_100="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
MANIFEST_20="${OUT}/manifests/smoke20.json"
FROZEN="${DATA_DIR}/frozen_replay"
PARALLEL="${PARALLEL:-16}"

WAVE_A_VARIANTS=(
  base_agent_core
  rollback_o7_seed42
  rollback_o7_seed43
  rollback_o7_seed44
  rollback_prompt_hint_distill
  rollback_trajectory_imitation
  rollback_correct_only
  rollback_soft_replan_only
)

WAVE_B_VARIANTS=(
  rollback_hier_o7_seed42
  rollback_hier_o7_seed43
  rollback_hier_o7_seed44
  rollback_flat_o7_seed42_repro
  rollback_operation_only_seed42
  rollback_checkpoint_ranker_seed42
  rollback_hier_no_candidate_summary_seed42
  rollback_hier_prompt_hint_seed42
)

scope9_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" "${OUT}/preflight" \
    "${OUT}/wave_a" "${OUT}/wave_b" "${OUT}/wave_c" "${OUT}/manifests" "${DATA_DIR}"
}

scope9_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/round9_supervisor.log"
}

scope9_port_for_gpu() {
  echo $((18100 + $1))
}

scope9_record_pid() {
  local name="$1" pid="$2"
  echo "${pid}" > "${PID_DIR}/${name}.pid"
}

scope9_stop_recorded() {
  local name="$1"
  local f="${PID_DIR}/${name}.pid"
  if [[ -f "${f}" ]]; then
    local pid
    pid="$(cat "${f}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      sleep 2
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${f}"
  fi
}

scope9_merged_model() {
  local variant="$1"
  if [[ "${variant}" == "base_agent_core" ]]; then
    echo "${BASE_MODEL}"
  elif [[ -f "${OUT}/wave_b/${variant}/merged/config.json" ]]; then
    echo "${OUT}/wave_b/${variant}/merged"
  elif [[ -f "${R8_OUT}/merged/${variant}/config.json" ]]; then
    echo "${R8_OUT}/merged/${variant}"
  else
    echo "${BASE_MODEL}"
  fi
}

scope9_count_jsonl() {
  local f="$1"
  if [[ -f "${f}" ]]; then wc -l < "${f}" | tr -d ' '; else echo 0; fi
}
