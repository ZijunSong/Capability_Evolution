#!/usr/bin/env bash
# Round 10 shared helpers
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round10"
R9_OUT="${REPO_ROOT}/outputs/scope_round9"
LOG_DIR="${OUT}/logs"
MARKER_DIR="${OUT}/markers"
PID_DIR="${OUT}/pids"
DATA_DIR="${REPO_ROOT}/artifacts/datasets/scope_round10"
R9_DATA="${REPO_ROOT}/artifacts/datasets/scope_round9"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
OFFLINE_VALID="${R9_DATA}/frozen_replay/offline_valid.jsonl"
LIVE_VALID="${DATA_DIR}/live_split/live_valid.jsonl"
LIVE_TEST="${DATA_DIR}/live_split/live_test.jsonl"

TRAINING_VARIANTS=(
  rollback_live_aligned_seed42
  rollback_live_aligned_seed43
  rollback_live_aligned_seed44
  rollback_live_only_seed42
  rollback_offline_only_binary_seed42
  rollback_hard_continue_seed42
  rollback_source_token_seed42
  rollback_calibration_only
)

scope10_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" "${OUT}/preflight" \
    "${OUT}/training" "${OUT}/calibration" "${OUT}/eval" "${DATA_DIR}"
}

scope10_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/round10_supervisor.log"
}

scope10_port_for_gpu() {
  echo $((18200 + $1))
}

scope10_stop_recorded() {
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

scope10_merged_model() {
  local variant="$1"
  if [[ "${variant}" == "rollback_calibration_only" ]]; then
    echo "${REPO_ROOT}/outputs/scope_round8/merged/rollback_o7_seed42"
  elif [[ -f "${OUT}/training/${variant}/merged/config.json" ]]; then
    echo "${OUT}/training/${variant}/merged"
  else
    echo "${BASE_MODEL}"
  fi
}
