#!/usr/bin/env bash
# Round 13 shared helpers — outputs/scope_round13
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round13"
R11_OUT="${REPO_ROOT}/outputs/scope_round11"
LOG_DIR="${OUT}/logs"
MARKER_DIR="${OUT}/markers"
PID_DIR="${OUT}/pids"
DATA_DIR="${REPO_ROOT}/artifacts/datasets/scope_round13"
MANIFEST_DIR="${DATA_DIR}/manifests"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
COLLECT_MODEL="${R11_OUT}/phase_b/factorized_full_stage1_seed42/merged"

STAGE1_VARIANTS=(
  "r13_onpolicy_querynorm_seed42"
  "r13_onpolicy_querynorm_seed43"
  "r13_onpolicy_querynorm_seed44"
  "r13_onpolicy_querynorm_nohard_seed42"
  "r13_onpolicy_eventuniform_seed42"
)

r13_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" \
    "${OUT}/phase_a_shift" \
    "${OUT}/phase_b_stage1/training" \
    "${OUT}/phase_b_stage1/valid" \
    "${OUT}/phase_b_stage1/test" \
    "${OUT}/stage2_audit" \
    "${OUT}/stage2_targeted" \
    "${OUT}/smoke20" \
    "${OUT}/final100" \
    "${DATA_DIR}/manifests" \
    "${DATA_DIR}/onpolicy_raw/train" \
    "${DATA_DIR}/onpolicy_raw/valid" \
    "${DATA_DIR}/operation_sdi" \
    "${DATA_DIR}/checkpoint_targeted"
}

r13_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/r13_supervisor.log"
}

r13_port_for_gpu() {
  echo $((18700 + $1))
}

r13_stop_recorded() {
  local name="$1"
  local f="${PID_DIR}/${name}.pid"
  if [[ -f "${f}" ]]; then
    local pid
    pid="$(cat "${f}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      sleep 2
    fi
    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${f}"
  fi
}

r13_touch_hb() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
  date -Is > "${path}"
}
