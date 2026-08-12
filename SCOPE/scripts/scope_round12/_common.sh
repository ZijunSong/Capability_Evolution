#!/usr/bin/env bash
# Round 12 shared helpers — outputs/scope_round12
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round12"
R9_OUT="${REPO_ROOT}/outputs/scope_round9"
R10_FOLLOWUP="${REPO_ROOT}/outputs/scope_round10_followup"
R11_OUT="${REPO_ROOT}/outputs/scope_round11"
LOG_DIR="${OUT}/logs"
MARKER_DIR="${OUT}/markers"
PID_DIR="${OUT}/pids"
DATA_DIR="${REPO_ROOT}/artifacts/datasets/scope_round12"
R10_DATA="${REPO_ROOT}/artifacts/datasets/scope_round10"
R11_DATA="${REPO_ROOT}/artifacts/datasets/scope_round11"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
OFFLINE_VALID="${R10_DATA}/frozen_replay/offline_valid.jsonl"
BASE_LIVE="${R10_DATA}/frozen_replay/base_live.jsonl"
HIER_TRAIN="${R10_DATA}/hier_sdi/train_p0_75.jsonl"
HIER_VALID="${R10_DATA}/hier_sdi/valid.jsonl"

# Models
M0="${R11_OUT}/phase_b/factorized_full_stage1_seed42/merged"
M1="${R11_OUT}/phase_b/factorized_main_seed42/merged"
M2="${R10_FOLLOWUP}/phase_b/r10_main_noweight_seed42/merged"
C11L="${R11_OUT}/phase_b/factorized_ckpt_listwise_seed42/merged"
C11P="${R11_OUT}/phase_b/factorized_ckpt_pairwise_seed42/merged"

# Parallel job table for Barrier A/B (GPU → job name)
# GPU0-5: cross-view M×V ; GPU6: C11L oracle replay ; GPU7: C11P oracle replay
CROSS_VIEW_JOBS=(
  "M0_V0"
  "M0_V1"
  "M1_V0"
  "M1_V1"
  "M2_V0"
  "M2_V1"
)

r12_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" \
    "${OUT}/preflight" \
    "${OUT}/phase_a_ckpt_provenance/per_selector_scores" \
    "${OUT}/phase_b_operation_boundary" \
    "${OUT}/phase_c" \
    "${OUT}/phase_d_smoke20" \
    "${OUT}/phase_e_final100" \
    "${DATA_DIR}"
}

r12_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/r12_supervisor.log"
}

r12_port_for_gpu() {
  echo $((18600 + $1))
}

r12_stop_recorded() {
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

r12_model_for_job() {
  case "$1" in
    M0_*) echo "${M0}" ;;
    M1_*) echo "${M1}" ;;
    M2_*) echo "${M2}" ;;
    C11L) echo "${C11L}" ;;
    C11P) echo "${C11P}" ;;
    *) echo "" ;;
  esac
}

r12_view_for_job() {
  case "$1" in
    *_V0) echo "A0" ;;
    *_V1) echo "A1" ;;
    C11L|C11P) echo "A0" ;;  # stage2 uses stage2_text; A0 has candidates
    *) echo "A0" ;;
  esac
}
