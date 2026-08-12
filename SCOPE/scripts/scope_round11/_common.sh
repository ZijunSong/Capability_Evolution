#!/usr/bin/env bash
# Round 11 shared helpers — outputs/scope_round11
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round11"
R10_FOLLOWUP="${REPO_ROOT}/outputs/scope_round10_followup"
LOG_DIR="${OUT}/logs"
MARKER_DIR="${OUT}/markers"
PID_DIR="${OUT}/pids"
DATA_DIR="${REPO_ROOT}/artifacts/datasets/scope_round11"
R10_DATA="${REPO_ROOT}/artifacts/datasets/scope_round10"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
PHASE_A_MODEL="${R10_FOLLOWUP}/phase_b/r10_main_noweight_seed42/merged"
VIEWS_DIR="${DATA_DIR}/phase_a_views"
OFFLINE_VALID="${R10_DATA}/frozen_replay/offline_valid.jsonl"
BASE_LIVE="${R10_DATA}/frozen_replay/base_live.jsonl"
HIER_TRAIN="${R10_DATA}/hier_sdi/train_p0_75.jsonl"
HIER_VALID="${R10_DATA}/hier_sdi/valid.jsonl"

PHASE_A_VIEWS=(A0 A1 A2 A3 A4)
# GPU0-4: Phase A views; GPU5-7 spare / offline shards if needed
PHASE_B_VARIANTS=(
  factorized_main_seed42
  factorized_main_seed43
  factorized_main_seed44
  factorized_state_only_seed42
  factorized_full_stage1_seed42
  factorized_compact_signal_seed42
  factorized_ckpt_listwise_seed42
  factorized_ckpt_pairwise_seed42
)

r11_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" \
    "${OUT}/phase_a_state_factorization" "${OUT}/phase_b" "${OUT}/eval" \
    "${OUT}/phase_c_smoke20" "${OUT}/phase_d_final100" "${DATA_DIR}"
}

r11_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/r11_supervisor.log"
}

r11_port_for_gpu() {
  echo $((18500 + $1))
}

r11_stop_recorded() {
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
