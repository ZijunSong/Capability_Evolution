#!/usr/bin/env bash
# Round 10 followup (0808) shared helpers — outputs/scope_round10_followup
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round10_followup"
R10_OUT="${REPO_ROOT}/outputs/scope_round10"
R9_OUT="${REPO_ROOT}/outputs/scope_round9"
LOG_DIR="${OUT}/logs"
MARKER_DIR="${OUT}/markers"
PID_DIR="${OUT}/pids"
DATA_DIR="${REPO_ROOT}/artifacts/datasets/scope_round10"
R9_DATA="${REPO_ROOT}/artifacts/datasets/scope_round9"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
OFFLINE_VALID="${DATA_DIR}/frozen_replay/offline_valid.jsonl"
BASE_LIVE="${DATA_DIR}/frozen_replay/base_live.jsonl"
P0_ROOT="${R9_OUT}/wave_b_p0"

# GPU map matches 0808-todo1.md §3
PHASE_B_VARIANTS=(
  r10_main_noweight_seed42          # GPU0
  r10_main_noweight_seed43          # GPU1
  r10_main_noweight_seed44          # GPU2
  r10_p0_exact_repro_seed42         # GPU3
  r10_natural_prior_noweight_seed42 # GPU4
  r10_balanced50_noweight_seed42    # GPU5
  r10_stage1_state_only_seed42      # GPU6
  r10_threshold_only_p0_seed42      # GPU7 (+ aggregate)
)

followup_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" \
    "${OUT}/phase_a/calibration" "${OUT}/phase_b" "${OUT}/eval" \
    "${OUT}/phase_c_smoke20" "${OUT}/phase_d_final100"
}

followup_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/followup_supervisor.log"
}

followup_port_for_gpu() {
  echo $((18400 + $1))
}

followup_stop_recorded() {
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

followup_p0_merged() {
  local seed="$1"
  echo "${P0_ROOT}/rollback_hier_o7_seed${seed}/merged"
}
