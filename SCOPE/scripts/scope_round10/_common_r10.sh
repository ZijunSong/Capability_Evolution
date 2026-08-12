#!/usr/bin/env bash
# Round 10 (0807) shared helpers — rollback live parity + CONTINUE boundary
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
OFFLINE_VALID="${DATA_DIR}/frozen_replay/offline_valid.jsonl"
BASE_LIVE="${DATA_DIR}/frozen_replay/base_live.jsonl"
P0_ROOT="${R9_OUT}/wave_b_p0"

PHASE_B_VARIANTS=(
  r10_main_noweight_seed42
  r10_main_noweight_seed43
  r10_main_noweight_seed44
  r10_natural_prior_noweight_seed42
  r10_balanced50_noweight_seed42
  r10_p0_exact_repro_seed42
  r10_threshold_only_p0_seed42
  r10_stage1_state_only_seed42
)

scope10_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" \
    "${OUT}/preflight" "${OUT}/phase_a" "${OUT}/phase_b" "${OUT}/eval" "${DATA_DIR}"
}

scope10_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/round10_supervisor.log"
}

scope10_port_for_gpu() {
  echo $((18300 + $1))
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
    fi
    kill -9 "${pid}" 2>/dev/null || true
    rm -f "${f}"
  fi
}

scope10_p0_merged() {
  local seed="$1"
  echo "${P0_ROOT}/rollback_hier_o7_seed${seed}/merged"
}
