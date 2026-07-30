#!/usr/bin/env bash
# Round 6 shared helpers
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round6"
R5="${REPO_ROOT}/outputs/scope_round5"
LOG_DIR="${OUT}/logs"
STAGE_FILE="${OUT}/PIPELINE_STAGE"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"

scope6_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${OUT}/phase_b" "${OUT}/closed_loop" "${OUT}/calibration"
}

scope6_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/round6_supervisor.log"
}

scope6_set_stage() {
  echo "$1" > "${STAGE_FILE}"
  scope6_log "STAGE -> $1"
}

scope6_get_stage() {
  if [[ -f "${STAGE_FILE}" ]]; then
    tr -d '[:space:]' < "${STAGE_FILE}"
  else
    echo "init"
  fi
}

scope6_run_gpu() {
  local gpu="$1" logfile="$2"
  shift 2
  mkdir -p "$(dirname "${logfile}")"
  CUDA_VISIBLE_DEVICES="${gpu}" "$@" >> "${logfile}" 2>&1
}

scope6_snapshot_env() {
  {
    echo "date=$(date -Is)"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch=$(git branch --show-current 2>/dev/null || echo unknown)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
    python --version 2>/dev/null || true
  } > "${OUT}/environment_snapshot.txt"
  git diff HEAD > "${OUT}/git_diff_before_round6.patch" 2>/dev/null || true
}
