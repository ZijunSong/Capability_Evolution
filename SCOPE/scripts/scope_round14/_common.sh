#!/usr/bin/env bash
# Round14 shared helpers — outputs/scope_round14
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round14"
LOG_DIR="${OUT}/logs"
MARKER_DIR="${OUT}/markers"
PID_DIR="${OUT}/pids"
DATA_DIR="${REPO_ROOT}/artifacts/datasets/scope_round14"
MANIFEST_DIR="${DATA_DIR}/manifests"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
HARNESS_MINIMAL="${REPO_ROOT}/harness/configs/modules_minimal_v2.yaml"
HARNESS_FULL="${REPO_ROOT}/harness/configs/modules_full_v2.yaml"
R14_FRESH100="${MANIFEST_DIR}/R14_FRESH100.json"
R14_SMOKE20="${MANIFEST_DIR}/R14_SMOKE20.json"
O7_SEEDS=(42 43 44)
O7_CKPT_ROOT="${REPO_ROOT}/outputs/scope_round5/merged"

r14_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${MARKER_DIR}" "${PID_DIR}" \
    "${OUT}/gpu0_dup_anchor" \
    "${OUT}/gpu1_stop" \
    "${OUT}/gpu2_verify_routing" \
    "${OUT}/gpu3_evidence_admission" \
    "${OUT}/gpu4_context_budget" \
    "${OUT}/gpu5_external_verify" \
    "${OUT}/gpu6_rollback_lite" \
    "${OUT}/gpu7_method_ablation" \
    "${DATA_DIR}/manifests" \
    "${DATA_DIR}/rollback_lite"
}

r14_log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/r14_supervisor.log"
}

r14_port_for_gpu() {
  echo $((19400 + $1))
}

r14_o7_ckpt() {
  local seed="$1"
  echo "${O7_CKPT_ROOT}/o7_r64_seed${seed}"
}

r14_touch_hb() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
  date -Is > "${path}"
}

r14_gate_pass() {
  local gate_json="$1"
  local key="${2:-gate_a_pass}"
  python -c "import json;print(json.load(open('${gate_json}')).get('${key}', False))"
}

r14_stop_recorded() {
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

r14_kill_vllm_on_gpu() {
  local gpu="$1"
  local port
  port="$(r14_port_for_gpu "${gpu}")"
  pkill -f "run_gpu${gpu}_queue.sh" 2>/dev/null || true
  pkill -f "hmin_v2_dup_rollout.py.*--vllm-port ${port}" 2>/dev/null || true
  pkill -f "vllm.*${port}" 2>/dev/null || true
  pkill -f "CUDA_VISIBLE_DEVICES=${gpu}.*vllm" 2>/dev/null || true
  sleep 2
}

r14_wave0_complete() {
  local anchor="${OUT}/gpu0_dup_anchor"
  [[ -f "${anchor}/B_OFF/DONE" ]] \
    && [[ -f "${anchor}/B_ON/DONE" ]] \
    && [[ -f "${anchor}/T_OFF_seed42/DONE" ]] \
    && [[ -f "${anchor}/T_OFF_seed43/DONE" ]] \
    && [[ -f "${anchor}/T_OFF_seed44/DONE" ]]
}
