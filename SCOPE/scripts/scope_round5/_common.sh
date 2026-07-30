#!/usr/bin/env bash
# Round 5 pipeline stage helpers
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
OUT="${REPO_ROOT}/outputs/scope_round5"
LOG_DIR="${OUT}/logs"
STAGE_FILE="${OUT}/PIPELINE_STAGE"
SUPERVISOR_PID="${LOG_DIR}/pipeline_supervisor.pid"
SUPERVISOR_LOG="${LOG_DIR}/pipeline_supervisor.log"

scope5_setup() {
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  export PYTHONPATH="${REPO_ROOT}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}" "${OUT}/micro_overfit" "${OUT}/b4_full" "${OUT}/closed_loop"
}

scope5_log() {
  echo "[$(date -Is)] $*" >> "${SUPERVISOR_LOG}"
}

scope5_get_stage() {
  if [[ -f "${STAGE_FILE}" ]]; then
    tr -d '[:space:]' < "${STAGE_FILE}"
    return
  fi
  scope5_detect_stage
}

scope5_set_stage() {
  echo "$1" > "${STAGE_FILE}"
  scope5_log "STAGE -> $1"
}

scope5_detect_stage() {
  if [[ -f "${OUT}/ROUND5_COMPLETE" ]]; then echo "done"; return; fi
  if [[ -f "${OUT}/B6_COMPLETE" ]]; then echo "done"; return; fi
  if [[ -f "${OUT}/B5_COMPLETE" ]]; then echo "b6"; return; fi
  if [[ -f "${OUT}/B4_PASS" ]] && [[ "$(tr -d '[:space:]' < "${OUT}/B4_PASS")" == "True" ]]; then echo "b5"; return; fi
  if [[ -f "${OUT}/b4_full/B4_COMPLETE" ]]; then echo "b4_gate"; return; fi
  if pgrep -f "run_b4_train.py" >/dev/null 2>&1; then echo "b4_train"; return; fi
  local ready=0
  for tag in o7_r64_seed42 o7_r64_seed43 o7_r64_seed44 compact_json_seed42 compact_json_seed43 compact_json_seed44; do
    [[ -f "${OUT}/b4_full/${tag}/DONE" ]] && ready=$((ready + 1))
  done
  if [[ "${ready}" -eq 6 ]]; then echo "b4_eval"; return; fi
  if [[ "${ready}" -gt 0 ]]; then echo "b4_train"; return; fi
  if [[ -f "${OUT}/micro_overfit/MICRO_OVERFIT_MATRIX.md" ]]; then echo "b4_train"; return; fi
  if [[ -f "${OUT}/B2_PASS" ]] && [[ "$(tr -d '[:space:]' < "${OUT}/B2_PASS")" == "True" ]]; then echo "b3"; return; fi
  if [[ -f "${OUT}/B1_PASS" ]] && [[ "$(tr -d '[:space:]' < "${OUT}/B1_PASS")" == "True" ]]; then echo "b2"; return; fi
  if [[ -f "${OUT}/environment_snapshot.txt" ]]; then echo "b1"; return; fi
  echo "b0"
}

scope5_gate_true() {
  local f="$1"
  [[ -f "${f}" ]] && [[ "$(tr -d '[:space:]' < "${f}")" == "True" ]]
}

scope5_b4_train_count() {
  pgrep -f "run_b4_train.py" 2>/dev/null | wc -l | tr -d ' '
}

scope5_b4_done_count() {
  local n=0 tag
  for tag in o7_r64_seed42 o7_r64_seed43 o7_r64_seed44 compact_json_seed42 compact_json_seed43 compact_json_seed44; do
    [[ -f "${OUT}/b4_full/${tag}/DONE" ]] && n=$((n + 1))
  done
  echo "${n}"
}

scope5_job_dir() {
  local phase="$1" job="$2"
  local d="${LOG_DIR}/${phase}/${job}"
  mkdir -p "${d}"
  echo "${d}"
}

scope5_mark_done() {
  local phase="$1" job="$2"
  date -Is > "$(scope5_job_dir "${phase}" "${job}")/done"
}

scope5_is_done() {
  [[ -f "$(scope5_job_dir "$1" "$2")/done" ]]
}

scope5_launch_bg() {
  local phase="$1" job="$2" logfile="$3"
  shift 3
  local d pidfile
  d="$(scope5_job_dir "${phase}" "${job}")"
  pidfile="${d}/pid"
  if scope5_is_done "${phase}" "${job}"; then return 0; fi
  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then return 0; fi
  mkdir -p "$(dirname "${logfile}")"
  nohup bash -c "$*" >> "${logfile}" 2>&1 &
  echo $! > "${pidfile}"
  scope5_log "launch ${phase}/${job} pid=$(cat "${pidfile}")"
}
