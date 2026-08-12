#!/usr/bin/env bash
# One GPU: run model x view frozen replay on offline_valid + base_live.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r12_setup

GPU="${1:?gpu}"
JOB="${2:?job e.g. M0_V0}"
PORT="$(r12_port_for_gpu "${GPU}")"
MODEL="$(r12_model_for_job "${JOB}")"
VIEW="$(r12_view_for_job "${JOB}")"
VDIR="${OUT}/phase_b_operation_boundary/cross_view_replays/${JOB}"
MARKER="${VDIR}/DONE"

if [[ -z "${MODEL}" || ! -f "${MODEL}/config.json" ]]; then
  r12_log "ERROR: missing model for ${JOB}: ${MODEL}"
  exit 2
fi

if [[ -f "${MARKER}" ]]; then
  r12_log "Skip ${JOB} (DONE)"
  exit 0
fi

mkdir -p "${VDIR}/eval_offline_valid" "${VDIR}/eval_holdout"
heartbeat() { date -Is > "${VDIR}/HEARTBEAT"; }
heartbeat
HB_PID=""
start_hb() { ( while true; do heartbeat; sleep 60; done ) & HB_PID=$!; }
stop_hb() {
  if [[ -n "${HB_PID}" ]] && kill -0 "${HB_PID}" 2>/dev/null; then
    kill "${HB_PID}" 2>/dev/null || true
  fi
  HB_PID=""
}
trap 'stop_hb; r12_stop_recorded "vllm_port_${PORT}" || true' EXIT

IN_ROOT="${R11_DATA}/factorized_eval/${VIEW}"
if [[ ! -f "${IN_ROOT}/base_live.jsonl" ]]; then
  r12_log "Building factorized eval view=${VIEW}"
  python training/scope_round11/build_factorized_eval.py --view "${VIEW}" \
    >> "${LOG_DIR}/build_factorized_eval_${VIEW}.log" 2>&1
fi

run_split() {
  local split="$1"
  local tag="$2"
  local inp="${IN_ROOT}/${split}.jsonl"
  local outp="${VDIR}/eval_${tag}/canonical_vllm_replay.jsonl"
  local n_expected
  n_expected=$(wc -l < "${inp}" | tr -d ' ')
  if [[ -f "${outp}" ]] && [[ "$(wc -l < "${outp}" | tr -d ' ')" -ge "${n_expected}" ]]; then
    r12_log "${JOB} ${tag} already complete (${n_expected})"
    return 0
  fi
  rm -f "${outp}"
  r12_log "${JOB} vLLM replay ${tag} view=${VIEW} gpu=${GPU} port=${PORT}"
  start_hb
  # Pass physical GPU id: run_vllm_factorized_split overwrites CUDA_VISIBLE_DEVICES.
  SCOPE_VLLM_OUT_ROOT="${OUT}" \
    python training/scope_round11/run_vllm_factorized_split.py \
    --model-path "${MODEL}" \
    --input "${inp}" \
    --output "${outp}" \
    --port "${PORT}" \
    --gpu "${GPU}" \
    >> "${LOG_DIR}/cross_${JOB}_${tag}.log" 2>&1
  stop_hb
  heartbeat
}

run_split "offline_valid" "offline_valid"
run_split "base_live" "holdout"

r12_stop_recorded "vllm_port_${PORT}" || true
heartbeat
touch "${MARKER}"
r12_log "${JOB} DONE"
