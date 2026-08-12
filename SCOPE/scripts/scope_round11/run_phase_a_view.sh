#!/usr/bin/env bash
# Score one Phase-A view on offline_valid + base_live with canonical vLLM.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r11_setup

GPU="${1:?gpu}"
VIEW="${2:?view A0-A4}"
PORT="$(r11_port_for_gpu "${GPU}")"
ROOT="${OUT}/phase_a_state_factorization"
STALE_SEC="${R11_STALE_SEC:-3600}"

mkdir -p "${ROOT}/${VIEW}" "${ROOT}/offline_valid/${VIEW}" "${ROOT}/base_live/${VIEW}"
MARKER="${ROOT}/${VIEW}/DONE"
if [[ -f "${MARKER}" ]]; then
  r11_log "Skip Phase A ${VIEW} (DONE)"
  exit 0
fi

heartbeat() { date -Is > "${ROOT}/${VIEW}/HEARTBEAT"; }
heartbeat
HB_PID=""
start_hb() {
  ( while true; do heartbeat; sleep 60; done ) &
  HB_PID=$!
}
stop_hb() {
  if [[ -n "${HB_PID}" ]] && kill -0 "${HB_PID}" 2>/dev/null; then
    kill "${HB_PID}" 2>/dev/null || true
  fi
  HB_PID=""
}
trap 'stop_hb' EXIT

run_split() {
  local split="$1"
  local inp="${VIEWS_DIR}/${split}/${VIEW}.jsonl"
  local outp="${ROOT}/${split}/${VIEW}/canonical_vllm_replay.jsonl"
  local n_expected
  n_expected=$(wc -l < "${inp}" | tr -d ' ')
  if [[ ! -f "${inp}" ]]; then
    r11_log "ERROR: missing view input ${inp}"
    exit 2
  fi
  if [[ -f "${outp}" ]] && [[ "$(wc -l < "${outp}" | tr -d ' ')" -ge "${n_expected}" ]]; then
    r11_log "Phase A ${VIEW}/${split} already complete"
    return 0
  fi
  rm -f "${outp}"
  r11_log "Phase A ${VIEW} canonical-vLLM ${split} gpu=${GPU} port=${PORT}"
  start_hb
  SCOPE_VLLM_OUT_ROOT="${OUT}" CUDA_VISIBLE_DEVICES="${GPU}" \
    python training/scope_round9/run_vllm_replay_split.py \
    --model-path "${PHASE_A_MODEL}" --input "${inp}" --output "${outp}" \
    --port "${PORT}" --gpu "${GPU}" \
    >> "${LOG_DIR}/phase_a_${VIEW}_${split}.log" 2>&1
  stop_hb
  heartbeat
}

run_split offline_valid
run_split base_live
touch "${MARKER}"
r11_log "Phase A ${VIEW} DONE"
