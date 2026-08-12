#!/usr/bin/env bash
# Run Wave A replay for one GPU/variant
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

GPU="${1:?gpu}"
VARIANT="${WAVE_A_VARIANTS[$GPU]}"
PORT="$(scope9_port_for_gpu "${GPU}")"
VDIR="${OUT}/wave_a/${VARIANT}"
MODEL="$(scope9_merged_model "${VARIANT}")"
MARKER="${MARKER_DIR}/wave_a_${VARIANT}.DONE"
VLLM_PID_FILE="${PID_DIR}/vllm_${VARIANT}.pid"

if [[ -f "${MARKER}" ]]; then
  scope9_log "Skip Wave A ${VARIANT}"
  exit 0
fi

mkdir -p "${VDIR}/offline_valid" "${VDIR}/base_live" "${VDIR}/self_live"

run_split() {
  local split="$1" inp="$2"
  local hf_out="${VDIR}/${split}/hf_replay.jsonl"
  local vllm_out="${VDIR}/${split}/vllm_replay.jsonl"
  local n_expected
  n_expected=$(wc -l < "${inp}" | tr -d ' ')
  if [[ ! -f "${hf_out}" ]] || [[ "$(wc -l < "${hf_out}" | tr -d ' ')" -lt "${n_expected}" ]]; then
    rm -f "${hf_out}"
    scope9_log "${VARIANT} HF ${split}"
    CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/replay_frozen_hf.py \
      --model-path "${MODEL}" --input "${inp}" --output "${hf_out}" --device cuda:0 \
      >> "${LOG_DIR}/wave_a_${VARIANT}_${split}.log" 2>&1
  fi
  if [[ ! -f "${vllm_out}" ]] || [[ "$(wc -l < "${vllm_out}" | tr -d ' ')" -lt "${n_expected}" ]]; then
    rm -f "${vllm_out}"
    scope9_log "${VARIANT} vLLM ${split}"
    if [[ -f "${VLLM_PID_FILE}" ]]; then
      scope9_stop_recorded "vllm_${VARIANT}"
    fi
    CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/run_vllm_replay_split.py \
      --model-path "${MODEL}" --input "${inp}" --output "${vllm_out}" --port "${PORT}" --gpu "${GPU}" \
      >> "${LOG_DIR}/wave_a_${VARIANT}_${split}_vllm.log" 2>&1
  fi
}

verify_outputs() {
  for split in offline_valid base_live self_live; do
    local inp="${FROZEN}/${split}.jsonl"
    [[ "${split}" == "self_live" ]] && inp="${FROZEN}/self_live/${VARIANT}.jsonl"
    local n_exp hf_out vllm_out n_hf n_vl
    n_exp=$(wc -l < "${inp}" | tr -d ' ')
    hf_out="${VDIR}/${split}/hf_replay.jsonl"
    vllm_out="${VDIR}/${split}/vllm_replay.jsonl"
    n_hf=$(wc -l < "${hf_out}" | tr -d ' ')
    n_vl=$(wc -l < "${vllm_out}" | tr -d ' ')
    if [[ "${n_hf}" -lt "${n_exp}" ]] || [[ "${n_vl}" -lt "${n_exp}" ]]; then
      scope9_log "ERROR: ${VARIANT} ${split} incomplete hf=${n_hf} vllm=${n_vl} expected=${n_exp}"
      return 1
    fi
  done
}

run_split offline_valid "${FROZEN}/offline_valid.jsonl"
run_split base_live "${FROZEN}/base_live.jsonl"
run_split self_live "${FROZEN}/self_live/${VARIANT}.jsonl"

verify_outputs

# Re-apply shared decide() on stored logits so HF/vLLM use identical tie-break.
python training/scope_round9/redecide_replay_logits.py --variant-dir "${VDIR}" \
  >> "${LOG_DIR}/wave_a_${VARIANT}_redecide.log" 2>&1

if ! python training/scope_round9/aggregate_frozen_replay.py \
  --variant-dir "${VDIR}" --output "${VDIR}/WAVE_A_REPORT.json"; then
  scope9_log "ERROR: Wave A Barrier A FAILED for ${VARIANT}"
  exit 2
fi

# Release any recorded vLLM PID for this variant before DONE.
scope9_stop_recorded "vllm_${VARIANT}" || true
scope9_stop_recorded "vllm_port_${PORT}" || true

touch "${MARKER}"
scope9_log "Wave A ${VARIANT} DONE"
