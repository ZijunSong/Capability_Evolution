#!/usr/bin/env bash
# Train + offline eval + HF/vLLM parity for one Wave B variant on one GPU.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

GPU="${1:?gpu}"
VARIANT="${WAVE_B_VARIANTS[$GPU]}"
PORT="$(scope9_port_for_gpu "${GPU}")"
VDIR="${OUT}/wave_b/${VARIANT}"
MARKER="${VDIR}/DONE"
VALID_FROZEN="${DATA_DIR}/frozen_replay/offline_valid.jsonl"
HOLDOUT_REPLAY_IN="${DATA_DIR}/frozen_replay/base_live.jsonl"

if [[ -f "${MARKER}" ]]; then
  scope9_log "Skip Wave B ${VARIANT}"
  exit 0
fi

mkdir -p "${VDIR}/eval_offline_valid" "${VDIR}/eval_holdout"

scope9_log "${VARIANT} train"
CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/run_wave_b_train.py \
  --variant "${VARIANT}" --gpu cuda:0 \
  >> "${LOG_DIR}/wave_b_${VARIANT}.log" 2>&1

MODEL="$(scope9_merged_model "${VARIANT}")"
if [[ ! -f "${MODEL}/config.json" ]]; then
  scope9_log "ERROR: missing merged model for ${VARIANT}: ${MODEL}"
  exit 1
fi

run_eval_split() {
  local split="$1" inp="$2"
  local hf_out="${VDIR}/eval_${split}/hf_replay.jsonl"
  local vllm_out="${VDIR}/eval_${split}/vllm_replay.jsonl"
  local n_expected
  n_expected=$(wc -l < "${inp}" | tr -d ' ')
  mkdir -p "${VDIR}/eval_${split}"
  if [[ ! -f "${hf_out}" ]] || [[ "$(wc -l < "${hf_out}" | tr -d ' ')" -lt "${n_expected}" ]]; then
    rm -f "${hf_out}"
    scope9_log "${VARIANT} HF eval ${split}"
    CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/replay_frozen_hf.py \
      --model-path "${MODEL}" --input "${inp}" --output "${hf_out}" --device cuda:0 \
      >> "${LOG_DIR}/wave_b_${VARIANT}_${split}_hf.log" 2>&1
  fi
  if [[ ! -f "${vllm_out}" ]] || [[ "$(wc -l < "${vllm_out}" | tr -d ' ')" -lt "${n_expected}" ]]; then
    rm -f "${vllm_out}"
    scope9_log "${VARIANT} vLLM eval ${split}"
    CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/run_vllm_replay_split.py \
      --model-path "${MODEL}" --input "${inp}" --output "${vllm_out}" --port "${PORT}" --gpu "${GPU}" \
      >> "${LOG_DIR}/wave_b_${VARIANT}_${split}_vllm.log" 2>&1
  fi
}

run_eval_split offline_valid "${VALID_FROZEN}"
run_eval_split holdout "${HOLDOUT_REPLAY_IN}"

python training/scope_round9/aggregate_wave_b_report.py \
  --variant-dir "${VDIR}" --variant "${VARIANT}" \
  --output "${VDIR}/TRAIN_AND_EVAL_REPORT.json"

scope9_stop_recorded "vllm_port_${PORT}" || true
touch "${MARKER}"
scope9_log "Wave B ${VARIANT} DONE"
