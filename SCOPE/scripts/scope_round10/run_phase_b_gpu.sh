#!/usr/bin/env bash
# Phase B: train+merge+HF/vLLM replay for one GPU queue slot (0807 §8/§12).
set -euo pipefail
source "$(dirname "$0")/_common_r10.sh"
scope10_setup

GPU="${1:?gpu 0-7}"
VARIANT="${PHASE_B_VARIANTS[$GPU]}"
PORT="$(scope10_port_for_gpu "${GPU}")"
VDIR="${OUT}/phase_b/${VARIANT}"
MARKER="${VDIR}/DONE"
GATE_A="${OUT}/PARITY_GATE.json"

if [[ ! -f "${GATE_A}" ]]; then
  scope10_log "ERROR: missing PARITY_GATE.json — refuse Phase B"
  exit 2
fi
PASS=$(python -c "import json; print(json.load(open('${GATE_A}')).get('pass', False))")
if [[ "${PASS}" != "True" ]]; then
  scope10_log "PARITY_GATE.pass=false — STOP_AFTER_PHASE_A; skip ${VARIANT}"
  exit 3
fi

if [[ -f "${MARKER}" ]]; then
  scope10_log "Skip Phase B ${VARIANT} (DONE)"
  exit 0
fi

mkdir -p "${VDIR}/eval_offline_valid" "${VDIR}/eval_holdout" "${VDIR}/reports"

run_eval_split() {
  local split="$1" inp="$2"
  local hf_out="${VDIR}/eval_${split}/hf_replay.jsonl"
  local vllm_out="${VDIR}/eval_${split}/vllm_replay.jsonl"
  local n_expected
  n_expected=$(wc -l < "${inp}" | tr -d ' ')
  if [[ ! -f "${hf_out}" ]] || [[ "$(wc -l < "${hf_out}" | tr -d ' ')" -lt "${n_expected}" ]]; then
    rm -f "${hf_out}"
    scope10_log "${VARIANT} HF ${split}"
    CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/replay_frozen_hf.py \
      --model-path "${MODEL}" --input "${inp}" --output "${hf_out}" --device cuda:0 --dtype float32 \
      >> "${LOG_DIR}/phase_b_${VARIANT}_${split}_hf.log" 2>&1
  fi
  if [[ ! -f "${vllm_out}" ]] || [[ "$(wc -l < "${vllm_out}" | tr -d ' ')" -lt "${n_expected}" ]]; then
    rm -f "${vllm_out}"
    scope10_log "${VARIANT} vLLM ${split}"
    CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/run_vllm_replay_split.py \
      --model-path "${MODEL}" --input "${inp}" --output "${vllm_out}" --port "${PORT}" --gpu "${GPU}" \
      >> "${LOG_DIR}/phase_b_${VARIANT}_${split}_vllm.log" 2>&1
  fi
}

if [[ "${VARIANT}" == "r10_threshold_only_p0_seed42" ]]; then
  scope10_log "${VARIANT}: threshold sweep on P0 seed42 (no train)"
  MODEL="$(scope10_p0_merged 42)"
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round10/calibrate_threshold_p0.py \
    --model-path "${MODEL}" \
    --offline "${OFFLINE_VALID}" \
    --holdout "${BASE_LIVE}" \
    --out-dir "${VDIR}" \
    >> "${LOG_DIR}/phase_b_${VARIANT}.log" 2>&1
  touch "${MARKER}"
  scope10_log "${VARIANT} DONE"
  exit 0
fi

scope10_log "${VARIANT} train"
CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round10/run_phase_b_train.py \
  --variant "${VARIANT}" --gpu cuda:0 --out-root "${OUT}/phase_b" --force-retrain \
  >> "${LOG_DIR}/phase_b_${VARIANT}_train.log" 2>&1

MODEL="${VDIR}/merged"
if [[ ! -f "${MODEL}/config.json" ]]; then
  scope10_log "ERROR: missing merged model ${MODEL}"
  exit 1
fi

run_eval_split offline_valid "${OFFLINE_VALID}"
run_eval_split holdout "${BASE_LIVE}"

python training/scope_round9/aggregate_wave_b_report.py \
  --variant-dir "${VDIR}" --variant "${VARIANT}" \
  --output "${VDIR}/TRAIN_AND_EVAL_REPORT.json"

scope10_stop_recorded "vllm_port_${PORT}" || true
touch "${MARKER}"
scope10_log "Phase B ${VARIANT} DONE"
