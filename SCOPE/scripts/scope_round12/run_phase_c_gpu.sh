#!/usr/bin/env bash
# Phase C: train/eval one full_stage1 seed OR ckpt_canonical_listwise seed.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r12_setup

GPU="${1:?gpu}"
VARIANT="${2:?variant}"
PORT="$(r12_port_for_gpu "${GPU}")"
VDIR="${OUT}/phase_c/${VARIANT}"
MARKER="${VDIR}/DONE"
VIEW=A0

if [[ -f "${MARKER}" ]]; then
  r12_log "Skip Phase C ${VARIANT} (DONE)"
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
}
trap 'stop_hb; r12_stop_recorded "vllm_port_${PORT}" || true' EXIT

# Reuse Round11 seed42 full_stage1 / listwise when applicable
if [[ "${VARIANT}" == "full_stage1_seed42" && -f "${R11_OUT}/phase_b/factorized_full_stage1_seed42/merged/config.json" ]]; then
  r12_log "Reuse R11 full_stage1_seed42 merged"
  ln -sfn "${R11_OUT}/phase_b/factorized_full_stage1_seed42/merged" "${VDIR}/merged"
  cp -f "${R11_OUT}/phase_b/factorized_full_stage1_seed42/train_only_report.json" "${VDIR}/" || true
elif [[ "${VARIANT}" == "ckpt_canonical_listwise_seed42" && -f "${R11_OUT}/phase_b/factorized_ckpt_listwise_seed42/merged/config.json" ]]; then
  r12_log "Reuse R11 ckpt_listwise_seed42 merged as canonical listwise seed42"
  ln -sfn "${R11_OUT}/phase_b/factorized_ckpt_listwise_seed42/merged" "${VDIR}/merged"
  cp -f "${R11_OUT}/phase_b/factorized_ckpt_listwise_seed42/train_only_report.json" "${VDIR}/" || true
else
  r12_log "${VARIANT} train on GPU${GPU}"
  start_hb
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round12/run_phase_c_train.py \
    --variant "${VARIANT}" --gpu="cuda:0" --out-dir "${VDIR}" \
    >> "${LOG_DIR}/phase_c_${VARIANT}_train.log" 2>&1
  stop_hb
fi

MODEL="${VDIR}/merged"
if [[ ! -f "${MODEL}/config.json" ]]; then
  r12_log "ERROR missing merged ${MODEL}"
  exit 1
fi

if [[ ! -f "${R11_DATA}/factorized_eval/${VIEW}/base_live.jsonl" ]]; then
  python training/scope_round11/build_factorized_eval.py --view "${VIEW}" \
    >> "${LOG_DIR}/build_factorized_eval_${VIEW}.log" 2>&1
fi

USE_S2=""
if [[ "${VARIANT}" == ckpt_canonical_listwise_* ]]; then
  USE_S2="--use-stage2-ranker"
fi

run_eval() {
  local split="$1" tag="$2"
  local inp="${R11_DATA}/factorized_eval/${VIEW}/${split}.jsonl"
  local outp="${VDIR}/eval_${tag}/canonical_vllm_replay.jsonl"
  local n_expected
  n_expected=$(wc -l < "${inp}" | tr -d ' ')
  if [[ -f "${outp}" ]] && [[ "$(wc -l < "${outp}" | tr -d ' ')" -ge "${n_expected}" ]]; then
    return 0
  fi
  rm -f "${outp}"
  start_hb
  SCOPE_VLLM_OUT_ROOT="${OUT}" \
    python training/scope_round11/run_vllm_factorized_split.py \
    --model-path "${MODEL}" --input "${inp}" --output "${outp}" \
    --port "${PORT}" --gpu "${GPU}" ${USE_S2} \
    >> "${LOG_DIR}/phase_c_${VARIANT}_${tag}.log" 2>&1
  stop_hb
}

run_eval offline_valid offline_valid
run_eval base_live holdout

python training/scope_round11/score_variant.py \
  --variant-dir "${VDIR}" --variant "${VARIANT}" \
  --output "${VDIR}/TRAIN_AND_EVAL_REPORT.json" \
  >> "${LOG_DIR}/phase_c_${VARIANT}_score.log" 2>&1

# offline-only threshold calibration artifact
python training/scope_round12/calibrate_seed_threshold.py \
  --variant-dir "${VDIR}" \
  >> "${LOG_DIR}/phase_c_${VARIANT}_tau.log" 2>&1 || true

touch "${MARKER}"
r12_log "Phase C ${VARIANT} DONE"
