#!/usr/bin/env bash
# Run a named Phase B variant on an arbitrary physical GPU.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r11_setup

GPU="${1:?physical gpu}"
VARIANT="${2:?variant name}"
PORT="$(r11_port_for_gpu "${GPU}")"
VDIR="${OUT}/phase_b/${VARIANT}"
MARKER="${VDIR}/DONE"

if [[ -f "${MARKER}" ]]; then
  r11_log "Skip Phase B ${VARIANT} (DONE)"
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
trap 'stop_hb' EXIT

r11_log "${VARIANT} train on physical GPU${GPU}"
start_hb
if [[ ! -f "${VDIR}/merged/config.json" ]] || [[ ! -f "${VDIR}/train_only_report.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round11/run_phase_b_train.py \
    --variant "${VARIANT}" --gpu="cuda:0" --out-root "${OUT}/phase_b" \
    >> "${LOG_DIR}/phase_b_${VARIANT}_train.log" 2>&1
fi
stop_hb
heartbeat

MODEL="${VDIR}/merged"
if [[ ! -f "${MODEL}/config.json" ]]; then
  r11_log "ERROR: missing merged model ${MODEL}"
  exit 1
fi

VVIEW=$(python -c "import json; print(json.load(open('${VDIR}/train_only_report.json')).get('stage1_view','A1'))")
if [[ ! -f "${DATA_DIR}/factorized_eval/${VVIEW}/base_live.jsonl" ]]; then
  python training/scope_round11/build_factorized_eval.py --view "${VVIEW}" \
    >> "${LOG_DIR}/build_factorized_eval_${VVIEW}.log" 2>&1
fi

run_eval_split() {
  local split="$1" tag="$2" inp="$3"
  local outp="${VDIR}/eval_${tag}/canonical_vllm_replay.jsonl"
  local n_expected
  n_expected=$(wc -l < "${inp}" | tr -d ' ')
  if [[ -f "${outp}" ]] && [[ "$(wc -l < "${outp}" | tr -d ' ')" -ge "${n_expected}" ]]; then
    r11_log "${VARIANT} ${tag} already complete"
    return 0
  fi
  rm -f "${outp}"
  r11_log "${VARIANT} factorized-vLLM ${tag} view=${VVIEW} gpu=${GPU}"
  start_hb
  SCOPE_VLLM_OUT_ROOT="${OUT}" CUDA_VISIBLE_DEVICES="${GPU}" \
    python training/scope_round11/run_vllm_factorized_split.py \
    --model-path "${MODEL}" --input "${inp}" --output "${outp}" \
    --port "${PORT}" --gpu "${GPU}" --use-stage2-ranker \
    >> "${LOG_DIR}/phase_b_${VARIANT}_${tag}_eval.log" 2>&1
  stop_hb
  heartbeat
}

run_eval_split offline_valid offline_valid "${DATA_DIR}/factorized_eval/${VVIEW}/offline_valid.jsonl"
run_eval_split base_live holdout "${DATA_DIR}/factorized_eval/${VVIEW}/base_live.jsonl"

python training/scope_round11/score_variant.py \
  --variant-dir "${VDIR}" --variant "${VARIANT}" \
  --output "${VDIR}/TRAIN_AND_EVAL_REPORT.json" \
  >> "${LOG_DIR}/phase_b_${VARIANT}_score.log" 2>&1

r11_stop_recorded "vllm_port_${PORT}" || true
heartbeat
touch "${MARKER}"
r11_log "Phase B ${VARIANT} DONE"
