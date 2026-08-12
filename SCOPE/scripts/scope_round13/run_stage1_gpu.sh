#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

GPU="${1:?gpu}"
VARIANT="${2:?variant}"
OUT_DIR="${OUT}/phase_b_stage1/training/${VARIANT}"
mkdir -p "${OUT_DIR}"
r13_touch_hb "${OUT_DIR}/HEARTBEAT"

# Already fully done (train + eval metrics)
if [[ -f "${OUT_DIR}/DONE" ]] && [[ -f "${OUT_DIR}/merged/config.json" ]] \
  && [[ -f "${OUT_DIR}/eval_valid/METRICS.json" ]]; then
  r13_log "Skip stage1 ${VARIANT} (DONE+metrics)"
  exit 0
fi

r13_log "Stage1 train GPU${GPU} ${VARIANT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

HB_PID=""
cleanup() {
  if [[ -n "${HB_PID}" ]]; then
    kill "${HB_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

(
  while [[ ! -f "${OUT_DIR}/DONE" ]] && [[ ! -f "${OUT_DIR}/FAILED" ]]; do
    date -Is > "${OUT_DIR}/HEARTBEAT"
    sleep 60
  done
) &
HB_PID=$!

rm -f "${OUT_DIR}/FAILED"

set +e
python training/scope_round13/run_stage1_train.py \
  --variant "${VARIANT}" \
  --gpu cuda:0 \
  --out-root "${OUT}/phase_b_stage1/training" \
  >> "${LOG_DIR}/stage1_${VARIANT}.log" 2>&1
TRAIN_RC=$?
set -e

if [[ "${TRAIN_RC}" -ne 0 ]]; then
  r13_log "Stage1 TRAIN FAILED rc=${TRAIN_RC} ${VARIANT}"
  echo "TRAIN_FAILED rc=${TRAIN_RC} $(date -Is)" > "${OUT_DIR}/FAILED"
  exit "${TRAIN_RC}"
fi

# VALID replay with vLLM
PORT="$(r13_port_for_gpu "${GPU}")"
set +e
python training/scope_round13/eval_stage1_split.py \
  --variant-dir "${OUT_DIR}" \
  --split valid \
  --gpu "${GPU}" \
  --port "${PORT}" \
  >> "${LOG_DIR}/stage1_${VARIANT}_eval.log" 2>&1
EVAL_RC=$?
set -e

if [[ "${EVAL_RC}" -ne 0 ]]; then
  r13_log "Stage1 EVAL FAILED rc=${EVAL_RC} ${VARIANT}"
  echo "EVAL_FAILED rc=${EVAL_RC} $(date -Is)" > "${OUT_DIR}/FAILED"
  # Keep merged/train artifacts; allow retry of eval by clearing FAILED externally.
  exit "${EVAL_RC}"
fi

date -Is > "${OUT_DIR}/DONE"
rm -f "${OUT_DIR}/FAILED"
r13_log "Stage1 DONE ${VARIANT}"
