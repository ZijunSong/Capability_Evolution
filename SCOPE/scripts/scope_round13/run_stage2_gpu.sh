#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

GPU="${1:?gpu}"
VARIANT="${2:?variant}"
OUT_DIR="${OUT}/stage2_targeted/training/${VARIANT}"
mkdir -p "${OUT_DIR}"
r13_touch_hb "${OUT_DIR}/HEARTBEAT"

if [[ -f "${OUT_DIR}/DONE" ]] && [[ -f "${OUT_DIR}/merged/config.json" ]]; then
  r13_log "Skip stage2 ${VARIANT} (DONE)"
  exit 0
fi

GATE="${OUT}/stage2_targeted/DATASET_GATE.json"
if [[ ! -f "${GATE}" ]]; then
  r13_log "Stage2 gate missing; building natural dataset"
  python training/scope_round13/build_natural_stage2.py >> "${LOG_DIR}/build_natural_stage2.log" 2>&1
fi
pass=$(python -c "import json;print(json.load(open('${GATE}')).get('NONDEGENERATE_STAGE2_DATA_PASS', False))")
if [[ "${pass}" != "True" ]]; then
  r13_log "STOP Stage2: NONDEGENERATE_STAGE2_DATA_PASS=false"
  echo "STAGE2_DATA_GATE_FAIL" > "${OUT}/STOP_REASON.txt"
  exit 2
fi

r13_log "Stage2 train GPU${GPU} ${VARIANT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

HB_PID=""
cleanup() { [[ -n "${HB_PID}" ]] && kill "${HB_PID}" 2>/dev/null || true; }
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
python training/scope_round13/run_stage2_pointer_train.py \
  --variant "${VARIANT}" \
  --gpu cuda:0 \
  --out-root "${OUT}/stage2_targeted/training" \
  >> "${LOG_DIR}/stage2_${VARIANT}.log" 2>&1
RC=$?
set -e
if [[ "${RC}" -ne 0 ]]; then
  echo "TRAIN_FAILED rc=${RC} $(date -Is)" > "${OUT_DIR}/FAILED"
  r13_log "Stage2 FAILED ${VARIANT} rc=${RC}"
  exit "${RC}"
fi
date -Is > "${OUT_DIR}/DONE"
rm -f "${OUT_DIR}/FAILED"
r13_log "Stage2 DONE ${VARIANT}"
