#!/usr/bin/env bash
# Run one on-policy collection shard on a single GPU.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

GPU="${1:?gpu}"
SPLIT="${2:?train|valid|test}"
SHARD="${3:?shardN}"
N_SHARDS="${4:?n_shards}"

if [[ "${SPLIT}" == "train" ]]; then
  MANIFEST="${MANIFEST_DIR}/R13_TRAIN200.json"
  OUT_DIR="${DATA_DIR}/onpolicy_raw/train/${SHARD}"
elif [[ "${SPLIT}" == "valid" ]]; then
  MANIFEST="${MANIFEST_DIR}/R13_VALID100.json"
  OUT_DIR="${DATA_DIR}/onpolicy_raw/valid/${SHARD}"
else
  MANIFEST="${MANIFEST_DIR}/R13_TEST100.json"
  OUT_DIR="${DATA_DIR}/onpolicy_raw/test/${SHARD}"
fi

PORT="$(r13_port_for_gpu "${GPU}")"
mkdir -p "${OUT_DIR}"
r13_touch_hb "${OUT_DIR}/HEARTBEAT"

if [[ -f "${OUT_DIR}/DONE" ]] && [[ -f "${OUT_DIR}/rollback_events.jsonl" ]]; then
  r13_log "Skip collect GPU${GPU} ${SPLIT}/${SHARD} (DONE)"
  exit 0
fi

r13_log "Collect GPU${GPU} ${SPLIT}/${SHARD} port=${PORT} -> ${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
python training/scope_round13/collect_onpolicy.py \
  --output-dir "${OUT_DIR}" \
  --manifest "${MANIFEST}" \
  --shard "${SHARD}" \
  --n-shards "${N_SHARDS}" \
  --model-path "${COLLECT_MODEL}" \
  --vllm-port "${PORT}" \
  --parallel "${PARALLEL:-16}" \
  --split-name "${SPLIT}" \
  --resume \
  >> "${LOG_DIR}/collect_${SPLIT}_${SHARD}_gpu${GPU}.log" 2>&1

r13_log "Collect DONE GPU${GPU} ${SPLIT}/${SHARD}"
