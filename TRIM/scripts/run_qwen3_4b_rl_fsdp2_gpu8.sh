#!/usr/bin/env bash
# Single-node 8-GPU TRIM RL: per-rank vLLM TP=1 rollout + FSDP2 CISPO.
#
# This is NOT Scheme A (device_map=auto). run_train.py sees
# --training-backend verl and torchruns nproc=8. Each rank pins one GPU,
# rolls out, closes vLLM, then runs one FSDP2 optimizer.step.
#
# CUDA_VISIBLE_DEVICES must list eight ids (0,1,2,3,4,5,6,7). Do not pass
# 0-8; that is nine cards. Do not put a comment after a line-continuation
# backslash — it breaks --max-turns 40.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash TRIM/scripts/run_qwen3_4b_rl_fsdp2_gpu8.sh
#
# Optional env: PY, MODEL_NAME, OUT, TRAIN_STEPS, TRAIN_METHOD, COMPONENT,
# TRAIN_GROUPS_PER_STEP, GROUP_SIZE, MAX_TURNS, MAX_MODEL_LEN, GPU_UTIL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRIM_GPU_KEEPALIVE=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PY:-python}"
MODEL_NAME="${MODEL_NAME:-../../models/Qwen3-4B-Instruct-2507}"
OUT="${OUT:-outputs/qwen3_4b_rl_fsdp2_$(date +%Y%m%d_%H%M%S)}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_METHOD="${TRAIN_METHOD:-rl}"
COMPONENT="${COMPONENT:-all}"
SCAPE="${TRIM_ROOT}/../SCAPE-EasyOPD"

PYTHONPATH="${PYTHONPATH:-}:${TRIM_ROOT}:${SCAPE}" \
"${PY}" scripts/run_train.py \
  --training-backend verl \
  --nproc-per-node 8 \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name "${MODEL_NAME}" \
  --train_method "${TRAIN_METHOD}" \
  --component "${COMPONENT}" \
  --train-env local_legacy \
  --train-data sec \
  --train-steps "${TRAIN_STEPS}" \
  --train-groups-per-step "${TRAIN_GROUPS_PER_STEP:-32}" \
  --group-size "${GROUP_SIZE:-8}" \
  --train-micro-batch-size "${TRAIN_MICRO_BATCH_SIZE:-4}" \
  --max-turns "${MAX_TURNS:-40}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --out "${OUT}"
