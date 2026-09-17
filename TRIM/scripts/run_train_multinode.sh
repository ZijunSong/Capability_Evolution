#!/usr/bin/env bash
# Multi-node TRIM RL. Run the SAME command on every node with a different NODE_RANK.
#
# 4 × 8-GPU nodes (32 cards), Qwen-4B: one vLLM actor per GPU, rank-0 HF train.
# --out must be a shared filesystem. --max-turns 40 is the training horizon
# (the CLI default is 6; this script always passes 40).
set -euo pipefail

NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0..$((NNODES-1)) on each node}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to node 0 hostname or IP}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MODEL_NAME="${MODEL_NAME:-/mnt/songzijun/models/Qwen3-4B-Instruct-2507}"
OUT="${OUT:-outputs/rl_multinode_$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHONPATH="${PYTHONPATH:-}:TRIM:SCAPE-EasyOPD" \
python TRIM/scripts/run_train_multinode.py \
  --nnodes "${NNODES}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --rollout-replicas "${ROLLOUT_REPLICAS}" \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name "${MODEL_NAME}" \
  --train_method trim \
  --component all \
  --train-env local_legacy \
  --train-data sec \
  --train-steps "${TRAIN_STEPS:-16}" \
  --train-groups-per-step "${TRAIN_GROUPS_PER_STEP:-32}" \
  --group-size 8 \
  --train-micro-batch-size 4 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --out "${OUT}"
