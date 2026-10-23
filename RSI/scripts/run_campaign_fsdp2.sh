#!/usr/bin/env bash
# Multi-GPU training stage launcher (torch_fsdp2 backend).
# Requires: torchrun process group; the campaign config must select
# training.backend=torch_fsdp2.  Rollout and training workers may occupy GPUs
# in phases via immutable PolicyArtifacts (see docs/runbook.md).
set -u

CONFIG="${1:-configs/pilot_multidomain.yaml}"
NNODES="${NNODES:-1}"
NPROC="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29622}"

cd "$(dirname "$0")/.."

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m rsi.cli train --config "$CONFIG" --bank "${2:?champion bank ref required}"
