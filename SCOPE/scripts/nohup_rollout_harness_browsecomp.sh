#!/usr/bin/env bash
# nohup launcher for full BrowseComp Harness rollout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/harness_rollout_browsecomp_full}"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OUTPUT_DIR
export SPLIT="${SPLIT:-all}"
export LIMIT="${LIMIT:-0}"
export MAX_TURNS="${MAX_TURNS:-35}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export VLLM_PORT="${VLLM_PORT:-8771}"
export PARALLEL="${PARALLEL:-2}"
export RESUME="${RESUME:-1}"
export RERANKER="${RERANKER:-none}"
export RETRIEVAL="${RETRIEVAL:-bm25}"

nohup bash "${REPO_ROOT}/scripts/rollout_harness_browsecomp_4gpu.sh" \
  > "${OUTPUT_DIR}/nohup_rollout.log" 2>&1 &

echo $! > "${OUTPUT_DIR}/nohup_rollout.pid"
echo "Started harness rollout PID=$(cat "${OUTPUT_DIR}/nohup_rollout.pid")"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}  max_turns=${MAX_TURNS} max_tokens=${MAX_TOKENS} parallel=${PARALLEL}"
echo "Log: ${OUTPUT_DIR}/nohup_rollout.log"
