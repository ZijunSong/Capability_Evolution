#!/usr/bin/env bash
# nohup launcher for full BrowseComp bare rollout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/bare_rollout_browsecomp_full}"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export OUTPUT_DIR
export SPLIT="${SPLIT:-all}"
export LIMIT="${LIMIT:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export VLLM_PORT="${VLLM_PORT:-8770}"
export PARALLEL="${PARALLEL:-8}"
export RESUME="${RESUME:-0}"

nohup bash "${REPO_ROOT}/scripts/rollout_bare_browsecomp_4gpu.sh" \
  > "${OUTPUT_DIR}/nohup_rollout.log" 2>&1 &

echo $! > "${OUTPUT_DIR}/nohup_rollout.pid"
echo "Started bare rollout PID=$(cat "${OUTPUT_DIR}/nohup_rollout.pid")"
echo "Config: max_new_tokens=${MAX_NEW_TOKENS} temperature=${TEMPERATURE} max_model_len=${MAX_MODEL_LEN} parallel=${PARALLEL}"
echo "Log: ${OUTPUT_DIR}/nohup_rollout.log"
