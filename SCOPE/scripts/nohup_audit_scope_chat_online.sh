#!/usr/bin/env bash
# nohup launcher: chat-online DecisionState audit on GPUs 4-7.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/scope_chat_decision_audit}"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export OUTPUT_DIR
export LIMIT="${LIMIT:-40}"
export SEED="${SEED:-42}"
export MAX_TURNS="${MAX_TURNS:-35}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export VLLM_PORT="${VLLM_PORT:-8773}"
export PARALLEL="${PARALLEL:-2}"
export RESUME="${RESUME:-1}"
export RERANKER="${RERANKER:-none}"
export RETRIEVAL="${RETRIEVAL:-bm25}"

nohup bash "${REPO_ROOT}/scripts/audit_scope_chat_online_4gpu.sh" \
  > "${OUTPUT_DIR}/nohup_chat_audit.log" 2>&1 &

echo $! > "${OUTPUT_DIR}/nohup_chat_audit.pid"
echo "Started chat DecisionState audit PID=$(cat "${OUTPUT_DIR}/nohup_chat_audit.pid")"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}  limit=${LIMIT}  port=${VLLM_PORT}"
echo "Log: ${OUTPUT_DIR}/nohup_chat_audit.log"
