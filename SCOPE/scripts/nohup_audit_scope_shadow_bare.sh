#!/usr/bin/env bash
# nohup launcher for SCOPE Bare Shadow Audit on GPUs 4-7.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/scope_shadow_audit_bare}"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export OUTPUT_DIR
export MODE="${MODE:-bare-replay}"
export LIMIT="${LIMIT:-0}"
export MAX_TURNS="${MAX_TURNS:-35}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export VLLM_PORT="${VLLM_PORT:-8772}"
export PARALLEL="${PARALLEL:-2}"
export RESUME="${RESUME:-1}"
export RERANKER="${RERANKER:-none}"
export RETRIEVAL="${RETRIEVAL:-bm25}"
export BARE_JSONL="${BARE_JSONL:-/data/ppnm/BiSHOP/outputs/bare_rollout_browsecomp_full/bare_rollouts.jsonl}"

nohup bash "${REPO_ROOT}/scripts/audit_scope_shadow_bare_4gpu.sh" \
  > "${OUTPUT_DIR}/nohup_audit.log" 2>&1 &

echo $! > "${OUTPUT_DIR}/nohup_audit.pid"
echo "Started shadow audit PID=$(cat "${OUTPUT_DIR}/nohup_audit.pid")"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}  mode=${MODE}  port=${VLLM_PORT}"
echo "Log: ${OUTPUT_DIR}/nohup_audit.log"
