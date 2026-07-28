#!/usr/bin/env bash
# Phase 0 Minimal Runtime BrowseComp+ rollout (reuses rollout_harness_browsecomp_4gpu.sh).
# Only changes Harness module YAML + V8D env flags; same model/BM25/driver/max_turns.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/minimal_runtime_browsecomp_full830}"
HARNESS_CONFIG="${HARNESS_CONFIG:-$REPO_ROOT/harness/configs/modules_minimal.yaml}"
SCOPE_CONFIG="${SCOPE_CONFIG:-$REPO_ROOT/configs/scope/minimal_runtime.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OUTPUT_DIR
export HARNESS_CONFIG
export SPLIT="${SPLIT:-all}"
export LIMIT="${LIMIT:-0}"
export MAX_TURNS="${MAX_TURNS:-35}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export VLLM_PORT="${VLLM_PORT:-8772}"
export PARALLEL="${PARALLEL:-2}"
export RESUME="${RESUME:-0}"
export RERANKER="${RERANKER:-none}"
export RETRIEVAL="${RETRIEVAL:-bm25}"
export USE_LEGACY_API_AGENT="${USE_LEGACY_API_AGENT:-0}"

# Critical: ultra_core reads V8D_* at import time — must be off BEFORE python starts.
# Do NOT rely on apply_harness_config alone for Minimal Runtime.
export V8D_SUBTRACTIVE_CURATION=0
export V8D_IMPORTANCE_TAGGING=0
export V8D_AUTO_POPULATE_FIRST_SEARCH=0
export V8D_EVIDENCE_GRAPH=0
export V8D_SENTENCE_COMPRESS=0
export V8D_CONTENT_DEDUP=0
export V8D_VERIFY_TOOL=0
export V8D_TOKEN_BUDGET_MARKER=0
export V8D_CHUNK_NEIGHBORS=0
export ABLATE_VERIFY_UNAVAILABLE=1
export ABLATE_REVIEW_DOCS_UNAVAILABLE=1

# Match Full Harness v2 ChatDecisionDriver entry (module ablation only).
export CHAT_MIN_TURNS_BEFORE_END="${CHAT_MIN_TURNS_BEFORE_END:-8}"
export CHAT_MIN_CURATED_BEFORE_END="${CHAT_MIN_CURATED_BEFORE_END:-1}"
export CHAT_MAX_WM_CHARS="${CHAT_MAX_WM_CHARS:-18000}"
export CHAT_MAX_RECENT_TURNS="${CHAT_MAX_RECENT_TURNS:-4}"
export SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-1}"
export SAVE_FULL_TRAJECTORIES="${SAVE_FULL_TRAJECTORIES:-0}"

mkdir -p "${OUTPUT_DIR}"
cp -f "${SCOPE_CONFIG}" "${OUTPUT_DIR}/scope_minimal_runtime.yaml"

echo "[minimal] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[minimal] HARNESS_CONFIG=${HARNESS_CONFIG}"
echo "[minimal] V8D flags forced OFF for import-time ultra_core"
echo "[minimal] LIMIT=${LIMIT} RESUME=${RESUME} PORT=${VLLM_PORT}"

bash "${REPO_ROOT}/scripts/rollout_harness_browsecomp_4gpu.sh"

# Normalize Phase-0 artifact names from harness rollout outputs.
python "${REPO_ROOT}/scripts/finalize_minimal_runtime_artifacts.py" \
  --output-dir "${OUTPUT_DIR}" \
  --harness-config "${HARNESS_CONFIG}" \
  --scope-config "${SCOPE_CONFIG}"
