#!/usr/bin/env bash
# SCOPE Shadow Audit on Bare trajectories — GPUs 4-7 (TP=4).
# Runs bare-replay (CPU) then online multi-turn audit (vLLM) with M1+M2 only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/scope_shadow_audit_bare}"
BARE_JSONL="${BARE_JSONL:-/data/ppnm/BiSHOP/outputs/bare_rollout_browsecomp_full/bare_rollouts.jsonl}"
SCOPE_CONFIG="${SCOPE_CONFIG:-$REPO_ROOT/configs/scope/shadow_audit_m1_m2.yaml}"
HARNESS_CONFIG="${HARNESS_CONFIG:-$REPO_ROOT/harness/configs/modules_full.yaml}"
# Default bare-replay: Qwen Instruct is incompatible with Harmony token online path.
# Online multi-turn DecisionState audit requires a Harmony-compatible checkpoint.
MODE="${MODE:-bare-replay}"
SPLIT_LIMIT="${LIMIT:-0}"
MAX_TURNS="${MAX_TURNS:-35}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
VLLM_PORT="${VLLM_PORT:-8772}"
PARALLEL="${PARALLEL:-2}"
RESUME="${RESUME:-1}"
RERANKER="${RERANKER:-none}"
RETRIEVAL="${RETRIEVAL:-bm25}"
BROWSECOMP_BM25_INDEX_PATH="${BROWSECOMP_BM25_INDEX_PATH:-$REPO_ROOT/external/BrowseComp-Plus/indexes/bm25}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="${REPO_ROOT}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX/lib/jvm}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$REPO_ROOT/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}"
export BROWSECOMPPLUS_QUERIES_PATH="${BROWSECOMPPLUS_QUERIES_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/queries.tsv}"
export BROWSECOMP_BM25_INDEX_PATH

# Student runtime (WM) uses full modules; shadow registry is M1+M2 only via SCOPE_CONFIG.
export V8D_SUBTRACTIVE_CURATION="${V8D_SUBTRACTIVE_CURATION:-1}"
export V8D_IMPORTANCE_TAGGING="${V8D_IMPORTANCE_TAGGING:-1}"
export V8D_AUTO_POPULATE_FIRST_SEARCH="${V8D_AUTO_POPULATE_FIRST_SEARCH:-1}"
export V8D_EVIDENCE_GRAPH="${V8D_EVIDENCE_GRAPH:-1}"
export V8D_SENTENCE_COMPRESS="${V8D_SENTENCE_COMPRESS:-1}"
export V8D_CONTENT_DEDUP="${V8D_CONTENT_DEDUP:-1}"
export V8D_VERIFY_TOOL="${V8D_VERIFY_TOOL:-1}"
export V8D_TOKEN_BUDGET_MARKER="${V8D_TOKEN_BUDGET_MARKER:-1}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

echo "=== SCOPE Shadow Audit (Bare → M1/M2) ==="
echo "Mode:            ${MODE}"
echo "Bare jsonl:      ${BARE_JSONL}"
echo "SCOPE config:    ${SCOPE_CONFIG}"
echo "Harness config:  ${HARNESS_CONFIG}"
echo "Model:           ${MODEL_PATH}"
echo "GPUs:            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "max_turns:       ${MAX_TURNS}"
echo "max_tokens:      ${MAX_TOKENS}"
echo "temperature:     ${TEMPERATURE}"
echo "Output:          ${OUTPUT_DIR}"
echo

ARGS=(
  --mode "${MODE}"
  --bare-jsonl "${BARE_JSONL}"
  --config "${SCOPE_CONFIG}"
  --harness-config "${HARNESS_CONFIG}"
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --max-turns "${MAX_TURNS}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --vllm-port "${VLLM_PORT}"
  --tensor-parallel-size 4
  --parallel "${PARALLEL}"
  --reranker "${RERANKER}"
  --retrieval "${RETRIEVAL}"
  --bm25-index-path "${BROWSECOMP_BM25_INDEX_PATH}"
)
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi
if [[ "${SPLIT_LIMIT}" != "0" ]]; then
  ARGS+=(--limit "${SPLIT_LIMIT}")
fi

python training/audit_scope_shadow_bare.py "${ARGS[@]}"

echo
echo "Done. See ${OUTPUT_DIR}/shadow_audit_*_stats.json"
