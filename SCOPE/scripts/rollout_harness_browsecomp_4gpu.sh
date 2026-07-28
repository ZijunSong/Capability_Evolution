#!/usr/bin/env bash
# Full BrowseComp+ Harness rollout on 4 GPUs.
# Uses local vLLM + OpenAI chat API path (required for Qwen2.5 Instruct;
# Harmony token path is incompatible with non-Harmony checkpoints).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/harness_rollout_browsecomp_full}"
# v2: Ultra ChatDecisionDriver + deterministic truncation (see modules_full_v2.yaml).
# Legacy TokenBudget agent: USE_LEGACY_API_AGENT=1 HARNESS_CONFIG=.../modules_full.yaml
HARNESS_CONFIG="${HARNESS_CONFIG:-$REPO_ROOT/harness/configs/modules_full_v2.yaml}"
export USE_LEGACY_API_AGENT="${USE_LEGACY_API_AGENT:-0}"
SPLIT="${SPLIT:-all}"
LIMIT="${LIMIT:-0}"
MAX_TURNS="${MAX_TURNS:-35}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
VLLM_PORT="${VLLM_PORT:-8771}"
PARALLEL="${PARALLEL:-2}"
RESUME="${RESUME:-1}"
RERANKER="${RERANKER:-none}"
RETRIEVAL="${RETRIEVAL:-bm25}"
BROWSECOMP_BM25_INDEX_PATH="${BROWSECOMP_BM25_INDEX_PATH:-$REPO_ROOT/external/BrowseComp-Plus/indexes/bm25}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-harness-policy}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${REPO_ROOT}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX/lib/jvm}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$REPO_ROOT/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}"
export BROWSECOMPPLUS_QUERIES_PATH="${BROWSECOMPPLUS_QUERIES_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/queries.tsv}"
export BROWSECOMPPLUS_QRELS_GOLD_PATH="${BROWSECOMPPLUS_QRELS_GOLD_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/qrel_golds.txt}"
export BROWSECOMPPLUS_QRELS_EVIDENCE_PATH="${BROWSECOMPPLUS_QRELS_EVIDENCE_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt}"
export BROWSECOMP_BM25_INDEX_PATH

# Harness-1 module flags (full operating point)
export V8D_SUBTRACTIVE_CURATION="${V8D_SUBTRACTIVE_CURATION:-1}"
export V8D_IMPORTANCE_TAGGING="${V8D_IMPORTANCE_TAGGING:-1}"
export V8D_AUTO_POPULATE_FIRST_SEARCH="${V8D_AUTO_POPULATE_FIRST_SEARCH:-1}"
export V8D_EVIDENCE_GRAPH="${V8D_EVIDENCE_GRAPH:-1}"
export V8D_SENTENCE_COMPRESS="${V8D_SENTENCE_COMPRESS:-1}"
export V8D_CONTENT_DEDUP="${V8D_CONTENT_DEDUP:-1}"
export V8D_VERIFY_TOOL="${V8D_VERIFY_TOOL:-1}"
export V8D_TOKEN_BUDGET_MARKER="${V8D_TOKEN_BUDGET_MARKER:-1}"
export SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-1}"
export SAVE_FULL_TRAJECTORIES="${SAVE_FULL_TRAJECTORIES:-0}"
# Weak-policy guards used by ChatDecisionDriver (ultra_chat_v2)
export CHAT_MIN_TURNS_BEFORE_END="${CHAT_MIN_TURNS_BEFORE_END:-8}"
export CHAT_MIN_CURATED_BEFORE_END="${CHAT_MIN_CURATED_BEFORE_END:-1}"
export CHAT_MAX_WM_CHARS="${CHAT_MAX_WM_CHARS:-18000}"
export CHAT_MAX_RECENT_TURNS="${CHAT_MAX_RECENT_TURNS:-4}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -s "${BROWSECOMPPLUS_ANSWERS_PATH}" && ! -s "${BROWSECOMPPLUS_QUERIES_PATH}" ]]; then
  echo "BrowseComp+ data missing; running setup_browsecomp_data.sh ..."
  bash "${REPO_ROOT}/scripts/setup_browsecomp_data.sh"
fi

if [[ "${RETRIEVAL}" == "bm25" ]]; then
  if ! compgen -G "${BROWSECOMP_BM25_INDEX_PATH}/segments_*" > /dev/null; then
    echo "BM25 index missing; running setup_browsecomp_bm25_index.sh ..."
    bash "${REPO_ROOT}/scripts/setup_browsecomp_bm25_index.sh"
  fi
fi

VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1"
VLLM_LOG="${OUTPUT_DIR}/vllm_server.log"
VLLM_PID_FILE="${OUTPUT_DIR}/vllm_server.pid"

cleanup() {
  if [[ -f "${VLLM_PID_FILE}" ]]; then
    vpid="$(cat "${VLLM_PID_FILE}" || true)"
    if [[ -n "${vpid}" ]] && kill -0 "${vpid}" 2>/dev/null; then
      echo "[harness] Stopping vLLM pid=${vpid} ..."
      kill "${vpid}" 2>/dev/null || true
      wait "${vpid}" 2>/dev/null || true
    fi
    rm -f "${VLLM_PID_FILE}"
  fi
}
trap cleanup EXIT

echo "=== BrowseComp+ FULL Harness rollout ==="
echo "Model:           ${MODEL_PATH}"
echo "Harness config:  ${HARNESS_CONFIG}"
echo "GPUs:            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Split:           ${SPLIT} (limit=${LIMIT:-all})"
echo "max_turns:       ${MAX_TURNS}"
echo "max_tokens/turn: ${MAX_TOKENS}"
echo "temperature:     ${TEMPERATURE}"
echo "max_model_len:   ${MAX_MODEL_LEN}"
echo "parallel:        ${PARALLEL}"
echo "reranker:        ${RERANKER}"
echo "retrieval:       ${RETRIEVAL}"
echo "bm25 index:      ${BROWSECOMP_BM25_INDEX_PATH}"
echo "policy:          api-over-local-vLLM (${VLLM_URL})"
echo "Output:          ${OUTPUT_DIR}"
echo

TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
echo "[harness] Starting local vLLM (TP=4) at ${VLLM_URL} ..."
echo "[harness] tool-call-parser=${TOOL_CALL_PARSER}"
nohup vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size 4 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype bfloat16 \
  --disable-custom-all-reduce \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  > "${VLLM_LOG}" 2>&1 &
echo $! > "${VLLM_PID_FILE}"

# Wait for vLLM readiness
python - <<PY
import time, urllib.request, sys
url = "${VLLM_URL}/models"
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print("[harness] vLLM ready:", url, flush=True)
                sys.exit(0)
    except Exception:
        time.sleep(3)
print("[harness] vLLM failed to become ready; see ${VLLM_LOG}", flush=True)
sys.exit(1)
PY

# Point chat-API policy at local vLLM (overrides .env MIFY placeholder).
export base_url="${VLLM_URL}"
export api_key="EMPTY"
export model_name="${SERVED_MODEL_NAME}"

ARGS=(
  --model-path "${MODEL_PATH}"
  --harness-config "${HARNESS_CONFIG}"
  --split "${SPLIT}"
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
  --output-dir "${OUTPUT_DIR}"
  --policy api
  --no-manage-vllm
  --vllm-url "${VLLM_URL}"
)
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi
if [[ "${LIMIT}" != "0" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

python training/rollout_harness_browsecomp.py "${ARGS[@]}"

echo
echo "Done. See ${OUTPUT_DIR}/harness_rollout_manifest.json"
