#!/usr/bin/env bash
# SCOPE v3 protocol smoke: BrowseComp+ 20 queries, Dup + Premature Stop only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/scope_v3_protocol_smoke20}"
SCOPE_CONFIG="${SCOPE_CONFIG:-$REPO_ROOT/configs/scope/sdi_dup_premature.yaml}"
HARNESS_CONFIG="${HARNESS_CONFIG:-$REPO_ROOT/harness/configs/modules_full_v2.yaml}"
LIMIT="${LIMIT:-20}"
SEED="${SEED:-42}"
MAX_TURNS="${MAX_TURNS:-35}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
VLLM_PORT="${VLLM_PORT:-8774}"
PARALLEL="${PARALLEL:-2}"
RESUME="${RESUME:-0}"
RETRIEVAL="${RETRIEVAL:-bm25}"
RERANKER="${RERANKER:-none}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-scope-v3-smoke}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
BROWSECOMP_BM25_INDEX_PATH="${BROWSECOMP_BM25_INDEX_PATH:-$REPO_ROOT/external/BrowseComp-Plus/indexes/bm25}"

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

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1"
VLLM_LOG="${OUTPUT_DIR}/vllm_server.log"
VLLM_PID_FILE="${OUTPUT_DIR}/vllm_server.pid"

cleanup() {
  if [[ -f "${VLLM_PID_FILE}" ]]; then
    vpid="$(cat "${VLLM_PID_FILE}" || true)"
    if [[ -n "${vpid}" ]] && kill -0 "${vpid}" 2>/dev/null; then
      echo "[v3-smoke] Stopping vLLM pid=${vpid} ..."
      kill "${vpid}" 2>/dev/null || true
      wait "${vpid}" 2>/dev/null || true
    fi
    rm -f "${VLLM_PID_FILE}"
  fi
}
trap cleanup EXIT

echo "=== SCOPE v3 protocol smoke (20q) ==="
echo "SCOPE config:    ${SCOPE_CONFIG}"
echo "Harness config:  ${HARNESS_CONFIG}"
echo "GPUs:            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Output:          ${OUTPUT_DIR}"
echo

echo "[v3-smoke] Starting vLLM TP=4 ..."
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

python - <<PY
import time, urllib.request, sys
url = "${VLLM_URL}/models"
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print("[v3-smoke] vLLM ready:", url, flush=True)
                sys.exit(0)
    except Exception:
        time.sleep(3)
print("[v3-smoke] vLLM failed; see ${VLLM_LOG}", flush=True)
sys.exit(1)
PY

export base_url="${VLLM_URL}"
export api_key="EMPTY"
export model_name="${SERVED_MODEL_NAME}"

ARGS=(
  --config "${SCOPE_CONFIG}"
  --harness-config "${HARNESS_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --model-path "${MODEL_PATH}"
  --limit "${LIMIT}"
  --seed "${SEED}"
  --max-turns "${MAX_TURNS}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --parallel "${PARALLEL}"
  --vllm-port "${VLLM_PORT}"
  --tensor-parallel-size 4
  --vllm-model-name "${SERVED_MODEL_NAME}"
  --no-manage-vllm
  --vllm-url "${VLLM_URL}"
  --retrieval "${RETRIEVAL}"
  --bm25-index-path "${BROWSECOMP_BM25_INDEX_PATH}"
  --reranker "${RERANKER}"
  --query-timeout-s "${QUERY_TIMEOUT_S:-600}"
)
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi

python training/audit_scope_v3_protocol_smoke.py "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/nohup_smoke.log"

echo "Done. See ${OUTPUT_DIR}/summary.json"
