#!/usr/bin/env bash
# Official Harness-1 BrowseComp+ vLLM evaluation launcher for SCAPE.
# User authorized running external `pat-jj/harness-1/inference/evaluate_harness1_vllm.py`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARNESS="$ROOT/external/harness-1"
N_QUERIES="${N_QUERIES:-10}"
SEED="${SEED:-42}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/official_harness1_browsecompplus_smoke}"
SCAPE_PYTHON="${SCAPE_PYTHON:-/opt/bishop-harness/bin/python}"
mkdir -p "$RUN_DIR/trajectories"
cd "$HARNESS"
if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export PYTHONPATH=.
export SAVE_TRAJECTORIES=1
export SAVE_FULL_TRAJECTORIES=1
export TRAJECTORY_SAVE_PATH="$RUN_DIR/trajectories"
export V8D_SUBTRACTIVE_CURATION=1
export V8D_IMPORTANCE_TAGGING=1
export V8D_AUTO_POPULATE_FIRST_SEARCH=1
export V8D_EVIDENCE_GRAPH=1
export V8D_SENTENCE_COMPRESS=1
export V8D_CHUNK_NEIGHBORS=0
export V8D_CONTENT_DEDUP=1
export V8D_VERIFY_TOOL=1
export V8D_TOKEN_BUDGET_MARKER=1
export V8D_ADAPTIVE_RERANK_INSTRUCTION=0
export SENTENCE_COMPRESS_K=4
export AUTO_POPULATE_TOP_K=8
export SEARCH_DISPLAY_LIMIT=10
export SEARCH_TOKEN_BUDGET=4096
export MAX_OBS_CHARS=15000
export DOC_SNIPPET_CHARS=120
export CURATED_DOC_CHARS=0
export MAX_TURNS=35
export SCAPE_RETRIEVAL_CORPUS="${SCAPE_RETRIEVAL_CORPUS:-$ROOT/outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl}"
exec "$SCAPE_PYTHON" inference/evaluate_harness1_vllm.py \
  --dataset browsecompplus \
  --split test \
  --collection-split test \
  --n-queries "$N_QUERIES" \
  --seed "$SEED" \
  --max-turns 40 \
  --temperature 1.0 \
  --max-tokens 2048 \
  --parallel "${PARALLEL:-2}" \
  --base-url "${HARNESS1_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --model harness-1 \
  --partial-output "$RUN_DIR/partial_results.jsonl" \
  --output "$RUN_DIR/eval_results.json"
