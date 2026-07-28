#!/usr/bin/env bash
# Offline smoke: Harness + BM25 without API keys, Java, or Lucene index.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CONDA_BASE:-/data/ppnm/miniconda3}/etc/profile.d/conda.sh"
conda activate "${BISHOP_CONDA_ENV:-bishop}"

export PYTHONPATH="${REPO_ROOT}"
export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$REPO_ROOT/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}"
export BROWSECOMPPLUS_QUERIES_PATH="${BROWSECOMPPLUS_QUERIES_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/queries.tsv}"
export BROWSECOMPPLUS_QRELS_GOLD_PATH="${BROWSECOMPPLUS_QRELS_GOLD_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/qrel_golds.txt}"
export BROWSECOMPPLUS_QRELS_EVIDENCE_PATH="${BROWSECOMPPLUS_QRELS_EVIDENCE_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt}"

cd "${REPO_ROOT}"

echo "=== pytest: harness bm25 smoke ==="
python -m pytest tests/test_harness_bm25_smoke.py tests/test_bm25_retrieval.py -q

echo
echo "=== script: smoke_harness_bm25 ==="
python training/smoke_harness_bm25.py

echo
echo "=== rollout tools-only smoke ==="
python training/rollout_harness_browsecomp.py \
  --smoke-retrieval \
  --tools-only \
  --queries-json tests/fixtures/browsecomp_sample_queries.json \
  --limit 1 \
  --no-manage-vllm \
  --output-dir outputs/smoke_harness_bm25_cli

echo
echo "All offline Harness BM25 smoke checks completed."
