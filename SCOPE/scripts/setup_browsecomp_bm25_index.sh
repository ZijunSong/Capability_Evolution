#!/usr/bin/env bash
# Download BrowseComp+ official BM25 Lucene index from HuggingFace.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX_ROOT="${BROWSECOMP_BM25_INDEX_ROOT:-$REPO_ROOT/external/BrowseComp-Plus/indexes}"
BM25_DIR="${BROWSECOMP_BM25_INDEX_PATH:-$INDEX_ROOT/bm25}"
HF_REPO="${BROWSECOMP_BM25_HF_REPO:-Tevatron/browsecomp-plus-indexes}"

PROXY_URL="${PROXY_URL:-http://127.0.0.1:7890}"
if curl -s -o /dev/null --max-time 5 -x "${PROXY_URL}" https://huggingface.co 2>/dev/null; then
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
fi

source "${CONDA_BASE:-/data/ppnm/miniconda3}/etc/profile.d/conda.sh"
conda activate "${BISHOP_CONDA_ENV:-bishop}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${REPO_ROOT}"

mkdir -p "${INDEX_ROOT}"

echo "=== Download BrowseComp+ BM25 index ==="
echo "HF repo:     ${HF_REPO}"
echo "HF endpoint: ${HF_ENDPOINT}"
echo "Output dir:  ${INDEX_ROOT}"
echo

if compgen -G "${BM25_DIR}/segments_*" > /dev/null; then
  echo "BM25 index already present at ${BM25_DIR}"
  exit 0
fi

if ! python -c "import pyserini" 2>/dev/null; then
  echo "Installing pyserini ..."
  pip install -q "pyserini>=1.2.0"
fi

if ! command -v javac >/dev/null 2>&1; then
  echo "WARNING: javac not found. Pyserini/Lucene requires Java 21+."
  echo "Install with: conda install -c conda-forge openjdk=21"
fi

if command -v hf >/dev/null 2>&1; then
  set +e
  hf download "${HF_REPO}" --repo-type dataset --include "bm25/*" --local-dir "${INDEX_ROOT}"
  DL_STATUS=$?
  set -e
  if [[ "${DL_STATUS}" -ne 0 ]]; then
    echo "hf download failed (exit ${DL_STATUS}); retrying via huggingface_hub ..."
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    repo_type="dataset",
    allow_patterns=["bm25/*"],
    local_dir="${INDEX_ROOT}",
)
print("snapshot_download OK")
PY
  fi
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${HF_REPO}" --repo-type dataset --include "bm25/*" --local-dir "${INDEX_ROOT}"
else
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    repo_type="dataset",
    allow_patterns=["bm25/*"],
    local_dir="${INDEX_ROOT}",
)
print("snapshot_download OK")
PY
fi

RESOLVED="$(python - <<'PY'
from harness.retrieval.bm25_backend import resolve_bm25_index_path
print(resolve_bm25_index_path())
PY
)"

echo
echo "BM25 index ready: ${RESOLVED}"
echo "Export: export BROWSECOMP_BM25_INDEX_PATH='${RESOLVED}'"
