#!/usr/bin/env bash
# Download/decrypt BrowseComp+ query files for bare rollout.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
BC_ROOT="${REPO_ROOT}/external/BrowseComp-Plus"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ ! -d "${BC_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/texttron/BrowseComp-Plus "${BC_ROOT}"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "${BC_ROOT}/data" "${BC_ROOT}/topics-qrels"

python "${BC_ROOT}/scripts_build_index/decrypt_dataset.py" \
  --output "${BC_ROOT}/data/browsecomp_plus_decrypted.jsonl" \
  --generate-tsv "${BC_ROOT}/topics-qrels/queries.tsv"

wc -l "${BC_ROOT}/data/browsecomp_plus_decrypted.jsonl"
wc -l "${BC_ROOT}/topics-qrels/queries.tsv"

cat > "${REPO_ROOT}/.env.browsecomp.paths" <<EOF
BROWSECOMPPLUS_QUERIES_PATH=${BC_ROOT}/topics-qrels/queries.tsv
BROWSECOMPPLUS_QRELS_GOLD_PATH=${BC_ROOT}/topics-qrels/qrel_golds.txt
BROWSECOMPPLUS_QRELS_EVIDENCE_PATH=${BC_ROOT}/topics-qrels/qrel_evidence.txt
BROWSECOMPPLUS_ANSWERS_PATH=${BC_ROOT}/data/browsecomp_plus_decrypted.jsonl
EOF

echo "Wrote ${REPO_ROOT}/.env.browsecomp.paths"
