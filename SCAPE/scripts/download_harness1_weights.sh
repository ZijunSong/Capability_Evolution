#!/usr/bin/env bash
# Background download of pat-jj/harness-1 via HF hub (proxy-friendly).
set -euo pipefail
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-/data/ppnm/models/harness-1}"
LOG="${SCAPE_ROOT}/outputs/preflight/download_harness1.log"
mkdir -p "$(dirname "$DEST")" "${SCAPE_ROOT}/outputs/preflight"

export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7890}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# shellcheck disable=SC1091
source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop

echo "[$(date -Iseconds)] downloading pat-jj/harness-1 -> ${DEST}" | tee -a "$LOG"
python - <<PY 2>&1 | tee -a "$LOG"
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="pat-jj/harness-1",
    local_dir="${DEST}",
    local_dir_use_symlinks=False,
)
print("DONE", p)
PY
echo "[$(date -Iseconds)] finished" | tee -a "$LOG"
