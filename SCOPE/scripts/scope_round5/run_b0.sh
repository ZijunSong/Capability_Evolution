#!/usr/bin/env bash
# Round 5 B0 — freeze environment and run unit tests
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

OUT="${REPO_ROOT}/outputs/scope_round5"
mkdir -p "${OUT}" "${OUT}/logs"

{
  echo "git HEAD: $(git rev-parse HEAD)"
  echo "branch: $(git branch --show-current)"
  echo "date: $(date -Is)"
  python - <<'PY'
import hashlib, torch, transformers, peft
from pathlib import Path
repo = Path(".")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()
print(f"CUDA: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"peft: {peft.__version__}")
print(f"base_model: /data/ppnm/models/Qwen2.5-7B-Instruct")
for label, p in [
    ("round3_train", "artifacts/datasets/dup_sdi_round3/train.jsonl"),
    ("round4_overfit128", "artifacts/datasets/dup_sdi_round4_overfit128/train.jsonl"),
    ("query_manifest", "artifacts/datasets/round2_audit_100q/query_manifest.json"),
    ("modules_minimal_v2", "harness/configs/modules_minimal_v2.yaml"),
]:
    fp = repo / p
    print(f"{label}_sha256: {sha(fp) if fp.exists() else 'MISSING'}")
PY
} > "${OUT}/environment_snapshot.txt"

pytest tests/scope/ -q --tb=no 2>&1 | tee "${OUT}/logs/b0_unit_tests.log" || true
echo "B0 complete"
