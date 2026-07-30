#!/usr/bin/env bash
# Round 4: resume from Barrier 2 (after Barrier 1 offline eval completes)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/outputs/scope_round4/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/round4_barrier2_onward.log") 2>&1

echo "=== Round 4 resume Barrier 2+ $(date -Is) ==="

# Verify B1
python - <<'PY'
import json
from pathlib import Path
repo = Path(".")
forced = (repo / "outputs/scope_round4/metric_audit/FORCED_EPISODE_REPORT.md").read_text()
b1_forced = "B1 forced episode PASS:** True" in forced
offline = repo / "outputs/scope_round4/metric_audit/offline_eval_fixed.json"
if not offline.exists():
    raise SystemExit("offline_eval_fixed.json missing — wait for Barrier 1 offline eval")
d = json.loads(offline.read_text())
b1_offline = d.get("b1_offline_valid", False)
b1 = b1_forced and b1_offline
Path("outputs/scope_round4/B1_PASS").write_text(str(b1) + "\n")
print(f"B1_PASS={b1}")
if not b1:
    raise SystemExit("B1 FAIL")
PY

echo "[gate] Barrier 2"
bash scripts/scope_round4/run_barrier2_8gpu.sh

python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/scope_round4/scorer_audit")
mismatch = False
for p in root.glob("*.summary.json"):
    s = json.loads(p.read_text())
    for k in ("train_vs_offline_prediction_mismatch_rate",
              "offline_vs_runtime_prediction_mismatch_rate",
              "prompt_mismatch_rate"):
        if s.get(k, 0) > 0:
            mismatch = True
b2 = not mismatch
Path("outputs/scope_round4/B2_PASS").write_text(str(b2) + "\n")
print(f"B2_PASS={b2}")
if not b2:
    raise SystemExit("B2 FAIL — fix shared scorer before Barrier 3")
PY

echo "[gate] Barrier 3"
bash scripts/scope_round4/run_barrier3_postfix.sh
echo "=== Barriers 2-3 complete $(date -Is) ==="
