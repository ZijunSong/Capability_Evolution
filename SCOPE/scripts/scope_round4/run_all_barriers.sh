#!/usr/bin/env bash
# Round 4 master orchestrator — runs barriers sequentially with gates (nohup safe)
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
MASTER_LOG="${LOG_DIR}/round4_master.log"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=== Round 4 master start $(date -Is) ==="

# --- Barrier 1 ---
echo "[gate] Barrier 1: metric audit"
bash scripts/scope_round4/run_barrier1.sh

# Wait for background offline eval (nohup child of barrier1, not our child — poll by PID)
OFFLINE_PID="$(cat "${LOG_DIR}/barrier1_offline_eval.pid" 2>/dev/null || true)"
if [[ -n "${OFFLINE_PID}" ]]; then
  echo "[gate] waiting for offline eval PID=${OFFLINE_PID}"
  while kill -0 "${OFFLINE_PID}" 2>/dev/null; do
    sleep 30
    tail -1 "${LOG_DIR}/barrier1_offline_eval.log" 2>/dev/null || true
  done
  echo "[gate] offline eval finished"
fi

# Check B1 pass
python - <<'PY'
import json
from pathlib import Path
repo = Path(".")
forced = (repo / "outputs/scope_round4/metric_audit/FORCED_EPISODE_REPORT.md").read_text()
b1_forced = "B1 forced episode PASS:** True" in forced
offline = repo / "outputs/scope_round4/metric_audit/offline_eval_fixed.json"
b1_offline = False
if offline.exists():
    d = json.loads(offline.read_text())
    b1_offline = d.get("b1_offline_valid", False)
b1 = b1_forced and b1_offline
print(f"B1_PASS={b1} (forced={b1_forced}, offline={b1_offline})")
Path("outputs/scope_round4/B1_PASS").write_text(str(b1) + "\n")
if not b1:
    raise SystemExit("B1 FAIL — stopping before Barrier 2")
PY

# --- Barrier 2 ---
echo "[gate] Barrier 2: scorer consistency"
nohup bash scripts/scope_round4/run_barrier2_8gpu.sh > "${LOG_DIR}/barrier2_master.log" 2>&1 &
B2_PID=$!
echo "${B2_PID}" > "${LOG_DIR}/barrier2_master.pid"
wait "${B2_PID}"

python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/scope_round4/scorer_audit")
mismatch = False
for p in root.glob("*.summary.json"):
    s = json.loads(p.read_text())
    if s.get("train_vs_offline_prediction_mismatch_rate", 0) > 0:
        mismatch = True
    if s.get("offline_vs_runtime_prediction_mismatch_rate", 0) > 0:
        mismatch = True
    if s.get("prompt_mismatch_rate", 0) > 0:
        mismatch = True
b2 = not mismatch
print(f"B2_PASS={b2} (any_mismatch={mismatch})")
Path("outputs/scope_round4/B2_PASS").write_text(str(b2) + "\n")
if not b2:
    print("B2 FAIL — scorer mismatch detected; fix shared scorer before Barrier 3")
    raise SystemExit(1)
PY

# --- Barrier 3 ---
echo "[gate] Barrier 3: postfix replay"
bash scripts/scope_round4/run_barrier3_postfix.sh

echo "=== Round 4 barriers 1-3 complete $(date -Is) ==="
echo "Review outputs/scope_round4/ and decide Barrier 4+ based on checkpoint recovery."
