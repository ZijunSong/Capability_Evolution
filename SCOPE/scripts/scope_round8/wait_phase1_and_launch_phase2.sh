#!/usr/bin/env bash
# Wait for Phase 1, check gates, supplement data if needed, launch Phase 2
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

ROLLBACK_DIRS=(
  "${OUT}/rollback_collection/natural/shard0"
  "${OUT}/rollback_collection/natural/shard1"
  "${OUT}/rollback_collection/natural/shard2"
  "${OUT}/rollback_collection/natural/shard3"
  "${OUT}/rollback_collection/injected/shard0"
  "${OUT}/rollback_collection/injected/shard1"
  "${OUT}/rollback_collection/injected/shard2"
  "${OUT}/rollback_collection/injected/shard3"
)

scope8_log "wait_phase1_and_phase2: polling Phase 1 completion"

while true; do
  complete=$(python - <<'PY'
from training.scope_round8.check_phase1_gates import phase1_complete
print("yes" if phase1_complete() else "no")
PY
)
  if [[ "${complete}" == "yes" ]]; then
    scope8_log "Phase 1 complete"
    break
  fi
  scope8_log "Phase 1 still running..."
  bash "${REPO_ROOT}/scripts/scope_round8/status.sh" | tail -15
  sleep 120
done

scope8_log "Supplementary rollback collection (if needed)"
bash "${REPO_ROOT}/scripts/scope_round8/supplement_rollback_collection.sh" \
  >> "${LOG_DIR}/supplement_rollback.log" 2>&1 || true

scope8_log "Building rollback SDI dataset"
python training/scope_round8/build_rollback_sdi_dataset.py \
  --input-dirs "${ROLLBACK_DIRS[@]}" \
  --output-dir "${REPO_ROOT}/artifacts/datasets/scope_round8/rollback_sdi"

scope8_log "Checking Phase 1 gates"
python training/scope_round8/check_phase1_gates.py

GATE_FILE="${OUT}/HARD_CAPABILITY_GATE.json"
if ! python - <<PY
import json, sys
g=json.load(open("${GATE_FILE}"))
if not g.get("all_gates_pass"):
    print("gates_failed")
    if not g.get("gate_1c",{}).get("gate_1c_pass"):
        sys.exit(2)
    sys.exit(1)
PY
then
  rc=$?
  if [[ "${rc}" -eq 2 ]]; then
    scope8_log "Gate 1C failed — re-run supplement and rebuild once"
    bash "${REPO_ROOT}/scripts/scope_round8/supplement_rollback_collection.sh" \
      >> "${LOG_DIR}/supplement_rollback_retry.log" 2>&1 || true
    python training/scope_round8/build_rollback_sdi_dataset.py \
      --input-dirs "${ROLLBACK_DIRS[@]}" \
      --output-dir "${REPO_ROOT}/artifacts/datasets/scope_round8/rollback_sdi"
    python training/scope_round8/check_phase1_gates.py
  fi
  if ! python -c "import json; g=json.load(open('${GATE_FILE}')); import sys; sys.exit(0 if g.get('all_gates_pass') else 1)"; then
    scope8_log "ERROR: Gates not passed — Phase 2 not launched. See ${GATE_FILE}"
    exit 1
  fi
fi

scope8_log "All gates passed — launching Phase 2"
nohup bash "${REPO_ROOT}/scripts/scope_round8/launch_phase2_8gpu.sh" \
  >> "${LOG_DIR}/launch_phase2.nohup.log" 2>&1 &
echo $! > "${PID_DIR}/launch_phase2.pid"
scope8_log "Phase 2 launched pid=$(cat "${PID_DIR}/launch_phase2.pid")"
