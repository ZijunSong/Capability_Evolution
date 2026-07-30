#!/usr/bin/env bash
# Master Round 6 pipeline — sequential phases via nohup
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round6/_common.sh"
scope6_setup

scope6_snapshot_env
scope6_set_stage "phase_a"

scope6_log "=== Phase A: tests ==="
python -m pytest tests/scope/test_round6_scorer.py -q >> "${LOG_DIR}/pytest_round6.log" 2>&1

STAGE=$(scope6_get_stage)
scope6_log "Starting from stage check; current=${STAGE}"

scope6_log "=== Phase B: forensic audit ==="
bash "${REPO_ROOT}/scripts/scope_round6/run_phase_b_8gpu.sh" all

GATE="${OUT}/phase_b/ROOT_CAUSE_GATE.json"
H_RUNTIME=$(python -c "import json; d=json.load(open('${GATE}')); print(d.get('H_RUNTIME', False))")
H_CALIB=$(python -c "import json; d=json.load(open('${GATE}')); print(d.get('H_CALIB', False))")
H_SHIFT=$(python -c "import json; d=json.load(open('${GATE}')); print(d.get('H_SHIFT', False))")

scope6_log "Gate: H_RUNTIME=${H_RUNTIME} H_CALIB=${H_CALIB} H_SHIFT=${H_SHIFT}"

if [[ "${H_RUNTIME}" == "True" ]]; then
  scope6_log "C-RUNTIME branch: vLLM scorer fix applied in code; re-running parity subset"
  python training/scope_round6/run_parity_audit.py --mode adapter_merged --scorer o7_42 --gpu cuda:0 --n-states 128
  python training/scope_round6/root_cause_gate.py
fi

if [[ "${H_CALIB}" == "True" ]] || [[ "${H_RUNTIME}" != "True" ]]; then
  scope6_log "=== Phase C-CALIB ==="
  bash "${REPO_ROOT}/scripts/scope_round6/run_phase_c_calib.sh"
fi

if [[ "${H_SHIFT}" == "True" ]]; then
  scope6_log "=== Phase C-SHIFT noted; skipping full retrain in automated pass (see todo) ==="
  echo "C_SHIFT_REQUIRED=true" > "${OUT}/C_SHIFT_FLAG"
fi

scope6_log "=== Phase D: 50q holdout ==="
bash "${REPO_ROOT}/scripts/scope_round6/run_phase_d_8gpu.sh"

python training/scope_round6/build_round6_report.py
touch "${OUT}/ROUND6_COMPLETE"
scope6_set_stage "done"
scope6_log "=== Round 6 COMPLETE ==="
