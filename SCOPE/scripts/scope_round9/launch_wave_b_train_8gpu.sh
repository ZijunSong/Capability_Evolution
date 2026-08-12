#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

python training/scope_round9/build_hier_sdi_dataset.py 2>&1 | tee "${LOG_DIR}/build_hier_sdi.log"
python - <<'PY'
import json, sys
from pathlib import Path
gate = json.loads(Path("artifacts/datasets/scope_round9/hier_sdi/DATASET_GATE.json").read_text())
print(json.dumps(gate, indent=2))
if not gate.get("gate_pass"):
    sys.exit(2)
PY

for gpu in 0 1 2 3 4 5 6 7; do
  bash "$(dirname "$0")/run_wave_b_gpu.sh" "${gpu}" \
    >> "${LOG_DIR}/wave_b_supervisor.log" 2>&1 &
  sleep 3
done
wait

done_n=0
for v in "${WAVE_B_VARIANTS[@]}"; do
  [[ -f "${OUT}/wave_b/${v}/DONE" ]] && done_n=$((done_n + 1))
done
if [[ "${done_n}" -lt 8 ]]; then
  scope9_log "ERROR: Wave B incomplete (${done_n}/8)"
  exit 1
fi

python training/scope_round9/check_offline_gate_round9.py \
  --output "${OUT}/OFFLINE_GATE_ROUND9.json"
scope9_log "Wave B training complete"
