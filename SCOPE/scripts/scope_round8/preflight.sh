#!/usr/bin/env bash
# Round 8 preflight: manifest + pytest + config compare
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

scope8_log "Preflight start"
python training/scope_round8/create_query_manifest_830.py
python training/scope_round8/preflight_snapshot.py >> "${LOG_DIR}/preflight.log" 2>&1

scope8_log "Running pytest tests/scope/"
pytest tests/scope/ -q --tb=short 2>&1 | tee "${LOG_DIR}/pytest.log"
if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
  scope8_log "ERROR: pytest failed"
  exit 1
fi

python training/scope_round8/compare_agent_configs.py \
  --output-dir "${OUT}/agent_core_diagnostic/compare"

scope8_log "Preflight complete"
