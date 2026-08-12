#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup
scope9_log "Round 9 preflight"

pytest tests/scope/ tests/scope_round9/ -q --tb=short 2>&1 | tee "${OUT}/preflight/pytest.log"
if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
  scope9_log "ERROR: pytest failed"
  exit 1
fi

python training/scope_round9/reaggregate_round8.py 2>&1 | tee "${OUT}/preflight/reaggregate.log"
scope9_log "Preflight complete"
