#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

r13_log "Stage2 degeneracy audit"
python training/scope_round13/stage2_degeneracy_audit.py \
  >> "${LOG_DIR}/stage2_degeneracy_audit.log" 2>&1
r13_log "Stage2 audit DONE"
