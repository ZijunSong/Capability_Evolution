#!/usr/bin/env bash
# Run Offline Gate then Phase 3 if pass
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

LOG="${LOG_DIR}/offline_and_phase3"
mkdir -p "${LOG}"

scope8_log "Running Offline Gate evaluation"
CUDA_VISIBLE_DEVICES=0 python training/scope_round8/check_offline_gate.py \
  >> "${LOG}/offline_gate.log" 2>&1

if python3 -c "import json; g=json.load(open('${OUT}/OFFLINE_GATE.json')); exit(0 if g.get('phase3_eligible') else 1)"; then
  scope8_log "Offline operation gate PASS — launching Phase 3"
  bash "$(dirname "$0")/launch_phase3_8gpu.sh" >> "${LOG}/phase3_launcher.log" 2>&1
else
  scope8_log "Offline Gate FAIL — Phase 3 not started. See ${OUT}/OFFLINE_GATE.json"
  exit 1
fi
