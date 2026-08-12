#!/usr/bin/env bash
# Full Round11 continuum: Phase A -> Phase B -> Gate -> (optional C/D).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r11_setup

r11_log "=== Round11 continuum start ==="

# Phase A
if [[ ! -f "${OUT}/phase_a_state_factorization/PHASE_A_DECISION.json" ]]; then
  bash "$(dirname "$0")/launch_phase_a_8gpu.sh"
else
  r11_log "Phase A decision exists — skip"
fi

# Start monitor in background
nohup bash "$(dirname "$0")/monitor_loop.sh" auto \
  >> "${LOG_DIR}/monitor_loop.log" 2>&1 &
echo $! > "${PID_DIR}/monitor_loop.pid"
r11_log "monitor_loop pid=$(cat "${PID_DIR}/monitor_loop.pid")"

# Phase B
if [[ ! -f "${OUT}/FROZEN_LIVE_GATE.json" ]]; then
  bash "$(dirname "$0")/launch_phase_b_8gpu.sh"
else
  r11_log "FROZEN_LIVE_GATE exists — skip Phase B launch"
fi

PASS=$(python -c "import json; print(json.load(open('${OUT}/FROZEN_LIVE_GATE.json')).get('pass', False))")
if [[ "${PASS}" != "True" ]]; then
  r11_log "FROZEN_LIVE_GATE.pass=false — STOP_AFTER_PHASE_B; writing final reports"
  python training/scope_round11/write_final_report.py || true
  exit 0
fi

r11_log "Gate passed — Phase C/D launchers not auto-expanded in this continuum stub; see todo §8-9"
python training/scope_round11/write_final_report.py || true
