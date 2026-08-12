#!/usr/bin/env bash
# Round12 full continuum: preflight → Barrier A CPU → 8gpu A/B → conditional C.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r12_setup

r12_log "======== Round12 continuum start ========"

r12_log "Barrier 0 preflight"
python training/scope_round12/preflight.py >> "${LOG_DIR}/preflight.log" 2>&1 || r12_log "WARN preflight non-zero (see PREFLIGHT.json)"

r12_log "Barrier A CPU: canonical events + initial provenance + observability"
python training/scope_round12/build_canonical_ckpt_events.py >> "${LOG_DIR}/a1_events.log" 2>&1
python training/scope_round12/eval_selector_provenance.py >> "${LOG_DIR}/a2_prov.log" 2>&1
python training/scope_round12/ckpt_observability.py >> "${LOG_DIR}/a4_obs.log" 2>&1

# Early scalar calibration on existing M0×A0 (may update after cross-view completes)
r12_log "Early scalar calibration on existing M0×A0 replays"
python training/scope_round12/calibrate_boundary.py >> "${LOG_DIR}/b_early_cal.log" 2>&1 || true

r12_log "Launch 8gpu Barrier A/B parallel jobs"
bash "$(dirname "$0")/launch_barrier_ab_8gpu.sh"

r12_log "======== continuum launcher returned (monitor continues in background) ========"
