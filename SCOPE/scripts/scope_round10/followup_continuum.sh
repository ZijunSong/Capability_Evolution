#!/usr/bin/env bash
# Wait for Phase B → gate → optional Phase C → optional Phase D → final report.
set -euo pipefail
source "$(dirname "$0")/_common_followup.sh"
followup_setup

followup_log "followup_continuum: waiting for Phase B DONE markers"
while true; do
  done_n=0
  for v in "${PHASE_B_VARIANTS[@]}"; do
    [[ -f "${OUT}/phase_b/${v}/DONE" ]] && done_n=$((done_n + 1))
  done
  if [[ "${done_n}" -ge 8 ]]; then
    followup_log "all Phase B DONE (${done_n}/8)"
    break
  fi
  # also break if PHASE_B_GATE already written and launcher exited
  if [[ -f "${OUT}/PHASE_B_GATE.json" ]]; then
    alive=0
    for gpu in 0 1 2 3 4 5 6 7; do
      pf="${PID_DIR}/phase_b_gpu${gpu}.pid"
      if [[ -f "${pf}" ]] && kill -0 "$(cat "${pf}")" 2>/dev/null; then
        alive=1
      fi
    done
    if [[ "${alive}" -eq 0 && "${done_n}" -ge 7 ]]; then
      followup_log "Phase B mostly done (${done_n}/8), gate present — proceed to aggregate"
      break
    fi
  fi
  sleep 120
done

python training/scope_round10/aggregate_followup_phase_b_gate.py \
  >> "${LOG_DIR}/phase_b_aggregate_continuum.log" 2>&1 || true

PASS=$(python -c "import json; print(json.load(open('${OUT}/PHASE_B_GATE.json')).get('pass', False))" 2>/dev/null || echo False)
if [[ "${PASS}" != "True" ]]; then
  followup_log "PHASE_B_GATE FAIL — STOP_AFTER_PHASE_B"
  python training/scope_round10/write_followup_final_report.py
  exit 0
fi

followup_log "PHASE_B_GATE PASS — launch Phase C smoke20"
bash "$(dirname "$0")/launch_followup_phase_c_smoke20_8gpu.sh" \
  >> "${LOG_DIR}/phase_c_launch.out" 2>&1 || {
  followup_log "Phase C failed or gate FAIL"
  python training/scope_round10/write_followup_final_report.py
  exit 0
}

SMOKE=$(python -c "import json; print(json.load(open('${OUT}/SMOKE20_GATE.json')).get('pass', False))" 2>/dev/null || echo False)
if [[ "${SMOKE}" != "True" ]]; then
  followup_log "SMOKE20 FAIL — STOP_AFTER_PHASE_C"
  python training/scope_round10/write_followup_final_report.py
  exit 0
fi

followup_log "SMOKE20 PASS — launch Phase D final100"
bash "$(dirname "$0")/launch_followup_phase_d_final100_8gpu.sh" \
  >> "${LOG_DIR}/phase_d_launch.out" 2>&1 || true

python training/scope_round10/write_followup_final_report.py
followup_log "followup_continuum complete"
