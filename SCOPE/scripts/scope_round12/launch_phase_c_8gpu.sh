#!/usr/bin/env bash
# Phase C — only when SCALAR_BOUNDARY_REPAIR_PASS (invoked by monitor).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r12_setup

DEC="${OUT}/phase_b_operation_boundary/BARRIER_B_DECISION.json"
if [[ ! -f "${DEC}" ]]; then
  r12_log "ERROR missing ${DEC}"
  exit 2
fi
allow=$(python -c "import json; print(json.load(open('${DEC}')).get('allow_phase_c_mainline', False))")
if [[ "${allow}" != "True" ]]; then
  r12_log "Phase C blocked by Barrier B decision"
  touch "${MARKER_DIR}/STOP_AFTER_PHASE_C"
  exit 0
fi

# Observability gate for Stage2 retrain
obs="${OUT}/phase_a_ckpt_provenance/CKPT_OBSERVABILITY.json"
not_id=$(python -c "import json; print(json.load(open('${obs}')).get('CKPT_TARGET_NOT_IDENTIFIABLE', True))" 2>/dev/null || echo True)

# GPU0/1: full_stage1 seed43/44 (seed42 reuse)
# GPU2/3/4: listwise seeds if identifiable
declare -a PLAN=()
PLAN+=("0:full_stage1_seed42")
PLAN+=("1:full_stage1_seed43")
PLAN+=("5:full_stage1_seed44")
if [[ "${not_id}" == "False" ]]; then
  PLAN+=("2:ckpt_canonical_listwise_seed42")
  PLAN+=("3:ckpt_canonical_listwise_seed43")
  PLAN+=("4:ckpt_canonical_listwise_seed44")
else
  r12_log "Stage2 retrain skipped: CKPT_TARGET_NOT_IDENTIFIABLE=true"
fi

for item in "${PLAN[@]}"; do
  gpu="${item%%:*}"
  var="${item##*:}"
  r12_stop_recorded "phase_c_gpu${gpu}" || true
  r12_stop_recorded "vllm_port_$(r12_port_for_gpu "${gpu}")" || true
  nohup bash "$(dirname "$0")/run_phase_c_gpu.sh" "${gpu}" "${var}" \
    >> "${LOG_DIR}/supervisor_phase_c_${var}.log" 2>&1 &
  echo $! > "${PID_DIR}/phase_c_gpu${gpu}.pid"
  r12_log "started Phase C GPU${gpu} ${var} pid=$!"
  sleep 25
done

# Wait for completion (simple poll)
while true; do
  done_n=0
  total=${#PLAN[@]}
  for item in "${PLAN[@]}"; do
    var="${item##*:}"
    [[ -f "${OUT}/phase_c/${var}/DONE" ]] && done_n=$((done_n + 1))
  done
  r12_log "Phase C progress ${done_n}/${total}"
  if [[ "${done_n}" -ge "${total}" ]]; then
    python training/scope_round12/aggregate_phase_c_gate.py >> "${LOG_DIR}/phase_c_gate.log" 2>&1 || true
    touch "${MARKER_DIR}/STOP_AFTER_PHASE_C"
    break
  fi
  # stale restart
  for item in "${PLAN[@]}"; do
    gpu="${item%%:*}"
    var="${item##*:}"
    vdir="${OUT}/phase_c/${var}"
    [[ -f "${vdir}/DONE" ]] && continue
    if [[ -f "${vdir}/HEARTBEAT" ]]; then
      hb_ts=$(date -d "$(cat "${vdir}/HEARTBEAT")" +%s 2>/dev/null || echo 0)
      age=$(( $(date +%s) - hb_ts ))
      if [[ "${age}" -gt 5400 ]]; then
        r12_log "WATCHDOG restart Phase C ${var}"
        r12_stop_recorded "vllm_port_$(r12_port_for_gpu "${gpu}")" || true
        pkill -f "run_phase_c_gpu.sh ${gpu} ${var}" 2>/dev/null || true
        sleep 2
        nohup bash "$(dirname "$0")/run_phase_c_gpu.sh" "${gpu}" "${var}" \
          >> "${LOG_DIR}/supervisor_phase_c_${var}.log" 2>&1 &
        echo $! > "${PID_DIR}/phase_c_gpu${gpu}.pid"
      fi
    fi
  done
  sleep 120
done
r12_log "Phase C launcher exit"
