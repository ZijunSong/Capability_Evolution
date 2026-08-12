#!/usr/bin/env bash
# Monitor Phase A/B; restart stuck GPU workers; launch Phase B only if Gate A passes.
set -euo pipefail
source "$(dirname "$0")/_common_r10.sh"
scope10_setup

STALL_SEC="${STALL_SEC:-2400}"
INTERVAL="${CHECK_INTERVAL:-120}"

scope10_log "watch_phase_ab started stall=${STALL_SEC}s"

phase_b_launched=0

while true; do
  # Restart dead Phase A workers that are not DONE
  for g in 0 1 2 3 4 5 6 7; do
    marker="${MARKER_DIR}/phase_a_gpu${g}.DONE"
    [[ -f "${marker}" ]] && continue
    if ! pgrep -f "run_phase_a_gpu.sh ${g}" >/dev/null; then
      # check log freshness
      log="${LOG_DIR}/phase_a_gpu${g}_supervisor.log"
      if [[ -f "${log}" ]]; then
        age=$(( $(date +%s) - $(stat -c %Y "${log}") ))
        if [[ "${age}" -lt 30 ]]; then
          continue
        fi
      fi
      scope10_log "RESTART Phase A gpu${g}"
      nohup bash "$(dirname "$0")/run_phase_a_gpu.sh" "${g}" \
        >> "${LOG_DIR}/phase_a_gpu${g}_supervisor.log" 2>&1 &
      echo $! > "${PID_DIR}/phase_a_gpu${g}.pid"
    fi
  done

  # If Phase A all done and gate pass, launch Phase B once
  if [[ ${phase_b_launched} -eq 0 ]] && [[ -f "${OUT}/PARITY_GATE.json" ]]; then
    a_done=1
    for g in 0 1 2 3 4 5 6 7; do
      [[ -f "${MARKER_DIR}/phase_a_gpu${g}.DONE" ]] || a_done=0
    done
    pass=$(python -c "import json; print(json.load(open('${OUT}/PARITY_GATE.json')).get('pass', False))")
    if [[ "${a_done}" -eq 1 && "${pass}" == "True" ]]; then
      scope10_log "Gate A PASS — launching Phase B 8-GPU"
      for g in 0 1 2 3 4 5 6 7; do
        nohup bash "$(dirname "$0")/run_phase_b_gpu.sh" "${g}" \
          >> "${LOG_DIR}/phase_b_gpu${g}_supervisor.log" 2>&1 &
        echo $! > "${PID_DIR}/phase_b_gpu${g}.pid"
      done
      phase_b_launched=1
    elif [[ "${a_done}" -eq 1 && "${pass}" != "True" ]]; then
      scope10_log "Gate A FAIL — STOP_AFTER_PHASE_A (not launching Phase B)"
      phase_b_launched=-1
    fi
  fi

  # Restart Phase B workers if launched
  if [[ ${phase_b_launched} -eq 1 ]]; then
    for g in 0 1 2 3 4 5 6 7; do
      variant="${PHASE_B_VARIANTS[$g]}"
      [[ -f "${OUT}/phase_b/${variant}/DONE" ]] && continue
      if ! pgrep -f "run_phase_b_gpu.sh ${g}" >/dev/null; then
        log="${LOG_DIR}/phase_b_gpu${g}_supervisor.log"
        if [[ -f "${log}" ]]; then
          age=$(( $(date +%s) - $(stat -c %Y "${log}") ))
          if [[ "${age}" -gt "${STALL_SEC}" ]]; then
            scope10_log "RESTART Phase B gpu${g} ${variant} stalled ${age}s"
            # kill any orphan python on that GPU carefully via recorded pid only
            nohup bash "$(dirname "$0")/run_phase_b_gpu.sh" "${g}" \
              >> "${LOG_DIR}/phase_b_gpu${g}_supervisor.log" 2>&1 &
            echo $! > "${PID_DIR}/phase_b_gpu${g}.pid"
          fi
        fi
      fi
    done
    # aggregate when all DONE
    all_b=1
    for g in 0 1 2 3 4 5 6 7; do
      variant="${PHASE_B_VARIANTS[$g]}"
      [[ -f "${OUT}/phase_b/${variant}/DONE" ]] || all_b=0
    done
    if [[ "${all_b}" -eq 1 ]]; then
      scope10_log "Phase B complete — aggregating gates"
      python training/scope_round10/aggregate_phase_b_gate.py \
        >> "${LOG_DIR}/phase_b_aggregate.log" 2>&1 || true
      scope10_log "watch exiting after Phase B aggregate"
      exit 0
    fi
  fi

  if [[ ${phase_b_launched} -eq -1 ]]; then
    scope10_log "watch exiting after Gate A FAIL"
    exit 0
  fi

  sleep "${INTERVAL}"
done
