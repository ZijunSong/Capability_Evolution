#!/usr/bin/env bash
# Lightweight status logger every INTERVAL seconds.
set -euo pipefail
source "$(dirname "$0")/_common_followup.sh"
followup_setup
INTERVAL="${FOLLOWUP_MONITOR_INTERVAL:-300}"
STATUS="${LOG_DIR}/monitor_status.log"

followup_log "monitor_followup_loop start interval=${INTERVAL}s"
while true; do
  {
    echo "===== $(date -Is) ====="
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
    echo "-- phase_b --"
    for v in "${PHASE_B_VARIANTS[@]}"; do
      d="${OUT}/phase_b/${v}"
      done_f=$([[ -f "${d}/DONE" ]] && echo DONE || echo ...)
      merged=$([[ -f "${d}/merged/config.json" ]] && echo merged || echo -)
      hb=$(cat "${d}/HEARTBEAT" 2>/dev/null || echo none)
      echo "  ${v}: ${done_f} ${merged} hb=${hb}"
    done
    echo "-- procs --"
    ps -eo pid,etime,pcpu,cmd | grep -E 'run_phase_b_train|run_vllm_replay|rollback_closed_loop|watch_followup|followup_continuum' | grep -v grep || true
    for g in PHASE_B_GATE.json SMOKE20_GATE.json FINAL100_GATE.json; do
      if [[ -f "${OUT}/${g}" ]]; then
        python -c "import json; d=json.load(open('${OUT}/${g}')); print('${g}', 'pass=', d.get('pass'))" 2>/dev/null || echo "${g} present"
      fi
    done
  } | tee -a "${STATUS}"
  # exit when final report written or all gates decided with stop
  if [[ -f "${OUT}/ROUND10_FOLLOWUP_REPORT.md" && -f "${OUT}/ROOT_CAUSE_DECISION.json" ]]; then
    # keep running until continuum exits unless STOP after B without continuum
    if ! pgrep -f 'followup_continuum.sh' >/dev/null 2>&1; then
      if [[ -f "${OUT}/PHASE_B_GATE.json" ]]; then
        followup_log "monitor: final artifacts present and continuum idle — exit"
        exit 0
      fi
    fi
  fi
  sleep "${INTERVAL}"
done
