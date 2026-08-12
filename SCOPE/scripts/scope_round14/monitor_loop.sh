#!/usr/bin/env bash
# Round14 monitor loop — status only (relaunch is guardian's job)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

r14_log "monitor start (status-only; no auto-relaunch)"

while true; do
  bash "$(dirname "$0")/status.sh" >> "${LOG_DIR}/status_watch.log" 2>&1 || true
  {
    echo "[$(date -Is)] monitor_snapshot"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
    for gpu in 0 1 2 3 4 5 6 7; do
      case "${gpu}" in
        0) root="${OUT}/gpu0_dup_anchor" ;;
        1) root="${OUT}/gpu1_stop" ;;
        2) root="${OUT}/gpu2_verify_routing" ;;
        3) root="${OUT}/gpu3_evidence_admission" ;;
        4) root="${OUT}/gpu4_context_budget" ;;
        5) root="${OUT}/gpu5_external_verify" ;;
        6) root="${OUT}/gpu6_rollback_lite" ;;
        7) root="${OUT}/gpu7_method_ablation" ;;
      esac
      done_f="pending"
      [[ -f "${root}/DONE" ]] && done_f="DONE"
      echo "gpu${gpu} ${done_f} hb=$(cat "${root}/HEARTBEAT" 2>/dev/null || echo none)"
    done
  } >> "${LOG_DIR}/monitor_snapshot.log"
  sleep 300
done
