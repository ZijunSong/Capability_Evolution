#!/usr/bin/env bash
# Round14 guardian: stale heartbeat → kill vllm + relaunch queue
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

STALE_SEC="${R14_STALE_SEC:-1800}"
r14_log "guardian start stale=${STALE_SEC}s"

is_stale() {
  local hb="$1"
  # Missing heartbeat is NOT automatically stale (avoid killing wave0 / early jobs).
  [[ -f "${hb}" ]] || return 1
  local hb_ts now age
  hb_ts=$(date -d "$(cat "${hb}" | head -1)" +%s 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$((now - hb_ts))
  [[ "${age}" -gt "${STALE_SEC}" ]]
}

gpu_pid_alive() {
  local gpu="$1"
  local f="${PID_DIR}/gpu${gpu}.pid"
  [[ -f "${f}" ]] || return 1
  kill -0 "$(cat "${f}")" 2>/dev/null
}

wave0_alive() {
  local f="${PID_DIR}/wave0.pid"
  [[ -f "${f}" ]] && kill -0 "$(cat "${f}")" 2>/dev/null
}

relaunch_gpu() {
  local gpu="$1"
  r14_log "guardian restart GPU${gpu} (stale heartbeat)"
  r14_stop_recorded "gpu${gpu}"
  r14_kill_vllm_on_gpu "${gpu}"
  sleep 3
  nohup bash "$(dirname "$0")/run_gpu${gpu}_queue.sh" "${gpu}" \
    >> "${LOG_DIR}/gpu${gpu}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/gpu${gpu}.pid"
  # Seed a fresh heartbeat so we don't thrash
  case "${gpu}" in
    0) r14_touch_hb "${OUT}/gpu0_dup_anchor/HEARTBEAT" ;;
    1) r14_touch_hb "${OUT}/gpu1_stop/HEARTBEAT" ;;
    2) r14_touch_hb "${OUT}/gpu2_verify_routing/HEARTBEAT" ;;
    3) r14_touch_hb "${OUT}/gpu3_evidence_admission/HEARTBEAT" ;;
    4) r14_touch_hb "${OUT}/gpu4_context_budget/HEARTBEAT" ;;
    5) r14_touch_hb "${OUT}/gpu5_external_verify/HEARTBEAT" ;;
    6) r14_touch_hb "${OUT}/gpu6_rollback_lite/HEARTBEAT" ;;
    7) r14_touch_hb "${OUT}/gpu7_method_ablation/HEARTBEAT" ;;
  esac
}

while true; do
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
    hb="${root}/HEARTBEAT"
    done="${root}/DONE"
    [[ -f "${done}" ]] && continue

    # During wave0, GPUs 0-4 are owned by wave0 jobs — do not relaunch queues.
    if wave0_alive && [[ "${gpu}" -le 4 ]]; then
      r14_touch_hb "${OUT}/gpu0_dup_anchor/HEARTBEAT"
      continue
    fi

    if is_stale "${hb}"; then
      # If queue pid still alive and producing GPU mem, just refresh note in log
      if gpu_pid_alive "${gpu}"; then
        r14_log "guardian: GPU${gpu} stale HB but pid alive — kill+relaunch"
      fi
      relaunch_gpu "${gpu}"
    fi
  done
  {
    echo "[$(date -Is)] gpu_watch"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
  } >> "${LOG_DIR}/gpu_watch.csv"
  sleep 120
done
