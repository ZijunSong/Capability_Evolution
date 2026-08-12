#!/usr/bin/env bash
# Watchdog: restart stale Round11 GPU jobs; aggregate when ready.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r11_setup

STALE_SEC="${R11_STALE_SEC:-5400}"  # 90 min
PHASE="${1:-auto}"  # auto|a|b

r11_log "monitor_loop start phase=${PHASE} stale=${STALE_SEC}s"

is_stale() {
  local hb="$1"
  [[ -f "${hb}" ]] || return 0
  local hb_ts now age
  hb_ts=$(date -d "$(cat "${hb}")" +%s 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$((now - hb_ts))
  [[ "${age}" -gt "${STALE_SEC}" ]]
}

restart_phase_a_view() {
  local gpu="$1" view="$2"
  r11_log "WATCHDOG restart Phase A gpu=${gpu} view=${view}"
  # kill recorded vllm
  r11_stop_recorded "vllm_port_$(r11_port_for_gpu "${gpu}")" || true
  # kill lingering python on that gpu log? best-effort
  pkill -f "run_phase_a_view.sh ${gpu} ${view}" 2>/dev/null || true
  sleep 2
  nohup bash "$(dirname "$0")/run_phase_a_view.sh" "${gpu}" "${view}" \
    >> "${LOG_DIR}/phase_a_gpu${gpu}_${view}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/phase_a_gpu${gpu}.pid"
}

restart_phase_b_variant() {
  local gpu="$1"
  local variant="${PHASE_B_VARIANTS[$gpu]}"
  r11_log "WATCHDOG restart Phase B gpu=${gpu} variant=${variant}"
  r11_stop_recorded "vllm_port_$(r11_port_for_gpu "${gpu}")" || true
  pkill -f "run_phase_b_gpu.sh ${gpu}" 2>/dev/null || true
  # Also kill train/eval children bound to this CUDA device if stuck
  sleep 2
  nohup bash "$(dirname "$0")/run_phase_b_gpu.sh" "${gpu}" \
    >> "${LOG_DIR}/phase_b_gpu${gpu}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/phase_b_gpu${gpu}.pid"
}

while true; do
  # Detect phase
  if [[ "${PHASE}" == "auto" ]]; then
    if [[ ! -f "${OUT}/phase_a_state_factorization/PHASE_A_DECISION.json" ]]; then
      CUR=a
    elif [[ ! -f "${OUT}/FROZEN_LIVE_GATE.json" ]]; then
      CUR=b
    else
      r11_log "monitor: gate exists; exiting"
      exit 0
    fi
  else
    CUR="${PHASE}"
  fi

  if [[ "${CUR}" == "a" ]]; then
    done_n=0
    for i in 0 1 2 3 4; do
      view="${PHASE_A_VIEWS[$i]}"
      root="${OUT}/phase_a_state_factorization/${view}"
      if [[ -f "${root}/DONE" ]]; then
        done_n=$((done_n + 1))
        continue
      fi
      if is_stale "${root}/HEARTBEAT"; then
        # only restart if a process was expected
        if [[ -f "${PID_DIR}/phase_a_gpu${i}.pid" ]] || [[ -f "${root}/HEARTBEAT" ]]; then
          restart_phase_a_view "${i}" "${view}"
        fi
      fi
    done
    if [[ "${done_n}" -ge 5 ]]; then
      r11_log "Phase A all DONE — aggregating"
      python training/scope_round11/aggregate_phase_a.py >> "${LOG_DIR}/phase_a_aggregate.log" 2>&1 || true
      if [[ "${PHASE}" == "a" ]]; then
        exit 0
      fi
      PHASE=b
    fi
  fi

  if [[ "${CUR}" == "b" || "${PHASE}" == "b" ]]; then
    done_n=0
    for gpu in 0 1 2 3 4 5 6 7; do
      variant="${PHASE_B_VARIANTS[$gpu]}"
      vdir="${OUT}/phase_b/${variant}"
      if [[ -f "${vdir}/DONE" ]]; then
        done_n=$((done_n + 1))
        continue
      fi
      if is_stale "${vdir}/HEARTBEAT"; then
        if [[ -f "${PID_DIR}/phase_b_gpu${gpu}.pid" ]] || [[ -f "${vdir}/HEARTBEAT" ]]; then
          restart_phase_b_variant "${gpu}"
        fi
      fi
    done
    if [[ "${done_n}" -ge 8 ]]; then
      r11_log "Phase B all DONE — aggregating gate"
      python training/scope_round11/aggregate_phase_b_gate.py >> "${LOG_DIR}/phase_b_aggregate.log" 2>&1 || true
      exit 0
    fi
  fi

  # GPU occupancy snapshot
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    >> "${LOG_DIR}/gpu_watch.csv" 2>/dev/null || true
  sleep 120
done
