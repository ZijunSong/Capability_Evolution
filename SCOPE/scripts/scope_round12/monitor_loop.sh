#!/usr/bin/env bash
# Watchdog for Round12 jobs: restart stale GPU work; aggregate when ready.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r12_setup

STALE_SEC="${R12_STALE_SEC:-5400}"  # 90 min
PHASE="${1:-ab}"  # ab|c|auto

r12_log "monitor_loop start phase=${PHASE} stale=${STALE_SEC}s"

is_stale() {
  local hb="$1"
  [[ -f "${hb}" ]] || return 0
  local hb_ts now age
  hb_ts=$(date -d "$(cat "${hb}")" +%s 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$((now - hb_ts))
  [[ "${age}" -gt "${STALE_SEC}" ]]
}

JOBS=(M0_V0 M0_V1 M1_V0 M1_V1 M2_V0 M2_V1)

restart_cross() {
  local gpu="$1" job="$2"
  r12_log "WATCHDOG restart cross gpu=${gpu} job=${job}"
  r12_stop_recorded "vllm_port_$(r12_port_for_gpu "${gpu}")" || true
  pkill -f "run_cross_view_gpu.sh ${gpu} ${job}" 2>/dev/null || true
  sleep 2
  nohup bash "$(dirname "$0")/run_cross_view_gpu.sh" "${gpu}" "${job}" \
    >> "${LOG_DIR}/supervisor_cross_${job}.log" 2>&1 &
  echo $! > "${PID_DIR}/cross_gpu${gpu}.pid"
}

restart_ckpt() {
  local gpu="$1" sel="$2"
  r12_log "WATCHDOG restart ckpt gpu=${gpu} sel=${sel}"
  r12_stop_recorded "vllm_port_$(r12_port_for_gpu "${gpu}")" || true
  pkill -f "run_ckpt_selector_gpu.sh ${gpu} ${sel}" 2>/dev/null || true
  sleep 2
  nohup bash "$(dirname "$0")/run_ckpt_selector_gpu.sh" "${gpu}" "${sel}" \
    >> "${LOG_DIR}/supervisor_ckpt_${sel}.log" 2>&1 &
  echo $! > "${PID_DIR}/ckpt_gpu${gpu}.pid"
}

aggregate_ab() {
  r12_log "Aggregating Barrier A/B"
  python training/scope_round12/build_canonical_ckpt_events.py >> "${LOG_DIR}/agg_a_events.log" 2>&1 || true
  python training/scope_round12/eval_selector_provenance.py >> "${LOG_DIR}/agg_a_prov.log" 2>&1 || true
  python training/scope_round12/ckpt_observability.py >> "${LOG_DIR}/agg_a_obs.log" 2>&1 || true
  python training/scope_round12/calibrate_boundary.py >> "${LOG_DIR}/agg_b_cal.log" 2>&1 || true
  python training/scope_round12/aggregate_round12.py >> "${LOG_DIR}/agg_round12.log" 2>&1 || true
}

while true; do
  if [[ "${PHASE}" == "ab" || "${PHASE}" == "auto" ]]; then
    done_n=0
    for gpu in 0 1 2 3 4 5; do
      job="${JOBS[$gpu]}"
      root="${OUT}/phase_b_operation_boundary/cross_view_replays/${job}"
      if [[ -f "${root}/DONE" ]]; then
        done_n=$((done_n + 1))
        continue
      fi
      if is_stale "${root}/HEARTBEAT"; then
        if [[ -f "${PID_DIR}/cross_gpu${gpu}.pid" ]] || [[ -f "${root}/HEARTBEAT" ]]; then
          restart_cross "${gpu}" "${job}"
        fi
      fi
    done

    ckpt_done=0
    for pair in "6:C11L" "7:C11P"; do
      gpu="${pair%%:*}"
      sel="${pair##*:}"
      if [[ -f "${OUT}/phase_a_ckpt_provenance/per_selector_scores/${sel}_DONE" ]]; then
        ckpt_done=$((ckpt_done + 1))
        continue
      fi
      if is_stale "${OUT}/phase_a_ckpt_provenance/per_selector_scores/${sel}_HEARTBEAT"; then
        if [[ -f "${PID_DIR}/ckpt_gpu${gpu}.pid" ]] || [[ -f "${OUT}/phase_a_ckpt_provenance/per_selector_scores/${sel}_HEARTBEAT" ]]; then
          restart_ckpt "${gpu}" "${sel}"
        fi
      fi
    done

    r12_log "progress cross=${done_n}/6 ckpt=${ckpt_done}/2"
    if [[ "${done_n}" -ge 6 && "${ckpt_done}" -ge 2 ]]; then
      aggregate_ab
      if [[ -f "${OUT}/phase_b_operation_boundary/BARRIER_B_DECISION.json" ]]; then
        allow=$(python -c "import json; print(json.load(open('${OUT}/phase_b_operation_boundary/BARRIER_B_DECISION.json')).get('allow_phase_c_mainline', False))")
        if [[ "${allow}" == "True" ]]; then
          r12_log "Scalar pass — launching Phase C"
          bash "$(dirname "$0")/launch_phase_c_8gpu.sh" || true
          PHASE=c
        else
          r12_log "STOP after operation boundary (no Phase C mainline)"
          exit 0
        fi
      else
        r12_log "ERROR: missing BARRIER_B_DECISION after aggregate"
        exit 1
      fi
    fi
  fi

  if [[ "${PHASE}" == "c" ]]; then
    # Phase C completion handled inside launch/monitor_phase_c; exit when FROZEN or STOP marker
    if [[ -f "${OUT}/FROZEN_LIVE_GATE.json" ]] || [[ -f "${OUT}/markers/STOP_AFTER_PHASE_C" ]]; then
      r12_log "Phase C terminal marker present; monitor exit"
      exit 0
    fi
  fi

  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    >> "${LOG_DIR}/gpu_watch.csv" 2>/dev/null || true
  sleep 120
done
