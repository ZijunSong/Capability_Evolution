#!/usr/bin/env bash
# Watchdog: monitor followup Phase B, restart stale GPU slots.
set -euo pipefail
source "$(dirname "$0")/_common_followup.sh"
followup_setup

STALE_SEC="${FOLLOWUP_STALE_SEC:-14400}"  # 4h; train heartbeats every 60s so this = true hang
INTERVAL="${FOLLOWUP_WATCH_INTERVAL:-120}"
IDLE_MEM_MIB="${FOLLOWUP_IDLE_MEM_MIB:-2000}"

kill_gpu_tree() {
  local gpu="$1"
  local pidf="${PID_DIR}/phase_b_gpu${gpu}.pid"
  if [[ -f "${pidf}" ]]; then
    local pid; pid="$(cat "${pidf}")"
    if kill -0 "${pid}" 2>/dev/null; then
      followup_log "watchdog: kill GPU${gpu} tree pid=${pid}"
      pkill -P "${pid}" 2>/dev/null || true
      kill "${pid}" 2>/dev/null || true
      sleep 2
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
  # also kill stray CUDA procs on that GPU if vLLM hung
  followup_stop_recorded "vllm_port_$(followup_port_for_gpu "${gpu}")" || true
}

restart_gpu() {
  local gpu="$1"
  local variant="${PHASE_B_VARIANTS[$gpu]}"
  local vdir="${OUT}/phase_b/${variant}"
  followup_log "watchdog: restarting GPU${gpu} (${variant})"
  # clear partial incomplete eval outputs but keep merged model if present
  if [[ ! -f "${vdir}/merged/config.json" ]]; then
    rm -f "${vdir}/DONE"
  else
    # resume from eval
    rm -f "${vdir}/DONE"
  fi
  local log="${LOG_DIR}/phase_b_gpu${gpu}_supervisor.log"
  (
    bash "$(dirname "$0")/run_followup_phase_b_gpu.sh" "${gpu}"
  ) >> "${log}" 2>&1 &
  echo $! > "${PID_DIR}/phase_b_gpu${gpu}.pid"
  date -Is > "${vdir}/HEARTBEAT"
}

GRACE_SEC="${FOLLOWUP_WATCH_GRACE_SEC:-600}"
START_TS=$(date +%s)
followup_log "watchdog start stale=${STALE_SEC}s interval=${INTERVAL}s grace=${GRACE_SEC}s"
while true; do
  all_done=1
  now=$(date +%s)
  # During launch grace, never restart — avoid racing the 8-GPU launcher.
  in_grace=0
  if [[ $((now - START_TS)) -lt "${GRACE_SEC}" ]]; then
    in_grace=1
  fi
  for gpu in 0 1 2 3 4 5 6 7; do
    variant="${PHASE_B_VARIANTS[$gpu]}"
    vdir="${OUT}/phase_b/${variant}"
    mkdir -p "${vdir}"
    if [[ -f "${vdir}/DONE" ]]; then
      continue
    fi
    all_done=0
    hb="${vdir}/HEARTBEAT"
    pidf="${PID_DIR}/phase_b_gpu${gpu}.pid"
    running=0
    if [[ -f "${pidf}" ]] && kill -0 "$(cat "${pidf}")" 2>/dev/null; then
      running=1
    fi
    # Also treat any live train/eval python for this variant as running
    if [[ "${running}" -eq 0 ]] && pgrep -f "run_phase_b_train.py --variant ${variant}" >/dev/null 2>&1; then
      running=1
    fi
    if [[ "${running}" -eq 0 ]]; then
      if [[ "${in_grace}" -eq 1 ]]; then
        followup_log "watchdog: GPU${gpu} not running yet (grace) — wait"
        continue
      fi
      followup_log "watchdog: GPU${gpu} not running and no DONE — restart"
      restart_gpu "${gpu}"
      continue
    fi
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | tr -d ' ' || echo 0)
    if [[ -f "${hb}" ]]; then
      hb_ts=$(date -d "$(cat "${hb}")" +%s 2>/dev/null || echo 0)
      age=$((now - hb_ts))
      # Only treat as hang if heartbeat is stale AND GPU looks idle
      if [[ "${age}" -gt "${STALE_SEC}" && "${mem}" -lt "${IDLE_MEM_MIB}" ]]; then
        followup_log "watchdog: GPU${gpu} STALE age=${age}s mem=${mem}MiB — stop+rerun"
        kill_gpu_tree "${gpu}"
        for split in offline_valid holdout; do
          out="${vdir}/eval_${split}/canonical_vllm_replay.jsonl"
          if [[ -f "${out}" ]]; then
            n=$(wc -l < "${out}" | tr -d ' ')
            expected=402
            [[ "${split}" == "holdout" ]] && expected=3347
            if [[ "${n}" -lt "${expected}" ]]; then
              rm -f "${out}" "${vdir}/eval_${split}/vllm_replay.jsonl"
            fi
          fi
        done
        restart_gpu "${gpu}"
      elif [[ "${age}" -gt "${STALE_SEC}" ]]; then
        followup_log "watchdog: GPU${gpu} hb stale age=${age}s but mem=${mem}MiB — keep waiting"
      fi
    fi
  done
  if [[ "${all_done}" -eq 1 ]]; then
    followup_log "watchdog: all Phase B DONE"
    python training/scope_round10/aggregate_followup_phase_b_gate.py \
      >> "${LOG_DIR}/phase_b_aggregate_watchdog.log" 2>&1 || true
    exit 0
  fi
  # status line
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | tee -a "${LOG_DIR}/watchdog_gpu.csv" >/dev/null || true
  sleep "${INTERVAL}"
done
