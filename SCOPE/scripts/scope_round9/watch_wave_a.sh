#!/usr/bin/env bash
# Watch Wave A for stalls; restart a single GPU queue if its log is stale.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

STALE_SEC="${STALE_SEC:-900}"  # 15 minutes without log update => stall
CHECK_EVERY="${CHECK_EVERY:-120}"

variant_for_gpu() {
  echo "${WAVE_A_VARIANTS[$1]}"
}

is_stale() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local mtime now age
  mtime=$(stat -c %Y "$f")
  now=$(date +%s)
  age=$((now - mtime))
  [[ "$age" -gt "$STALE_SEC" ]]
}

restart_gpu() {
  local gpu="$1"
  local variant
  variant="$(variant_for_gpu "${gpu}")"
  scope9_log "WATCHDOG: restarting Wave A gpu=${gpu} variant=${variant}"
  # Kill only this GPU's run_wave_a and its children recorded PIDs.
  pkill -f "run_wave_a_gpu.sh ${gpu}" 2>/dev/null || true
  scope9_stop_recorded "vllm_${variant}" || true
  local port
  port="$(scope9_port_for_gpu "${gpu}")"
  scope9_stop_recorded "vllm_port_${port}" || true
  # Remove partial outputs for incomplete splits so resume regenerates them.
  rm -f "${MARKER_DIR}/wave_a_${variant}.DONE"
  sleep 3
  nohup bash "$(dirname "$0")/run_wave_a_gpu.sh" "${gpu}" \
    >> "${LOG_DIR}/wave_a_watchdog_${variant}.log" 2>&1 &
  scope9_log "WATCHDOG: relaunched pid=$!"
}

while true; do
  done_n=0
  done_n=$(find "${MARKER_DIR}" -maxdepth 1 -name 'wave_a_*.DONE' 2>/dev/null | wc -l | tr -d ' ' || echo 0)
  if [[ "${done_n}" -ge 8 ]]; then
    scope9_log "WATCHDOG: all 8 Wave A markers present"
    # Verify barrier pass
    fail=0
    for v in "${WAVE_A_VARIANTS[@]}"; do
      report="${OUT}/wave_a/${v}/WAVE_A_REPORT.json"
      if [[ ! -f "${report}" ]]; then
        scope9_log "WATCHDOG: missing report for ${v}"
        fail=1
        continue
      fi
      pass=$(python -c "import json; print(json.load(open('${report}')).get('barrier_a_pass', False))")
      if [[ "${pass}" != "True" ]]; then
        scope9_log "WATCHDOG: Barrier A FAIL ${v}"
        fail=1
      fi
    done
    exit "${fail}"
  fi

  for gpu in 0 1 2 3 4 5 6 7; do
    variant="$(variant_for_gpu "${gpu}")"
    marker="${MARKER_DIR}/wave_a_${variant}.DONE"
    [[ -f "${marker}" ]] && continue
    # Prefer the newest split log as heartbeat.
    newest=$(ls -t "${LOG_DIR}"/wave_a_${variant}_*.log 2>/dev/null | head -1 || true)
    if [[ -n "${newest}" ]] && is_stale "${newest}"; then
      # Confirm a process still exists or GPU idle with unfinished work.
      if pgrep -f "run_wave_a_gpu.sh ${gpu}" >/dev/null; then
        # Skip young processes (model load / vLLM startup can look idle).
        proc_etime=$(ps -o etimes= -p "$(pgrep -f "run_wave_a_gpu.sh ${gpu}" | head -1)" 2>/dev/null | tr -d ' ' || echo 0)
        if [[ "${proc_etime:-0}" -lt "${STALE_SEC}" ]]; then
          continue
        fi
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
        if [[ "${util}" -lt 5 ]]; then
          # Idle past STALE_SEC with unfinished work → hung
          restart_gpu "${gpu}"
        fi
      else
        restart_gpu "${gpu}"
      fi
    elif ! pgrep -f "run_wave_a_gpu.sh ${gpu}" >/dev/null; then
      # Only restart if this Wave A attempt has a fresh log and it went stale.
      # Ignore archived/old logs from previous attempts.
      if [[ -n "${newest}" ]] && is_stale "${newest}"; then
        scope9_log "WATCHDOG: gpu ${gpu} process missing with stale log; restarting"
        restart_gpu "${gpu}"
      fi
    fi
  done
  sleep "${CHECK_EVERY}"
done
