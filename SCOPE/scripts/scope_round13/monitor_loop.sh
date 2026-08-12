#!/usr/bin/env bash
# Watchdog + phase continuum for Round13.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

STALE_SEC="${R13_STALE_SEC:-3600}"  # 60 min
PHASE="${1:-auto}"

r13_log "monitor_loop start phase=${PHASE} stale=${STALE_SEC}s"

is_stale() {
  local hb="$1"
  [[ -f "${hb}" ]] || return 0
  local hb_ts now age
  hb_ts=$(date -d "$(cat "${hb}" | head -1)" +%s 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$((now - hb_ts))
  [[ "${age}" -gt "${STALE_SEC}" ]]
}

restart_collect() {
  local gpu="$1" split="$2" shard="$3" n_shards="$4"
  r13_log "WATCHDOG restart collect gpu=${gpu} ${split}/${shard}"
  # kill vllm on port
  local port
  port="$(r13_port_for_gpu "${gpu}")"
  pkill -f "vllm.*${port}" 2>/dev/null || true
  pkill -f "collect_onpolicy.py.*--vllm-port ${port}" 2>/dev/null || true
  pkill -f "run_collect_gpu.sh ${gpu} ${split}" 2>/dev/null || true
  sleep 3
  # remove DONE if incomplete
  local out_dir
  if [[ "${split}" == "train" ]]; then
    out_dir="${DATA_DIR}/onpolicy_raw/train/${shard}"
  elif [[ "${split}" == "valid" ]]; then
    out_dir="${DATA_DIR}/onpolicy_raw/valid/${shard}"
  else
    out_dir="${DATA_DIR}/onpolicy_raw/test/${shard}"
  fi
  rm -f "${out_dir}/DONE"
  nohup bash "$(dirname "$0")/run_collect_gpu.sh" "${gpu}" "${split}" "${shard}" "${n_shards}" \
    >> "${LOG_DIR}/collect_${split}_${shard}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/collect_${split}_gpu${gpu}.pid"
}

restart_stage1() {
  local gpu="$1"
  local variant="${STAGE1_VARIANTS[$gpu]}"
  r13_log "WATCHDOG restart stage1 gpu=${gpu} variant=${variant}"
  local port
  port="$(r13_port_for_gpu "${gpu}")"
  pkill -f "vllm.*${port}" 2>/dev/null || true
  pkill -f "run_stage1_gpu.sh ${gpu}" 2>/dev/null || true
  sleep 2
  rm -f "${OUT}/phase_b_stage1/training/${variant}/DONE"
  nohup bash "$(dirname "$0")/run_stage1_gpu.sh" "${gpu}" "${variant}" \
    >> "${LOG_DIR}/stage1_${variant}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/stage1_gpu${gpu}.pid"
}

collect_all_done() {
  local n=0
  for gpu in 0 1 2 3 4 5; do
    [[ -f "${DATA_DIR}/onpolicy_raw/train/shard${gpu}/DONE" ]] && n=$((n + 1))
  done
  for gpu in 6 7; do
    local s=$((gpu - 6))
    [[ -f "${DATA_DIR}/onpolicy_raw/valid/shard${s}/DONE" ]] && n=$((n + 1))
  done
  [[ "${n}" -ge 8 ]]
}

stage1_all_done() {
  local n=0
  for v in "${STAGE1_VARIANTS[@]}"; do
    local root="${OUT}/phase_b_stage1/training/${v}"
    if [[ -f "${root}/DONE" ]] && [[ -f "${root}/merged/config.json" ]] \
      && [[ -f "${root}/eval_valid/METRICS.json" ]]; then
      n=$((n + 1))
    fi
  done
  [[ "${n}" -ge 5 ]]
}

advance_after_collect() {
  r13_log "Collect complete — Barrier2 + Barrier3 + Stage2 audit + Stage1 train + TEST collect"
  python training/scope_round13/operation_observability.py \
    >> "${LOG_DIR}/operation_observability.log" 2>&1 || true
  if [[ -f "${OUT}/phase_a_shift/OPERATION_OBSERVABILITY.json" ]]; then
    pass=$(python -c "import json;print(json.load(open('${OUT}/phase_a_shift/OPERATION_OBSERVABILITY.json')).get('OPERATION_OBSERVABILITY_PASS', False))")
    if [[ "${pass}" != "True" ]]; then
      r13_log "STOP: OPERATION_OBSERVABILITY_PASS=false"
      echo "STOP_AFTER_OBSERVABILITY" > "${OUT}/STOP_REASON.txt"
      exit 0
    fi
  fi
  python training/scope_round13/distribution_shift_audit.py \
    >> "${LOG_DIR}/distribution_shift.log" 2>&1 || true
  python training/scope_round13/build_operation_sdi.py \
    >> "${LOG_DIR}/build_operation_sdi.log" 2>&1

  # Stage2 audit (CPU)
  nohup bash "$(dirname "$0")/run_stage2_audit.sh" \
    >> "${LOG_DIR}/stage2_audit_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/stage2_audit.pid"

  # Stage1 train GPU0-4
  bash "$(dirname "$0")/launch_stage1_5gpu.sh"

  # Parallel: TEST100 collect on GPU5-7 (for sealed TEST gate later)
  mkdir -p "${DATA_DIR}/onpolicy_raw/test"
  for gpu in 5 6 7; do
    shard_idx=$((gpu - 5))
    shard="shard${shard_idx}"
    nohup bash "$(dirname "$0")/run_collect_gpu.sh" "${gpu}" test "${shard}" 3 \
      >> "${LOG_DIR}/collect_test_${shard}_supervisor.log" 2>&1 &
    echo $! > "${PID_DIR}/collect_test_gpu${gpu}.pid"
    r13_log "started test ${shard} on GPU${gpu} pid=$!"
    sleep 15
  done
  touch "${MARKER_DIR}/post_collect_advanced"
}

test_collect_done() {
  local n=0
  for s in 0 1 2 3 4; do
    [[ -f "${DATA_DIR}/onpolicy_raw/test/shard${s}/DONE" ]] && n=$((n + 1))
  done
  [[ "${n}" -ge 5 ]]
}

advance_after_stage1() {
  r13_log "Stage1 train complete — invoking ext continuum (gates + stage2)"
  # Prefer waiting for full TEST100 (5 shards); if not ready, still run VALID gate.
  if test_collect_done; then
    python training/scope_round13/build_operation_sdi.py --with-test \
      >> "${LOG_DIR}/build_operation_sdi_test.log" 2>&1 || true
  fi
  bash "$(dirname "$0")/advance_after_stage1_ext.sh" \
    >> "${LOG_DIR}/advance_after_stage1_ext.log" 2>&1 || true
}

while true; do
  # Detect phase
  if [[ "${PHASE}" == "auto" ]]; then
    if ! collect_all_done; then
      CUR=collect
    elif [[ ! -f "${MARKER_DIR}/post_collect_advanced" ]]; then
      advance_after_collect
      CUR=stage1
    elif ! stage1_all_done; then
      CUR=stage1
    elif [[ ! -f "${MARKER_DIR}/post_stage1_advanced" ]]; then
      advance_after_stage1
      CUR=done
    else
      r13_log "monitor: continuum complete; exiting"
      exit 0
    fi
  else
    CUR="${PHASE}"
  fi

  if [[ "${CUR}" == "collect" ]]; then
    for gpu in 0 1 2 3 4 5; do
      shard="shard${gpu}"
      root="${DATA_DIR}/onpolicy_raw/train/${shard}"
      [[ -f "${root}/DONE" ]] && continue
      if is_stale "${root}/HEARTBEAT"; then
        restart_collect "${gpu}" train "${shard}" 6
      fi
    done
    for gpu in 6 7; do
      s=$((gpu - 6))
      shard="shard${s}"
      root="${DATA_DIR}/onpolicy_raw/valid/${shard}"
      [[ -f "${root}/DONE" ]] && continue
      if is_stale "${root}/HEARTBEAT"; then
        restart_collect "${gpu}" valid "${shard}" 2
      fi
    done
    if collect_all_done && [[ ! -f "${MARKER_DIR}/post_collect_advanced" ]]; then
      advance_after_collect
    fi
  fi

  if [[ "${CUR}" == "stage1" ]]; then
    for gpu in 0 1 2 3 4; do
      variant="${STAGE1_VARIANTS[$gpu]}"
      root="${OUT}/phase_b_stage1/training/${variant}"
      [[ -f "${root}/DONE" ]] && [[ -f "${root}/eval_valid/METRICS.json" ]] && continue
      # FAILED marker or stale heartbeat → restart
      if [[ -f "${root}/FAILED" ]] || is_stale "${root}/HEARTBEAT"; then
        rm -f "${root}/FAILED"
        # Keep merged weights if present; only clear DONE so eval/train can resume.
        rm -f "${root}/DONE"
        restart_stage1 "${gpu}"
      else
        # Dead pid with fresh HB (zombie wrapper) → restart
        pidf="${PID_DIR}/stage1_gpu${gpu}.pid"
        if [[ -f "${pidf}" ]]; then
          pid=$(cat "${pidf}")
          if ! kill -0 "${pid}" 2>/dev/null; then
            # Also check no live run_stage1_gpu for this gpu
            if ! pgrep -f "run_stage1_gpu.sh ${gpu} " >/dev/null 2>&1; then
              r13_log "stage1 gpu${gpu} pid dead; restarting"
              rm -f "${root}/DONE" "${root}/FAILED"
              restart_stage1 "${gpu}"
            fi
          fi
        fi
      fi
    done
    # Watch TEST collect shards 0-4 (manifest n_shards=5)
    for s in 0 1 2 3 4; do
      shard="shard${s}"
      root="${DATA_DIR}/onpolicy_raw/test/${shard}"
      [[ -f "${root}/DONE" ]] && continue
      # Map shard -> preferred GPU: 5,6,7,5,6 cycling free cards when stage1 holds 0-4
      gpu=$((5 + (s % 3)))
      if is_stale "${root}/HEARTBEAT"; then
        restart_collect "${gpu}" test "${shard}" 5
      fi
    done
    if stage1_all_done && [[ ! -f "${MARKER_DIR}/post_stage1_advanced" ]]; then
      # Prefer waiting for test collect; if train finished first, still advance VALID;
      # TEST gate will run when test.jsonl exists.
      advance_after_stage1
    fi
  fi

  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    >> "${LOG_DIR}/gpu_watch.csv" 2>/dev/null || true
  sleep 90
done
