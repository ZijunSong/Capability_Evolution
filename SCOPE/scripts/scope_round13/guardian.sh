#!/usr/bin/env bash
# Round13 guardian: detect stuck/failed jobs, restart, keep GPUs busy, advance continuum.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

STALE_SEC="${R13_STALE_SEC:-1800}"
r13_log "guardian start stale=${STALE_SEC}s"

is_stale() {
  local hb="$1"
  [[ -f "${hb}" ]] || return 0
  local hb_ts now age
  hb_ts=$(date -d "$(cat "${hb}" | head -1)" +%s 2>/dev/null || echo 0)
  now=$(date +%s)
  age=$((now - hb_ts))
  [[ "${age}" -gt "${STALE_SEC}" ]]
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 2000 {print $1}'
}

gpu_claimed() {
  local gpu="$1"
  pgrep -af "run_stage1_gpu.sh ${gpu} |run_stage2_gpu.sh ${gpu} |run_collect_gpu.sh ${gpu} " >/dev/null 2>&1
}

launch_stage2_on_free() {
  local gate="${OUT}/stage2_targeted/DATASET_GATE.json"
  [[ -f "${gate}" ]] || return 0
  # External restart lock
  [[ -f "${PID_DIR}/stage2_launch.lock" ]] && return 0
  local pass
  pass=$(python -c "import json;print(json.load(open('${gate}')).get('NONDEGENERATE_STAGE2_DATA_PASS', False))")
  [[ "${pass}" == "True" ]] || return 0
  # Only one new Stage2 launch per guardian tick to avoid double-booking.
  local -a variants=(r13_ckpt_pointer_seed42 r13_ckpt_pointer_seed43 r13_ckpt_pointer_seed44)
  for v in "${variants[@]}"; do
    local root="${OUT}/stage2_targeted/training/${v}"
    [[ -f "${root}/DONE" ]] && continue
    if ps -eo args | grep -F "run_stage2_pointer_train.py --variant ${v}" | grep -v grep >/dev/null; then
      continue
    fi
    if ps -eo args | grep -E "run_stage2_gpu.sh [0-9]+ ${v}" | grep -v grep >/dev/null; then
      continue
    fi
    local gpu=""
    for g in $(free_gpus); do
      if gpu_claimed "${g}"; then
        continue
      fi
      gpu=$g
      break
    done
    [[ -n "${gpu}" ]] || return 0
    r13_log "guardian launch stage2 ${v} on GPU${gpu}"
    nohup bash "$(dirname "$0")/run_stage2_gpu.sh" "${gpu}" "${v}" \
      >> "${LOG_DIR}/stage2_${v}_supervisor.log" 2>&1 &
    echo $! > "${PID_DIR}/stage2_${v}.pid"
    sleep 30
    return 0
  done
}

while true; do
  # Stage1 FAILED / dead
  for gpu in 0 1 2 3 4; do
    v="${STAGE1_VARIANTS[$gpu]}"
    root="${OUT}/phase_b_stage1/training/${v}"
    if [[ -f "${root}/DONE" ]] && [[ -f "${root}/eval_valid/METRICS.json" ]]; then
      continue
    fi
    if [[ -f "${root}/FAILED" ]] || is_stale "${root}/HEARTBEAT"; then
      r13_log "guardian restart stage1 ${v} on GPU${gpu}"
      pkill -f "run_stage1_gpu.sh ${gpu} " 2>/dev/null || true
      pkill -f "run_stage1_train.py --variant ${v}" 2>/dev/null || true
      sleep 2
      rm -f "${root}/FAILED" "${root}/DONE"
      nohup bash "$(dirname "$0")/run_stage1_gpu.sh" "${gpu}" "${v}" \
        >> "${LOG_DIR}/stage1_${v}_supervisor.log" 2>&1 &
      echo $! > "${PID_DIR}/stage1_gpu${gpu}.pid"
      sleep 5
    fi
  done

  # TEST shards 3/4
  for s in 3 4; do
    root="${DATA_DIR}/onpolicy_raw/test/shard${s}"
    [[ -f "${root}/DONE" ]] && continue
    if is_stale "${root}/HEARTBEAT"; then
      gpu=$((5 + (s - 3)))
      r13_log "guardian restart test shard${s} on GPU${gpu}"
      pkill -f "run_collect_gpu.sh ${gpu} test" 2>/dev/null || true
      pkill -f "vllm.*$((18700+gpu))" 2>/dev/null || true
      sleep 2
      rm -f "${root}/DONE"
      nohup bash "$(dirname "$0")/run_collect_gpu.sh" "${gpu}" test "shard${s}" 5 \
        >> "${LOG_DIR}/collect_test_shard${s}_supervisor.log" 2>&1 &
      echo $! > "${PID_DIR}/collect_test_gpu${gpu}.pid"
    fi
  done

  # Stage2 FAILED / dead / schedule on free GPUs
  for v in r13_ckpt_pointer_seed42 r13_ckpt_pointer_seed43 r13_ckpt_pointer_seed44; do
    root="${OUT}/stage2_targeted/training/${v}"
    [[ -f "${root}/DONE" ]] && continue
    # If trainer python is alive, refresh HB and do not kill.
    if ps -eo args | grep -F "run_stage2_pointer_train.py --variant ${v}" | grep -v grep >/dev/null; then
      date -Is > "${root}/HEARTBEAT"
      rm -f "${root}/FAILED"
      continue
    fi
    if [[ -f "${root}/FAILED" ]] || is_stale "${root}/HEARTBEAT"; then
      r13_log "guardian mark stage2 ${v} for relaunch (trainer dead)"
      rm -f "${root}/FAILED"
    fi
  done
  launch_stage2_on_free

  # Continuum advance
  n_done=0
  for v in "${STAGE1_VARIANTS[@]}"; do
    root="${OUT}/phase_b_stage1/training/${v}"
    if [[ -f "${root}/DONE" ]] && [[ -f "${root}/eval_valid/METRICS.json" ]]; then
      n_done=$((n_done + 1))
    fi
  done
  if [[ "${n_done}" -ge 5 ]] && [[ ! -f "${MARKER_DIR}/post_stage1_advanced" ]]; then
    r13_log "guardian: all stage1 done → advance"
    bash "$(dirname "$0")/advance_after_stage1_ext.sh" \
      >> "${LOG_DIR}/advance_after_stage1_ext.log" 2>&1 || true
  fi

  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    >> "${LOG_DIR}/gpu_watch.csv" 2>/dev/null || true
  sleep 120
done
