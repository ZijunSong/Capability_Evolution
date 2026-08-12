#!/usr/bin/env bash
# One-shot Round14 health check + stuck recovery for wave0/rollback
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

STUCK_MIN="${R14_STUCK_MIN:-45}"
now=$(date +%s)

check_rollout_progress() {
  local name="$1" out="$2" gpu="$3" pidfile="$4"
  local ep="${out}/episodes.jsonl"
  local done="${out}/DONE"
  [[ -f "${done}" ]] && { echo "OK ${name} DONE"; return 0; }
  local n=0
  [[ -f "${ep}" ]] && n=$(wc -l < "${ep}" | tr -d ' ')
  local mtime=0
  if [[ -f "${ep}" ]]; then
    mtime=$(stat -c %Y "${ep}")
  elif [[ -d "${out}" ]]; then
    mtime=$(stat -c %Y "${out}")
  fi
  local age_min=$(( (now - mtime) / 60 ))
  local alive=0
  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    alive=1
  fi
  echo "CHK ${name} ep=${n} age_min=${age_min} alive=${alive} gpu=${gpu}"
  if [[ "${alive}" -eq 0 ]]; then
    r14_log "watchdog: ${name} dead — restart gpu${gpu}"
    return 2
  fi
  if [[ "${n}" -eq 0 && "${age_min}" -gt "${STUCK_MIN}" ]]; then
    r14_log "watchdog: ${name} stuck 0 episodes ${age_min}m — kill+restart"
    return 2
  fi
  # Heartbeat touch for guardian
  r14_touch_hb "${OUT}/gpu0_dup_anchor/HEARTBEAT"
  return 0
}

restart_wave0_cond() {
  local gpu="$1" cond="$2" seed="$3"
  local out="${OUT}/gpu0_dup_anchor/${cond}"
  [[ "${cond}" == "T_OFF" ]] && out="${OUT}/gpu0_dup_anchor/T_OFF_seed${seed}"
  r14_kill_vllm_on_gpu "${gpu}"
  sleep 2
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python training/scope_round14/run_module_retirement_eval.py \
    --capability duplicate_evidence \
    --manifest "${R14_FRESH100}" \
    --output-dir "${out}" \
    --gpu "${gpu}" --seed "${seed}" --conditions "${cond}" \
    --temperature 0.0 --parallel "${R14_PARALLEL:-16}" \
    --flat-output --resume --run-closed-loop \
    >> "${LOG_DIR}/wave0_dup_calibrate.log" 2>&1 &
  echo $! > "${PID_DIR}/wave0_gpu${gpu}.pid"
}

# Wave0 conditions
check_rollout_progress B_OFF "${OUT}/gpu0_dup_anchor/B_OFF" 0 "${PID_DIR}/wave0_gpu0.pid" || \
  restart_wave0_cond 0 B_OFF 42
check_rollout_progress B_ON "${OUT}/gpu0_dup_anchor/B_ON" 1 "${PID_DIR}/wave0_gpu1.pid" || \
  restart_wave0_cond 1 B_ON 42
check_rollout_progress T42 "${OUT}/gpu0_dup_anchor/T_OFF_seed42" 2 "${PID_DIR}/wave0_gpu2.pid" || \
  restart_wave0_cond 2 T_OFF 42
check_rollout_progress T43 "${OUT}/gpu0_dup_anchor/T_OFF_seed43" 3 "${PID_DIR}/wave0_gpu3.pid" || \
  restart_wave0_cond 3 T_OFF 43
check_rollout_progress T44 "${OUT}/gpu0_dup_anchor/T_OFF_seed44" 4 "${PID_DIR}/wave0_gpu4.pid" || \
  restart_wave0_cond 4 T_OFF 44

# GPU6/5 train heartbeats
for d in "${OUT}/gpu6_rollback_lite" "${OUT}/gpu5_external_verify"; do
  if [[ ! -f "${d}/DONE" ]]; then
    r14_touch_hb "${d}/HEARTBEAT"
  fi
done

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
  | tee -a "${LOG_DIR}/gpu_watch.csv" >/dev/null

# If wave0 complete and continuum waiting, note it
if r14_wave0_complete; then
  echo "WAVE0_COMPLETE"
  if [[ ! -f "${OUT}/gpu0_dup_anchor/DUP_RETIREMENT_GATE.json" ]]; then
    python training/scope_round14/aggregate_dup_anchor.py --anchor-dir "${OUT}/gpu0_dup_anchor" || true
  fi
fi
