#!/usr/bin/env bash
# Monitor H20 clean-init AUTO OPD: stuck cleanup, relaunch, phase advance.
set -euo pipefail
trap '' HUP

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/h20_clean_auto_0817}"
PID_DIR="${OUT_ROOT}/pids"
LOG_DIR="${OUT_ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
STUCK_SECS="${STUCK_SECS:-900}"
LAUNCH="${SCAPE_ROOT}/scripts/launch_h20_clean_auto_0817.sh"

mkdir -p "${PID_DIR}" "${LOG_DIR}" "${OUT_ROOT}"

alive() {
  local pf="$1"
  [[ -f "$pf" ]] || return 1
  local pid
  pid=$(cat "$pf" 2>/dev/null || true)
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk '{print $1}'
}

latest_mtime() {
  local gpu="$1" phase="$2"
  local latest=0 t
  for f in \
    "${LOG_DIR}/${phase}_gpu${gpu}_queue.log" \
    "${OUT_ROOT}/phase_${phase}/gpu${gpu}"/progress.json \
    "${OUT_ROOT}/phase_${phase}"/*/progress.json \
    "${OUT_ROOT}/phase_A/gpu${gpu}"/*/progress.json \
    "${OUT_ROOT}/phase_B"/gpu"${gpu}"/*/progress.json \
    "${OUT_ROOT}/phase_C/gpu${gpu}"/progress.json \
    "${OUT_ROOT}/phase_D"/*/progress.json \
    "${OUT_ROOT}/phase_E"/*/progress.json \
    "${OUT_ROOT}/phase_G"/*/progress.json
  do
    [[ -e "$f" ]] || continue
    t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    if (( t > latest )); then latest=$t; fi
  done
  echo "${latest}"
}

our_python_pids() {
  # Only this round's scripts.
  pgrep -af "run_clean_base_eval128|run_format_repair_cell|collect_auto_clean_states|run_auto_clean_value|run_auto_clean_real_eval|run_true_scape_stage_l_cell|run_harmony_runtime_audit|build_h20_query_manifests" \
    | awk '{print $1}' || true
}

kill_queue() {
  local gpu="$1" phase="$2"
  local pf="${PID_DIR}/${phase}_gpu${gpu}.pid"
  if alive "$pf"; then
    local pid
    pid=$(cat "$pf")
    echo "[monitor] kill queue pid=${pid} gpu=${gpu} phase=${phase}"
    pkill -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

phase_gpu_done() {
  local gpu="$1" phase="$2"
  [[ -f "${OUT_ROOT}/phase_${phase}/gpu${gpu}/ALL_DONE" ]]
}

all_gpus_done() {
  local phase="$1"
  local g
  for g in 0 1 2 3 4 5 6 7; do
    if [[ "${phase}" == "B" && ( "${g}" -ge 4 ) ]]; then
      continue
    fi
    if ! phase_gpu_done "${g}" "${phase}"; then
      return 1
    fi
  done
  return 0
}

relaunch() {
  local gpu="$1" phase="$2"
  if [[ "${phase}" == "STOP" || "${phase}" == "DONE" ]]; then
    return 0
  fi
  if phase_gpu_done "${gpu}" "${phase}"; then
    return 0
  fi
  local pf="${PID_DIR}/${phase}_gpu${gpu}.pid"
  if alive "$pf"; then
    local now mt util
    now=$(date +%s)
    mt=$(latest_mtime "${gpu}" "${phase}")
    util=$(gpu_util "${gpu}" || echo 0)
    util=${util:-0}
    # CPU jobs on 6/7 in phase A may be 0% GPU by design
    if [[ "${phase}" == "A" && ( "${gpu}" == "6" || "${gpu}" == "7" ) ]]; then
      return 0
    fi
    if [[ "${mt}" != "0" ]] && (( now - mt > STUCK_SECS )) && (( util < 5 )); then
      echo "[monitor] STUCK gpu${gpu} util=${util}% age=$((now-mt))s — cleanup+relaunch"
      kill_queue "${gpu}" "${phase}"
      rm -f "$pf"
    else
      return 0
    fi
  fi
  echo "[monitor] relaunch phase=${phase} gpu${gpu}"
  GPU_ONLY="${gpu}" OUT_ROOT="${OUT_ROOT}" bash "${LAUNCH}" &
  echo $! >"${pf}"
}

echo "[$(date -Iseconds)] monitor start pid=$$"
echo $$ > "${PID_DIR}/monitor.pid"

while true; do
  phase="A"
  if [[ -f "${OUT_ROOT}/PHASE" ]]; then
    phase="$(tr -d '[:space:]' < "${OUT_ROOT}/PHASE")"
  fi
  echo "[$(date -Iseconds)] monitor tick phase=${phase}"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true

  if [[ "${phase}" == "STOP" || "${phase}" == "DONE" ]]; then
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_h20_clean_auto_0817.py" --out "${OUT_ROOT}" \
      >>"${LOG_DIR}/aggregate.log" 2>&1 || true
    echo "[monitor] terminal phase=${phase}"
    break
  fi

  for g in 0 1 2 3 4 5 6 7; do
    relaunch "${g}" "${phase}"
  done

  if all_gpus_done "${phase}"; then
    echo "[monitor] phase ${phase} all GPU queues done — aggregate"
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_h20_clean_auto_0817.py" --out "${OUT_ROOT}" \
      >>"${LOG_DIR}/aggregate.log" 2>&1 || true
    newphase="$(tr -d '[:space:]' < "${OUT_ROOT}/PHASE")"
    if [[ "${newphase}" != "${phase}" ]]; then
      echo "[monitor] phase advance ${phase} -> ${newphase}"
      # drop old pid files so relaunch starts new queues
      for g in 0 1 2 3 4 5 6 7; do
        rm -f "${PID_DIR}/${phase}_gpu${g}.pid"
      done
    else
      echo "[monitor] aggregate kept phase=${phase}; sleep"
    fi
  fi
  sleep 90
done
