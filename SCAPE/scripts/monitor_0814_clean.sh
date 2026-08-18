#!/usr/bin/env bash
# Monitor 0814 Clean Mechanism: relaunch stuck GPU queues, aggregate, then C2.
set -euo pipefail
trap '' HUP

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/0814_clean_mechanism}"
PID_DIR="${OUT_ROOT}/pids"
LOG_DIR="${OUT_ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
STUCK_SECS="${STUCK_SECS:-1200}"
PHASE="${PHASE:-c0}"

mkdir -p "${PID_DIR}" "${LOG_DIR}"

alive() {
  local pf="$1"
  [[ -f "$pf" ]] || return 1
  local pid
  pid=$(cat "$pf" 2>/dev/null || true)
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

check_done() {
  local gpu="$1"
  if [[ "${PHASE}" == "c0" ]]; then
    [[ -f "${OUT_ROOT}/sft/gpu${gpu}/ALL_DONE" ]]
  else
    [[ -f "${OUT_ROOT}/micro/gpu${gpu}/ALL_DONE" ]]
  fi
}

latest_mtime() {
  local gpu="$1"
  local latest=0 t
  for f in \
    "${LOG_DIR}/c0_gpu${gpu}_queue.log" \
    "${LOG_DIR}/c2_gpu${gpu}_queue.log" \
    "${OUT_ROOT}/sft/gpu${gpu}"/*/STATUS_LIVE.md \
    "${OUT_ROOT}/sft/gpu${gpu}"/*/progress.json \
    "${OUT_ROOT}/evals/gpu${gpu}"/*/summary.json \
    "${OUT_ROOT}/micro/gpu${gpu}"/*/STATUS_LIVE.md
  do
    [[ -e "$f" ]] || continue
    t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    if (( t > latest )); then latest=$t; fi
  done
  echo "${latest}"
}

gpu_util() {
  local gpu="$1"
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | awk '{print $1}'
}

kill_our_gpu_python() {
  local gpu="$1"
  # Only kill python processes we launched (CUDA_VISIBLE_DEVICES=gpu and SCAPE 0814 scripts).
  local pf="${PID_DIR}/${PHASE}_gpu${gpu}.pid"
  if alive "$pf"; then
    local pid
    pid=$(cat "$pf")
    echo "[monitor] kill hung queue pid=${pid} gpu=${gpu}"
    pkill -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${pid}" 2>/dev/null || true
  fi
  # child python still holding the GPU
  pgrep -af "run_clean_|run_true_scape_stage_l_cell" | while read -r line; do
    echo "[monitor] leftover: ${line}"
  done
}

relaunch() {
  local gpu="$1"
  if check_done "${gpu}"; then return 0; fi
  local pf="${PID_DIR}/${PHASE}_gpu${gpu}.pid"
  if alive "$pf"; then
    local now mt util
    now=$(date +%s)
    mt=$(latest_mtime "${gpu}")
    util=$(gpu_util "${gpu}" || echo 0)
    util=${util:-0}
    # Dependency waiters (eval cards) have 0% util by design until SFT DONE.
    if [[ "${PHASE}" == "c0" ]]; then
      case "${gpu}" in
        4|5)
          if [[ ! -f "${OUT_ROOT}/sft/gpu0/full_s42_full/DONE" ]]; then
            return 0
          fi
          ;;
        6)
          if [[ ! -f "${OUT_ROOT}/sft/gpu2/tool_s42_full/DONE" ]]; then
            return 0
          fi
          ;;
      esac
    fi
    if [[ "${mt}" != "0" ]] && (( now - mt > STUCK_SECS )) && (( util < 5 )); then
      echo "[monitor] STUCK gpu${gpu} idle ${util}% log_age=$((now-mt))s — cleanup+relaunch"
      kill_our_gpu_python "${gpu}"
      rm -f "$pf"
    else
      return 0
    fi
  fi
  echo "[monitor] relaunch ${PHASE} gpu${gpu}"
  if [[ "${PHASE}" == "c0" ]]; then
    GPU_ONLY="${gpu}" OUT_ROOT="${OUT_ROOT}" bash "${SCAPE_ROOT}/scripts/launch_0814_clean_c0.sh" &
  else
    GPU_ONLY="${gpu}" OUT_ROOT="${OUT_ROOT}" bash "${SCAPE_ROOT}/scripts/launch_0814_clean_c2.sh" &
  fi
  echo $! >"${pf}"
}

all_done() {
  local g
  for g in 0 1 2 3 4 5 6 7; do
    if ! check_done "${g}"; then return 1; fi
  done
  return 0
}

while true; do
  echo "[$(date -Iseconds)] monitor tick phase=${PHASE}"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
  for g in 0 1 2 3 4 5 6 7; do
    relaunch "${g}"
  done
  "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_0814_clean.py" --out-dir "${OUT_ROOT}" --phase "${PHASE}" \
    >/tmp/scape_0814_agg.out 2>/tmp/scape_0814_agg.err || true
  if all_done; then
    echo "[monitor] ${PHASE} ALL_DONE"
    if [[ "${PHASE}" == "c0" ]]; then
      decision=$("${PYTHON_BIN}" -c "import json; print(json.load(open('${OUT_ROOT}/NEXT_DECISION.json')).get('NEXT_DECISION',''))")
      echo "[monitor] decision=${decision}"
      if [[ "${decision}" == "WAIT_FOR_VALUE_POSITIVE_TARGET" || "${decision}" == "CLEAN_BASE_BLOCKED" || "${decision}" == "CLEAN_MECHANISM_FAIL" ]]; then
        break
      fi
      # C1 gap is included in C0 evals; start C2 micro if base passed and gap exists.
      enter=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${OUT_ROOT}/DECISION_STATE.json')); print(d.get('gap',{}).get('enter_train', False) and d.get('gate',{}).get('any_pass', False))")
      if [[ "${enter}" == "True" ]]; then
        echo "[monitor] starting Phase C2 micro"
        PHASE=c2
        # reset pid files for c2
        continue
      fi
      echo "[monitor] C0 complete; C2 not entered (gap/gate)"
      break
    else
      break
    fi
  fi
  sleep 90
done

"${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_0814_clean.py" --out-dir "${OUT_ROOT}" --phase "${PHASE}" || true
echo "[monitor] exit phase=${PHASE}"
