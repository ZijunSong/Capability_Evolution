#!/usr/bin/env bash
# Monitor weighted Stage L retry; aggregate on completion.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/true_scape_evidence_graph}"
STAGE_L="${OUT_ROOT}/stage_l_retry"
LOG="${OUT_ROOT}/MONITOR_RETRY.log"
PID_DIR="${OUT_ROOT}/pids_retry"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
STUCK_MIN="${STUCK_MIN:-60}"
POLL_SEC="${POLL_SEC:-120}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

relaunch_gpu() {
  local g="$1"
  log "RELAUNCH retry gpu${g}"
  local pf="${PID_DIR}/gpu${g}.pid"
  [[ -f "$pf" ]] && kill "$(cat "$pf")" 2>/dev/null || true
  pkill -f "CUDA_VISIBLE_DEVICES=${g}.*run_true_scape_stage_l_cell.py.*weighted" 2>/dev/null || true
  sleep 3
  OUT_ROOT="${OUT_ROOT}" GPU_ONLY="${g}" \
    bash "${SCAPE_ROOT}/scripts/launch_true_scape_stage_l_retry_8gpu.sh" >>"$LOG" 2>&1 &
  echo $! >"${pf}"
}

log "retry monitor start"
sleep 60

while true; do
  done_gpus=0
  running=0
  for g in 0 1 2 3 4 5 6 7; do
    pf="${PID_DIR}/gpu${g}.pid"
    if [[ -f "${STAGE_L}/gpu${g}/ALL_DONE" ]]; then
      done_gpus=$((done_gpus + 1))
      continue
    fi
    if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
      running=$((running + 1))
      latest=0
      while IFS= read -r done_file; do
        ts=$(stat -c %Y "$done_file" 2>/dev/null || echo 0)
        [[ "$ts" -gt "$latest" ]] && latest=$ts
      done < <(find "${STAGE_L}/gpu${g}" -name DONE 2>/dev/null || true)
      now=$(date +%s)
      [[ "$latest" -eq 0 ]] && latest=$(stat -c %Y "$pf" 2>/dev/null || echo "$now")
      age=$(( (now - latest) / 60 ))
      if [[ "$age" -ge "$STUCK_MIN" ]]; then
        log "STUCK retry gpu${g} age=${age}m"
        kill "$(cat "$pf")" 2>/dev/null || true
        relaunch_gpu "$g"
      fi
    else
      if [[ ! -f "${STAGE_L}/gpu${g}/ALL_DONE" ]]; then
        relaunch_gpu "$g"
      fi
    fi
  done

  {
    echo "# STATUS_LIVE — true SCAPE evidence_graph RETRY"
    echo "- updated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "- done_gpus: ${done_gpus}/8"
    echo "- running: ${running}"
    echo "- cells_done: $(find "${STAGE_L}" -name DONE 2>/dev/null | wc -l)"
  } > "${OUT_ROOT}/STATUS_RETRY.md"

  "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_true_scape_evidence_graph.py" >>"$LOG" 2>&1 || true

  if [[ "$done_gpus" -ge 8 ]]; then
    log "ALL retry GPU queues complete"
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_true_scape_evidence_graph.py" | tee -a "$LOG"
    touch "${OUT_ROOT}/RETRY_ALL_DONE"
    break
  fi
  sleep "${POLL_SEC}"
done

log "retry monitor exit"
