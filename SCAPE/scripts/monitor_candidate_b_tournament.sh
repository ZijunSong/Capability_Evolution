#!/usr/bin/env bash
# Monitor Candidate-B tournament: micro → aggregate → 8K → aggregate.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/true_scape_candidate_b_tournament}"
STAGE_MICRO="${OUT_ROOT}/stage_l_micro"
STAGE_8K="${OUT_ROOT}/stage_l_8k"
LOG="${OUT_ROOT}/MONITOR.log"
PID_DIR_MICRO="${OUT_ROOT}/pids_micro"
PID_DIR_8K="${OUT_ROOT}/pids_8k"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
STUCK_MIN="${STUCK_MIN:-90}"
POLL_SEC="${POLL_SEC:-120}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

relaunch_micro_gpu() {
  local g="$1"
  log "RELAUNCH micro gpu${g}"
  local pf="${PID_DIR_MICRO}/gpu${g}.pid"
  [[ -f "$pf" ]] && kill "$(cat "$pf")" 2>/dev/null || true
  pkill -f "CUDA_VISIBLE_DEVICES=${g}.*run_true_scape_stage_l_cell.py.*stage_l_micro" 2>/dev/null || true
  sleep 3
  OUT_ROOT="${OUT_ROOT}" GPU_ONLY="${g}" \
    bash "${SCAPE_ROOT}/scripts/launch_candidate_b_tournament_micro_8gpu.sh" >>"$LOG" 2>&1 &
  echo $! >"${pf}"
}

relaunch_8k_gpu() {
  local g="$1"
  log "RELAUNCH 8k gpu${g}"
  local pf="${PID_DIR_8K}/gpu${g}.pid"
  [[ -f "$pf" ]] && kill "$(cat "$pf")" 2>/dev/null || true
  pkill -f "CUDA_VISIBLE_DEVICES=${g}.*run_true_scape_stage_l_cell.py.*stage_l_8k" 2>/dev/null || true
  sleep 3
  OUT_ROOT="${OUT_ROOT}" GPU_ONLY="${g}" \
    bash "${SCAPE_ROOT}/scripts/launch_candidate_b_winner_8k_8gpu.sh" >>"$LOG" 2>&1 &
  echo $! >"${pf}"
}

monitor_stage() {
  local stage_dir="$1" pid_dir="$2" relaunch_fn="$3" label="$4"
  local done_gpus=0 running=0
  for g in 0 1 2 3 4 5 6 7; do
    if [[ -f "${stage_dir}/gpu${g}/ALL_DONE" ]]; then
      done_gpus=$((done_gpus + 1))
      continue
    fi
    local pf="${pid_dir}/gpu${g}.pid"
    if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
      running=$((running + 1))
      local latest=0
      while IFS= read -r done_file; do
        ts=$(stat -c %Y "$done_file" 2>/dev/null || echo 0)
        [[ "$ts" -gt "$latest" ]] && latest=$ts
      done < <(find "${stage_dir}/gpu${g}" -name DONE 2>/dev/null || true)
      local now=$(date +%s)
      [[ "$latest" -eq 0 ]] && latest=$(stat -c %Y "$pf" 2>/dev/null || echo "$now")
      local age=$(( (now - latest) / 60 ))
      if [[ "$age" -ge "$STUCK_MIN" ]]; then
        log "STUCK ${label} gpu${g} age=${age}m"
        kill "$(cat "$pf")" 2>/dev/null || true
        "$relaunch_fn" "$g"
      fi
    else
      if [[ ! -f "${stage_dir}/gpu${g}/ALL_DONE" ]]; then
        "$relaunch_fn" "$g"
      fi
    fi
  done
  {
    echo "# STATUS — ${label}"
    echo "- updated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "- done_gpus: ${done_gpus}/8"
    echo "- running: ${running}"
    echo "- cells_done: $(find "${stage_dir}" -name DONE 2>/dev/null | wc -l)"
  } > "${OUT_ROOT}/STATUS_${label}.md"
  echo "$done_gpus"
}

log "monitor start"
mkdir -p "${OUT_ROOT}"

# Phase 1: micro
if [[ ! -f "${OUT_ROOT}/MICRO_ALL_DONE" ]]; then
  if [[ ! -f "${OUT_ROOT}/MICRO_LAUNCHED" ]]; then
    bash "${SCAPE_ROOT}/scripts/launch_candidate_b_tournament_micro_8gpu.sh" >>"$LOG" 2>&1 &
    touch "${OUT_ROOT}/MICRO_LAUNCHED"
    log "micro launch triggered"
    sleep 30
  fi
  while true; do
  done_gpus=$(monitor_stage "$STAGE_MICRO" "$PID_DIR_MICRO" relaunch_micro_gpu "MICRO" || true)
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_candidate_b_tournament.py" >>"$LOG" 2>&1 || true
    if [[ "${done_gpus:-0}" -ge 8 ]]; then
      log "micro ALL_DONE"
      touch "${OUT_ROOT}/MICRO_ALL_DONE"
      break
    fi
    sleep "${POLL_SEC}"
  done
fi

# Phase 2: aggregate + maybe 8K
"${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_candidate_b_tournament.py" | tee -a "$LOG"

winner="$("${PYTHON_BIN}" -c "
import json
from pathlib import Path
p=Path('${OUT_ROOT}/CANDIDATE_B_FINAL.json')
if p.exists():
  d=json.loads(p.read_text())
  w=d.get('winner_component_id')
  passed=bool(d.get('micro_pass'))
  print(w or '', passed)
else:
  print('', False)
")"
read -r WINNER_ID WINNER_PASS <<< "$winner"

if [[ "${WINNER_PASS}" == "True" && -n "${WINNER_ID}" && ! -f "${OUT_ROOT}/EIGHT_K_ALL_DONE" ]]; then
  log "launching 8K for winner=${WINNER_ID}"
  if [[ ! -f "${OUT_ROOT}/EIGHT_K_LAUNCHED" ]]; then
    WINNER="${WINNER_ID}" bash "${SCAPE_ROOT}/scripts/launch_candidate_b_winner_8k_8gpu.sh" >>"$LOG" 2>&1 &
    touch "${OUT_ROOT}/EIGHT_K_LAUNCHED"
    sleep 30
  fi
  while true; do
    done_gpus=$(monitor_stage "$STAGE_8K" "$PID_DIR_8K" relaunch_8k_gpu "8K" || true)
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_candidate_b_tournament.py" >>"$LOG" 2>&1 || true
    if [[ "${done_gpus:-0}" -ge 8 ]]; then
      log "8K ALL_DONE"
      touch "${OUT_ROOT}/EIGHT_K_ALL_DONE"
      break
    fi
    sleep "${POLL_SEC}"
  done
else
  log "skip 8K: pass=${WINNER_PASS} winner=${WINNER_ID}"
fi

"${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_candidate_b_tournament.py" | tee -a "$LOG"
touch "${OUT_ROOT}/TOURNAMENT_ALL_DONE"
log "tournament complete"
