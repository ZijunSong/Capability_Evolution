#!/usr/bin/env bash
# Monitor true-SCAPE evidence_graph 8-GPU Stage L; relaunch dead queues; aggregate on completion.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/true_scape_evidence_graph}"
STAGE_L="${OUT_ROOT}/stage_l"
LOG="${OUT_ROOT}/MONITOR.log"
PID_DIR="${OUT_ROOT}/pids"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
STUCK_MIN="${STUCK_MIN:-45}"
POLL_SEC="${POLL_SEC:-120}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

relaunch_gpu() {
  local g="$1"
  log "RELAUNCH gpu${g} queue"
  local pf="${PID_DIR}/gpu${g}.pid"
  if [[ -f "$pf" ]]; then
    kill "$(cat "$pf")" 2>/dev/null || true
  fi
  pkill -f "CUDA_VISIBLE_DEVICES=${g}.*run_true_scape_stage_l_cell.py" 2>/dev/null || true
  sleep 3
  OUT_ROOT="${OUT_ROOT}" GPU_ONLY="${g}" \
    bash "${SCAPE_ROOT}/scripts/launch_true_scape_evidence_graph_8gpu.sh" >>"$LOG" 2>&1 &
  echo $! >"${pf}"
}

maybe_stage_s() {
  local gate_l="${OUT_ROOT}/AGGREGATE.json"
  [[ -f "$gate_l" ]] || return 0
  if ! "${PYTHON_BIN}" -c "import json; d=json.load(open('$gate_l')); exit(0 if d.get('gate_l',{}).get('pass') else 1)"; then
    return 0
  fi
  if [[ -f "${OUT_ROOT}/stage_s/LAUNCHED" ]]; then
    return 0
  fi
  log "Gate L PASS — launching Stage S closed-loop eval"
  mkdir -p "${OUT_ROOT}/stage_s/logs" "${OUT_ROOT}/stage_s/pids"
  CKPT=$(find "${STAGE_L}/gpu0" -path '*/main_L8K_s42/hf_merged' -type d 2>/dev/null | head -1)
  [[ -z "$CKPT" ]] && CKPT="${MODEL_PATH:-/data/ppnm/models/harness-1}"
  for spec in "2:S2_trained_minus_graph:evidence_graph" "3:S3_trained_full:"; do
    IFS=: read -r gpu name comp <<<"$spec"
    out="${OUT_ROOT}/stage_s/${name}"
    [[ -f "${out}/DONE" ]] && continue
    nohup env GPU="$gpu" JOB_NAME="$name" COMPONENT="$comp" \
      OUT_ROOT="${OUT_ROOT}/stage_s" LIMIT=200 SPLIT=test \
      MODEL_PATH="$CKPT" \
      bash "${SCAPE_ROOT}/scripts/run_loo_worker.sh" \
      >"${OUT_ROOT}/stage_s/logs/${name}.log" 2>&1 &
    echo $! >"${OUT_ROOT}/stage_s/pids/${name}.pid"
  done
  touch "${OUT_ROOT}/stage_s/LAUNCHED"
}

log "monitor start OUT_ROOT=${OUT_ROOT} (180s grace before relaunch)"
sleep 180

while true; do
  done_gpus=0
  running=0
  dead=0
  status_lines=()

  for g in 0 1 2 3 4 5 6 7; do
    pf="${PID_DIR}/gpu${g}.pid"
    if [[ -f "${STAGE_L}/gpu${g}/ALL_DONE" ]]; then
      done_gpus=$((done_gpus + 1))
      status_lines+=("gpu${g}:DONE")
      continue
    fi
    if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
      running=$((running + 1))
      # stuck detection: no new DONE in gpu dir for STUCK_MIN
      latest=0
      while IFS= read -r done_file; do
        ts=$(stat -c %Y "$done_file" 2>/dev/null || echo 0)
        [[ "$ts" -gt "$latest" ]] && latest=$ts
      done < <(find "${STAGE_L}/gpu${g}" -name DONE 2>/dev/null || true)
      now=$(date +%s)
      if [[ "$latest" -eq 0 ]]; then
        # queue still warming up — use pid mtime
        latest=$(stat -c %Y "$pf" 2>/dev/null || echo "$now")
      fi
      age=$(( (now - latest) / 60 ))
      if [[ "$age" -ge "$STUCK_MIN" ]]; then
        log "STUCK gpu${g} age=${age}m — kill+relaunch"
        kill "$(cat "$pf")" 2>/dev/null || true
        pkill -f "CUDA_VISIBLE_DEVICES=${g}.*run_true_scape_stage_l_cell" 2>/dev/null || true
        relaunch_gpu "$g"
        dead=$((dead + 1))
      else
        status_lines+=("gpu${g}:RUN")
      fi
    else
      if [[ ! -f "${STAGE_L}/gpu${g}/ALL_DONE" ]]; then
        log "DEAD gpu${g} — relaunch"
        relaunch_gpu "$g"
        dead=$((dead + 1))
      fi
      status_lines+=("gpu${g}:DEAD")
    fi
  done

  # GPU heartbeat
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >>"${OUT_ROOT}/GPU_HEARTBEAT.csv" 2>/dev/null || true

  {
    echo "# STATUS_LIVE — true SCAPE evidence_graph"
    echo
    echo "- updated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "- done_gpus: ${done_gpus}/8"
    echo "- running: ${running}"
    echo "- relaunched: ${dead}"
    echo "- cells_done: $(find "${STAGE_L}" -name DONE 2>/dev/null | wc -l)"
    echo
    for line in "${status_lines[@]}"; do echo "- ${line}"; done
    echo
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/- /'
  } > "${OUT_ROOT}/STATUS_LIVE.md"

  "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_true_scape_evidence_graph.py" >>"$LOG" 2>&1 || true
  maybe_stage_s

  if [[ "$done_gpus" -ge 8 ]]; then
    log "ALL 8 GPU queues complete"
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_true_scape_evidence_graph.py" | tee -a "$LOG"
    touch "${OUT_ROOT}/ALL_DONE"
    break
  fi

  sleep "${POLL_SEC}"
done

log "monitor exit"
