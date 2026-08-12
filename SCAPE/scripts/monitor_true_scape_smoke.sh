#!/usr/bin/env bash
# Monitor true-SCAPE smoke groups; on failure clean GPU procs for that group and relaunch once.
set -euo pipefail

OUT_ROOT="${1:?out root}"
LOG_DIR="${2:?log dir}"
PYTHON_BIN="${3:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-21600}"
POLL_SEC="${POLL_SEC:-30}"
RELAUNCHED_A=0
RELAUNCHED_B=0

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

log() { echo "[$(date '+%F %T')] $*"| tee -a "${LOG_DIR}/monitor.log"; }

group_failed() {
  local g="$1"
  [[ -f "${OUT_ROOT}/group_${g}/FAILED.json" ]] && return 0
  if [[ -f "${LOG_DIR}/group_${g}.log" ]] && grep -qE 'EXIT:[1-9]|Traceback|CUDA out of memory|RuntimeError' "${LOG_DIR}/group_${g}.log"; then
    if [[ ! -f "${OUT_ROOT}/group_${g}/DONE" ]]; then
      return 0
    fi
  fi
  return 1
}

cleanup_group_gpus() {
  local gpus="$1"
  log "cleanup GPUs ${gpus}"
  # Kill python jobs bound to those visible devices via pids file if present
  for pidfile in "${OUT_ROOT}/pids"/group_*.pid; do
    [[ -f "$pidfile" ]] || continue
    local pid
    pid=$(cat "$pidfile" || true)
    if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
      # only kill if still running and screen died / failed
      true
    fi
  done
  # Best-effort: fuser processes on those /dev/nvidia*
  for gi in ${gpus//,/ }; do
    fuser -k "/dev/nvidia${gi}" 2>/dev/null || true
  done
  sleep 5
}

relaunch_a() {
  log "RELAUNCH Group A"
  screen -S scape_smoke_A -X quit 2>/dev/null || true
  cleanup_group_gpus "0,1,2,3"
  rm -f "${OUT_ROOT}/group_a/FAILED.json" "${OUT_ROOT}/group_a/DONE"
  screen -dmS scape_smoke_A bash -c "
    source /data/ppnm/miniconda3/etc/profile.d/conda.sh
    conda activate bishop
    export PYTHONPATH='${SCAPE_ROOT}'
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd '${SCAPE_ROOT}'
    '${PYTHON_BIN}' scripts/run_true_scape_pipeline_smoke.py \
      --group A --out '${OUT_ROOT}/group_a' --model-path '${MODEL_PATH}' \
      --component-id evidence_graph --epochs 1 \
      2>&1 | tee -a '${LOG_DIR}/group_a.log'
  "
}

relaunch_b() {
  log "RELAUNCH Group B"
  screen -S scape_smoke_B -X quit 2>/dev/null || true
  cleanup_group_gpus "4,5,6,7"
  rm -f "${OUT_ROOT}/group_b/FAILED.json" "${OUT_ROOT}/group_b/DONE"
  screen -dmS scape_smoke_B bash -c "
    source /data/ppnm/miniconda3/etc/profile.d/conda.sh
    conda activate bishop
    export PYTHONPATH='${SCAPE_ROOT}'
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd '${SCAPE_ROOT}'
    '${PYTHON_BIN}' scripts/run_true_scape_pipeline_smoke.py \
      --group B --out '${OUT_ROOT}/group_b' --model-path '${MODEL_PATH}' \
      --component-id evidence_graph --epochs 1 \
      2>&1 | tee -a '${LOG_DIR}/group_b.log'
  "
}

START=$(date +%s)
log "monitor start OUT_ROOT=${OUT_ROOT}"

while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  A_DONE=0; B_DONE=0
  [[ -f "${OUT_ROOT}/group_a/DONE" ]] && A_DONE=1
  [[ -f "${OUT_ROOT}/group_b/DONE" ]] && B_DONE=1

  # GPU heartbeat
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    > "${OUT_ROOT}/GPU_HEARTBEAT.csv" 2>/dev/null || true

  if group_failed a && [[ $RELAUNCHED_A -eq 0 ]]; then
    RELAUNCHED_A=1
    relaunch_a
  fi
  if group_failed b && [[ $RELAUNCHED_B -eq 0 ]]; then
    RELAUNCHED_B=1
    relaunch_b
  fi

  if [[ $A_DONE -eq 1 && $B_DONE -eq 1 ]]; then
    log "both groups DONE — aggregating audits"
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_true_scape_pipeline_smoke.py" \
      --group aggregate --out "${OUT_ROOT}"
    touch "${OUT_ROOT}/ALL_DONE"
    log "ALL_DONE"
    exit 0
  fi

  if [[ $ELAPSED -gt $MAX_WAIT_SEC ]]; then
    log "TIMEOUT after ${ELAPSED}s a_done=${A_DONE} b_done=${B_DONE}"
    echo "timeout" > "${OUT_ROOT}/MONITOR_TIMEOUT"
    exit 1
  fi

  log "wait a_done=${A_DONE} b_done=${B_DONE} elapsed=${ELAPSED}s"
  sleep "${POLL_SEC}"
done
