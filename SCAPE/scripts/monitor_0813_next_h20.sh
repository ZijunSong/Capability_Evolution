#!/usr/bin/env bash
# Monitor 0813_next_h20: relaunch stuck GPU queues, aggregate when done.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/0813_next_h20}"
PID_DIR="${OUT_ROOT}/pids"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
PHASE="${PHASE:-all}"

check_done() {
  local phase="$1" gpu="$2"
  if [[ "${phase}" == "phase_a" ]]; then
    [[ -f "${OUT_ROOT}/phase_a/gpu${gpu}/ALL_DONE" ]]
  elif [[ "${phase}" == "graph_hybrid" ]]; then
    [[ -f "${OUT_ROOT}/graph_hybrid/micro/gpu${gpu}/ALL_DONE" ]]
  else
    return 1
  fi
}

relaunch() {
  local phase="$1"
  for g in 0 1 2 3 4 5 6 7; do
    if check_done "${phase}" "${g}"; then continue; fi
    local pf="${PID_DIR}/${phase}_gpu${g}.pid"
    if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then continue; fi
    echo "[monitor] relaunch ${phase} gpu${g}"
    if [[ "${phase}" == "phase_a" ]]; then
      GPU_ONLY="${g}" OUT_ROOT="${OUT_ROOT}" bash "${SCAPE_ROOT}/scripts/launch_0813_next_h20_phase_a.sh" &
      echo $! >"${pf}"
    else
      GPU_ONLY="${g}" OUT_ROOT="${OUT_ROOT}" bash "${SCAPE_ROOT}/scripts/launch_0813_next_h20_graph_hybrid.sh" &
      echo $! >"${pf}"
    fi
  done
}

all_done() {
  local phase="$1"
  for g in 0 1 2 3 4 5 6 7; do
    if ! check_done "${phase}" "${g}"; then return 1; fi
  done
  return 0
}

cleanup_stale() {
  for pf in "${PID_DIR}"/*.pid; do
    [[ -f "$pf" ]] || continue
    pid=$(cat "$pf")
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pf"
    fi
  done
}

while true; do
  echo "[$(date -Iseconds)] monitor tick phase=${PHASE}"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
  cleanup_stale

  if [[ "${PHASE}" == "all" || "${PHASE}" == "phase_a" ]]; then
    relaunch phase_a
    if all_done phase_a; then
      echo "[monitor] Phase A complete — aggregating"
      "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_0813_next_h20.py" --out-dir "${OUT_ROOT}"
      if [[ "${PHASE}" == "phase_a" ]]; then break; fi
    fi
  fi

  if [[ "${PHASE}" == "all" || "${PHASE}" == "graph_hybrid" ]]; then
    if [[ -f "${OUT_ROOT}/LEARNABILITY_GATE_V2.json" ]]; then
      case=$(python3 -c "import json; print(json.load(open('${OUT_ROOT}/LEARNABILITY_GATE_V2.json'))['phase_a_decision']['case'])")
      if [[ "${case}" != "EXISTING_CHECKPOINT_STAGE_S" && "${case}" != "BLOCKED_BY_METRIC_BUG" ]]; then
        relaunch graph_hybrid
        if all_done graph_hybrid; then
          echo "[monitor] Graph-Hybrid micro complete"
          if [[ "${PHASE}" == "graph_hybrid" ]]; then break; fi
        fi
      fi
    fi
  fi

  if [[ "${PHASE}" == "all" ]] && all_done phase_a && all_done graph_hybrid; then
    echo "[monitor] ALL phases done"
    break
  fi

  sleep 120
done
