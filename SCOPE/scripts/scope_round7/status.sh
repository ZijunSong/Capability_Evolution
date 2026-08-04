#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

echo "=== Round 7 Status $(date -Is) ==="
echo "Branch: $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo ""
echo "=== Queue PIDs ==="
for G in 0 1 2 3 4 5 6 7; do
  PF="${PID_DIR}/gpu${G}.pid"
  if [[ -f "${PF}" ]]; then
    PID=$(cat "${PF}")
    if kill -0 "${PID}" 2>/dev/null; then
      echo "GPU${G}: RUNNING pid=${PID}"
    else
      echo "GPU${G}: EXITED pid=${PID}"
    fi
  else
    echo "GPU${G}: no pid file"
  fi
done
echo ""
echo "=== Markers ==="
ls -la "${MARKER_DIR}/" 2>/dev/null || echo "(none)"
echo ""
echo "=== Live trace progress (rerun) ==="
for d in "${OUT}/contract_trace/live_rerun"/*/; do
  [[ -d "$d" ]] || continue
  EP=$(wc -l < "${d}/episodes.jsonl" 2>/dev/null || echo 0)
  TR=$(wc -l < "${d}/live_dup_decision_trace.jsonl" 2>/dev/null || echo 0)
  GATE=$(python3 -c "import json; print(json.load(open('${d}/contract_gate.json'))['contract_gate_pass'])" 2>/dev/null || echo "-")
  echo "$(basename "$d"): episodes=${EP}/25 trace=${TR} gate=${GATE}"
done
echo ""
echo "=== Holdout progress (tau0 rerun) ==="
for d in "${OUT}/holdout_tau0_rerun"/*/; do
  [[ -d "$d" ]] || continue
  EP=$(wc -l < "${d}/episodes.jsonl" 2>/dev/null || echo 0)
  echo "$(basename "$d"): episodes=${EP}/25"
done
echo ""
echo "=== Live trace progress (archived) ==="
for d in "${OUT}/contract_trace/live"/*/; do
  [[ -d "$d" ]] || continue
  EP=$(wc -l < "${d}/episodes.jsonl" 2>/dev/null || echo 0)
  TR=$(wc -l < "${d}/live_dup_decision_trace.jsonl" 2>/dev/null || echo 0)
  echo "$(basename "$d"): episodes=${EP}/25 trace=${TR}"
done
echo ""
echo "=== Recent logs (last 3 lines each) ==="
for G in 0 1 2 3 4 5 6 7; do
  LF="${LOG_DIR}/gpu${G}.log"
  if [[ -f "${LF}" ]]; then
    echo "--- gpu${G}.log ---"
    tail -3 "${LF}" 2>/dev/null || true
  fi
done
