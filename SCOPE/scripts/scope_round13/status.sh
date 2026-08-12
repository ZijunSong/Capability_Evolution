#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo "=== Collect progress ==="
for split in train valid test; do
  root="${DATA_DIR}/onpolicy_raw/${split}"
  [[ -d "${root}" ]] || continue
  for d in "${root}"/shard*; do
    [[ -d "$d" ]] || continue
    ev=0; ep=0; donef="-"
    [[ -f "$d/rollback_events.jsonl" ]] && ev=$(wc -l < "$d/rollback_events.jsonl")
    [[ -f "$d/episodes.jsonl" ]] && ep=$(wc -l < "$d/episodes.jsonl")
    [[ -f "$d/DONE" ]] && donef="DONE"
    hb="-"; [[ -f "$d/HEARTBEAT" ]] && hb=$(cat "$d/HEARTBEAT" | head -1)
    echo "  ${split}/$(basename "$d"): events=${ev} episodes=${ep} ${donef} hb=${hb}"
  done
done

echo "=== Stage1 ==="
for v in "${STAGE1_VARIANTS[@]}"; do
  d="${OUT}/phase_b_stage1/training/${v}"
  if [[ -f "${d}/DONE" ]]; then echo "  ${v}: DONE"
  elif [[ -f "${d}/HEARTBEAT" ]]; then echo "  ${v}: running hb=$(cat ${d}/HEARTBEAT | head -1)"
  else echo "  ${v}: pending"; fi
done

echo "=== Gates / markers ==="
ls -1 "${OUT}"/*.json "${OUT}"/phase_b_stage1/*.json "${OUT}"/stage2_audit/*.json "${MARKER_DIR}"/* 2>/dev/null || true

echo "=== Monitor pid ==="
if [[ -f "${PID_DIR}/monitor_loop.pid" ]]; then
  pid=$(cat "${PID_DIR}/monitor_loop.pid")
  if kill -0 "$pid" 2>/dev/null; then echo "monitor running pid=$pid"
  else echo "monitor DEAD pid=$pid"; fi
fi
