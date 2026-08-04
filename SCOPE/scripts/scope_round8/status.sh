#!/usr/bin/env bash
# Round 8 GPU status monitor
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

echo "=== Round 8 Status $(date -Is) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "=== GPU PIDs ==="
for G in 0 1 2 3 4 5 6 7; do
  pf="${PID_DIR}/gpu${G}.pid"
  if [[ -f "${pf}" ]]; then
    pid=$(cat "${pf}")
    if kill -0 "${pid}" 2>/dev/null; then
      echo "GPU${G}: running pid=${pid}"
    else
      echo "GPU${G}: pid ${pid} finished"
    fi
  else
    echo "GPU${G}: no pid file"
  fi
done

echo ""
echo "=== Dup 830 progress ==="
for label in base seed42 seed43 seed44; do
  dir="${OUT}/dup_retention_830/${label}"
  if [[ -d "${dir}" ]]; then
  total=0
    for s in 0 1 2 3; do
      ep="${dir}/shard${s}/episodes.jsonl"
      n=$(scope8_count_episodes "${ep}")
      total=$((total + n))
      echo "  ${label}/shard${s}: ${n}"
    done
    echo "  ${label} total: ${total}/830"
  fi
done

echo ""
echo "=== AgentCore diagnostic ==="
find "${OUT}/agent_core_diagnostic" -name episodes.jsonl 2>/dev/null | while read -r ep; do
  echo "  $(dirname "${ep}"): $(scope8_count_episodes "${ep}")"
done

echo ""
echo "=== Rollback collection ==="
find "${OUT}/rollback_collection" -name rollback_events.jsonl 2>/dev/null | while read -r ep; do
  echo "  $(dirname "${ep}"): $(wc -l < "${ep}" | tr -d ' ') events"
done
