#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

echo "=== Round 9 status ==="
echo "Branch: $(git branch --show-current)"
echo ""
echo "Wave A:"
for v in "${WAVE_A_VARIANTS[@]}"; do
  m="${MARKER_DIR}/wave_a_${v}.DONE"
  [[ -f "${m}" ]] && s="DONE" || s="pending"
  echo "  ${v}: ${s}"
done
echo ""
echo "Wave B:"
for v in "${WAVE_B_VARIANTS[@]}"; do
  m="${OUT}/wave_b/${v}/DONE"
  [[ -f "${m}" ]] && s="DONE" || s="pending"
  echo "  ${v}: ${s}"
done
echo ""
echo "GPU:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
