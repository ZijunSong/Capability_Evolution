#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
scope10_setup

echo "=== Round 10 Status ==="
echo "Branch: $(git branch --show-current)"
echo "Markers:"
ls -1 "${MARKER_DIR}"/*.DONE 2>/dev/null || echo "  (none)"
echo ""
echo "Training variants:"
for v in "${TRAINING_VARIANTS[@]}"; do
  done_f=""
  [[ -f "${OUT}/training/${v}/DONE" ]] && done_f="[DONE]"
  echo "  ${v} ${done_f}"
done
echo ""
echo "GPU:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
echo ""
echo "Recent supervisor log:"
tail -5 "${LOG_DIR}/round10_supervisor.log" 2>/dev/null || true
