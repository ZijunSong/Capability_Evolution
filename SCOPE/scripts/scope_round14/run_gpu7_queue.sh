#!/usr/bin/env bash
# GPU7: Dup method ablation A–D
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

GPU="${1:-7}"
OUT_GPU="${OUT}/gpu7_method_ablation"
HB="${OUT_GPU}/HEARTBEAT"

r14_log "GPU${GPU} method ablation queue"
r14_touch_hb "${HB}"

python training/scope_round14/run_method_ablation_dup.py \
  --output-dir "${OUT_GPU}" \
  --gpu0-anchor "${OUT}/gpu0_dup_anchor" \
  2>&1 | tee "${LOG_DIR}/gpu7_ablation.log"

python training/scope_round14/aggregate_portfolio.py \
  --output-dir "${OUT}" \
  2>&1 | tee "${LOG_DIR}/gpu7_aggregate.log"

echo "DONE" > "${OUT_GPU}/DONE"
r14_log "GPU${GPU} ablation complete"
