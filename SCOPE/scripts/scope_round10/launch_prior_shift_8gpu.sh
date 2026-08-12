#!/usr/bin/env bash
# Barrier 2.2: score models in parallel across 8 GPUs, then merge.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope10_setup

MODELS=(
  rollback_o7_seed42
  rollback_o7_seed43
  rollback_o7_seed44
  rollback_hier_o7_seed42
  rollback_hier_o7_seed43
  rollback_hier_o7_seed44
  rollback_flat_o7_seed42_repro
  rollback_hier_prompt_hint_seed42
)

scope10_log "Prior shift: 8 models on GPU 0-7, then base_agent_core + merge"
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round10/prior_shift_report.py \
    --gpu cuda:0 --models "${MODELS[$gpu]}" \
    >> "${LOG_DIR}/prior_shift_${MODELS[$gpu]}.log" 2>&1 &
done
wait

CUDA_VISIBLE_DEVICES=0 python training/scope_round10/prior_shift_report.py \
  --gpu cuda:0 --models base_agent_core \
  >> "${LOG_DIR}/prior_shift_base_agent_core.log" 2>&1

python training/scope_round10/prior_shift_report.py --merge-only \
  >> "${LOG_DIR}/prior_shift_merge.log" 2>&1

scope10_log "Prior shift parallel complete"
