#!/usr/bin/env bash
# Barrier 3: score 3 O7 seeds in parallel on GPU 0-2, then aggregate.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope10_setup

scope10_log "Barrier 3: parallel calibration (GPU 0-2)"
for gpu in 0 1 2; do
  seed=$((42 + gpu))
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round10/calibrate_binary.py \
    --score-seed "${seed}" --gpu cuda:0 \
    >> "${LOG_DIR}/barrier3_seed${seed}.log" 2>&1 &
done
wait

python training/scope_round10/calibrate_binary.py --aggregate \
  2>&1 | tee "${LOG_DIR}/barrier3_aggregate.log"
scope10_log "Barrier 3 complete"
