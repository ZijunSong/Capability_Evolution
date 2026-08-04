#!/usr/bin/env bash
# GPU3: O7 seed44 live shard1 + contract + holdout shard2
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

GPU=3
PORT=9203
SEED=44
MODEL="${R5}/merged/o7_r64_seed${SEED}"
LIVE="${OUT}/contract_trace/live/o7_seed${SEED}_shard1_tau0"

scope7_log "GPU3 queue start (seed44)"
scope7_run_live "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" shard1 "${SEED}" "o7_r64_seed${SEED}" "o7_seed44_shard1"
scope7_contract_pipeline "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" "o7_seed44_shard1"

if scope7_gate_passed "${LIVE}"; then
  scope7_run_live "${GPU}" "${OUT}/holdout_tau0/seed${SEED}_shard2" "${MODEL}" "${PORT}" shard2 "${SEED}" "o7_r64_seed${SEED}" "o7_seed44_shard2"
else
  scope7_write_marker "o7_seed44_holdout_skipped" "failed" "${LIVE}" 0
fi

scope7_log "GPU3 queue complete"
