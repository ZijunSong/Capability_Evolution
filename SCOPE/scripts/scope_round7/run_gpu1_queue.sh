#!/usr/bin/env bash
# GPU1: O7 seed42 live shard1 + contract + holdout
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

GPU=1
PORT=9201
SEED=42
MODEL="${R5}/merged/o7_r64_seed${SEED}"
LIVE="${OUT}/contract_trace/live/o7_seed${SEED}_shard1_tau0"

scope7_log "GPU1 queue start (seed42)"
scope7_run_live "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" shard1 "${SEED}" "o7_r64_seed${SEED}" "o7_seed42_shard1"
scope7_contract_pipeline "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" "o7_seed42_shard1"

if scope7_gate_passed "${LIVE}"; then
  scope7_run_live "${GPU}" "${OUT}/holdout_tau0/seed${SEED}_shard2" "${MODEL}" "${PORT}" shard2 "${SEED}" "o7_r64_seed${SEED}" "o7_seed42_shard2"
  # shard3 handled by GPU4 after archived audit
else
  scope7_write_marker "o7_seed42_holdout_skipped" "failed" "${LIVE}" 0
fi

scope7_log "GPU1 queue complete"
