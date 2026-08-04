#!/usr/bin/env bash
# GPU0: Base live reference shard1 + contract + holdout shard2/3
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

GPU=0
PORT=9200
LIVE="${OUT}/contract_trace/live/base_shard1_tau0"
MODEL="${BASE_MODEL}"

scope7_log "GPU0 queue start"
scope7_run_live "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" shard1 0 base base_shard1_tau0
scope7_contract_pipeline "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" base_shard1_tau0

if scope7_gate_passed "${LIVE}"; then
  scope7_run_live "${GPU}" "${OUT}/holdout_tau0/base_shard2" "${MODEL}" "${PORT}" shard2 0 base base_shard2
  scope7_run_live "${GPU}" "${OUT}/holdout_tau0/base_shard3" "${MODEL}" "${PORT}" shard3 0 base base_shard3
else
  scope7_write_marker "base_holdout_skipped" "failed" "${LIVE}" 0
fi

scope7_log "GPU0 queue complete"
