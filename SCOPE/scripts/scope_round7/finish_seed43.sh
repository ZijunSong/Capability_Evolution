#!/usr/bin/env bash
# Clean rerun seed43 shard1 + contract pipeline + holdout + report
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup
export PARALLEL="${PARALLEL:-64}"
export SCOPE7_NO_RESUME=1

LIVE="${OUT}/contract_trace/live_rerun/o7_seed43_shard1_tau0"
TAG="o7_seed43_shard1_rerun"
MODEL="${R5}/merged/o7_r64_seed43"
HO="${OUT}/holdout_tau0_rerun"

scope7_log "seed43 clean rerun start"
scope7_wait_gpu_free 6 7200
scope7_run_live 6 "${LIVE}" "${MODEL}" 9226 shard1 43 o7_r64_seed43 "${TAG}"
scope7_contract_pipeline 6 "${LIVE}" "${MODEL}" 9226 "${TAG}"

if ! scope7_gate_passed "${LIVE}"; then
  scope7_log "seed43 gate FAILED; aborting holdout"
  exit 1
fi

scope7_log "seed43 gate passed; holdout shard2"
scope7_wait_gpu_free 4 7200
scope7_run_live 4 "${HO}/seed43_shard2" "${MODEL}" 9236 shard2 43 o7_r64_seed43 "${TAG}_shard2"

scope7_log "seed43 holdout shard3"
scope7_wait_gpu_free 5 7200
scope7_run_live 5 "${HO}/seed43_shard3" "${MODEL}" 9237 shard3 43 o7_r64_seed43 "${TAG}_shard3"

python training/scope_round7/build_round7_report.py
scope7_log "seed43 clean rerun DONE"
