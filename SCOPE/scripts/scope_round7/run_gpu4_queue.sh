#!/usr/bin/env bash
# GPU4: seed42 archived audit + independent replay + holdout shard3
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

GPU=4
PORT=9204
SEED=42
MODEL="${R5}/merged/o7_r64_seed${SEED}"
LIVE="${OUT}/contract_trace/live/o7_seed${SEED}_shard1_tau0"

scope7_log "GPU4 queue start (seed42 archived audit)"
CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round7/archived_state_audit.py \
  --seed "${SEED}" --gpu cuda:0 \
  --output-dir "${OUT}/contract_trace/replay_hf/archived" \
  >> "${LOG_DIR}/gpu4_archived_seed42.log" 2>&1

# Wait for GPU1 live trace then replay independently
for i in $(seq 1 360); do
  if [[ -f "${LIVE}/live_dup_decision_trace.jsonl" ]]; then break; fi
  sleep 60
done
if [[ -f "${LIVE}/live_dup_decision_trace.jsonl" ]]; then
  scope7_contract_pipeline "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" "o7_seed42_indep"
fi

if scope7_gate_passed "${LIVE}"; then
  scope7_run_live "${GPU}" "${OUT}/holdout_tau0/seed${SEED}_shard3" "${MODEL}" "${PORT}" shard3 "${SEED}" "o7_r64_seed${SEED}" "o7_seed42_shard3"
fi

scope7_log "GPU4 queue complete"
