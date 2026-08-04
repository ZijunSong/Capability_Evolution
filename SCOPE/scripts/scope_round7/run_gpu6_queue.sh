#!/usr/bin/env bash
# GPU6: seed44 archived audit + holdout shard3
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

GPU=6
PORT=9206
SEED=44
MODEL="${R5}/merged/o7_r64_seed${SEED}"
LIVE="${OUT}/contract_trace/live/o7_seed${SEED}_shard1_tau0"

scope7_log "GPU6 queue start (seed44 archived audit)"
CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round7/archived_state_audit.py \
  --seed "${SEED}" --gpu cuda:0 \
  --output-dir "${OUT}/contract_trace/replay_hf/archived" \
  >> "${LOG_DIR}/gpu6_archived_seed44.log" 2>&1

for i in $(seq 1 360); do
  if [[ -f "${LIVE}/live_dup_decision_trace.jsonl" ]]; then break; fi
  sleep 60
done
if [[ -f "${LIVE}/live_dup_decision_trace.jsonl" ]]; then
  scope7_contract_pipeline "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" "o7_seed44_indep"
fi

if scope7_gate_passed "${LIVE}"; then
  scope7_run_live "${GPU}" "${OUT}/holdout_tau0/seed${SEED}_shard3" "${MODEL}" "${PORT}" shard3 "${SEED}" "o7_r64_seed${SEED}" "o7_seed44_shard3"
fi

scope7_log "GPU6 queue complete"
