#!/usr/bin/env bash
# GPU5: seed43 archived audit + root cause + holdout shard3
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

GPU=5
PORT=9205
SEED=43
MODEL="${R5}/merged/o7_r64_seed${SEED}"
LIVE="${OUT}/contract_trace/live/o7_seed${SEED}_shard1_tau0"

scope7_log "GPU5 queue start (seed43 archived audit)"
CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round7/archived_state_audit.py \
  --seed "${SEED}" --gpu cuda:0 \
  --output-dir "${OUT}/contract_trace/replay_hf/archived" \
  >> "${LOG_DIR}/gpu5_archived_seed43.log" 2>&1

# Document Round6 threshold_zero/seed43 anomaly
python - <<'PY' >> "${LOG_DIR}/gpu5_root_cause.log" 2>&1
from pathlib import Path
import json
r6 = Path("outputs/scope_round6/closed_loop/calib_25q/threshold_zero/seed43")
out = Path("outputs/scope_round7/contract_trace/comparisons/seed43_threshold_zero_root_cause.md")
out.parent.mkdir(parents=True, exist_ok=True)
lines = ["# seed43 threshold_zero root cause\n"]
if r6.exists():
    sm = r6 / "summary.json"
    if sm.exists():
        d = json.loads(sm.read_text())
        lines.append(f"Round6 threshold_zero seed43 summary: {json.dumps(d.get('dup_telemetry', {}), indent=2)}\n")
lines.append("Round7 will re-run with tau=0 contract trace for direct comparison.\n")
out.write_text("".join(lines))
print(f"Wrote {out}")
PY

for i in $(seq 1 720); do
  if [[ -f "${LIVE}/live_dup_decision_trace.jsonl" ]]; then break; fi
  sleep 60
done
if [[ -f "${LIVE}/live_dup_decision_trace.jsonl" ]]; then
  scope7_contract_pipeline "${GPU}" "${LIVE}" "${MODEL}" "${PORT}" "o7_seed43_indep"
fi

if scope7_gate_passed "${LIVE}"; then
  scope7_run_live "${GPU}" "${OUT}/holdout_tau0/seed${SEED}_shard3" "${MODEL}" "${PORT}" shard3 "${SEED}" "o7_r64_seed${SEED}" "o7_seed43_shard3"
fi

scope7_log "GPU5 queue complete"
