#!/usr/bin/env bash
# Launch Round-9 P0 retrain for three main hier seeds on GPU 0/1/2 (nohup).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

mkdir -p "${OUT}/wave_b_p0" "${LOG_DIR}"

scope9_log "P0: resample hier train CONTINUE→~0.75"
python training/scope_round9/resample_hier_train_p0.py \
  --target-frac 0.75 --seed 42 \
  | tee "${LOG_DIR}/hier_train_p0_resample.log"

scope9_log "P0: launch main seeds on GPU 0/1/2"
for gpu in 0 1 2; do
  v="${WAVE_B_VARIANTS[$gpu]}"
  rm -rf "${OUT}/wave_b_p0/${v}"
  mkdir -p "${OUT}/wave_b_p0/${v}"
  : > "${LOG_DIR}/wave_b_p0_${v}.log"
  nohup bash "${REPO_ROOT}/scripts/scope_round9/run_wave_b_p0_gpu.sh" "${gpu}" \
    >> "${LOG_DIR}/wave_b_p0_${v}_worker.log" 2>&1 &
  echo $! > "${PID_DIR}/wave_b_p0_gpu${gpu}.pid"
  scope9_log "P0 started gpu=${gpu} variant=${v} pid=$(cat "${PID_DIR}/wave_b_p0_gpu${gpu}.pid")"
  sleep 2
done

# Waiter: when 3/3 DONE, write offline gate from wave_b_p0
nohup bash -c '
  set -euo pipefail
  source "'"${REPO_ROOT}"'/scripts/scope_round9/_common.sh"
  scope9_setup
  MAIN=(rollback_hier_o7_seed42 rollback_hier_o7_seed43 rollback_hier_o7_seed44)
  while true; do
    n=0
    for v in "${MAIN[@]}"; do
      [[ -f "'"${OUT}"'/wave_b_p0/${v}/DONE" ]] && n=$((n+1))
    done
    scope9_log "WAVE B P0 waiter: DONE ${n}/3"
    if [[ "${n}" -ge 3 ]]; then
      # Point check_offline_gate at wave_b_p0 by symlink swap? Instead run inline python.
      python - <<PY
from pathlib import Path
import json, sys
sys.path.insert(0, "'"${REPO_ROOT}"'")
from training.scope_round9.aggregate_wave_b_report import offline_gate
root = Path("'"${OUT}"'/wave_b_p0")
MAIN = ["rollback_hier_o7_seed42","rollback_hier_o7_seed43","rollback_hier_o7_seed44"]
reports=[]
variants={}
for p in sorted(root.glob("*/TRAIN_AND_EVAL_REPORT.json")):
    d=json.loads(p.read_text())
    variants[d.get("variant", p.parent.name)] = d
    if p.parent.name in MAIN:
        reports.append(d)
gate = offline_gate(reports)
bal=[r.get("offline_valid",{}).get("hf_metrics",{}).get("operation_balanced_accuracy",0) for r in reports]
span=(max(bal)-min(bal)) if bal else 999
gate["seed_span_operation_bal_acc"]=span
gate["seed_span_ok"]=span<=0.05
gate["offline_gate_pass"]=bool(gate.get("offline_gate_pass")) and gate["seed_span_ok"]
gate["variants"]={k:{
  "offline_bal_acc": v.get("offline_valid",{}).get("hf_metrics",{}).get("operation_balanced_accuracy"),
  "holdout_bal_acc": v.get("holdout",{}).get("hf_metrics",{}).get("operation_balanced_accuracy"),
  "offline_ContinueRecall": v.get("offline_valid",{}).get("hf_metrics",{}).get("ContinueRecall"),
  "holdout_ContinueRecall": v.get("holdout",{}).get("hf_metrics",{}).get("ContinueRecall"),
  "parity_offline": v.get("offline_valid",{}).get("parity_pass"),
  "parity_holdout": v.get("holdout",{}).get("parity_pass"),
} for k,v in variants.items()}
gate["p0"]=True
out=Path("'"${OUT}"'/OFFLINE_GATE_ROUND9_P0.json")
out.write_text(json.dumps(gate, indent=2)+"\n")
print(json.dumps(gate, indent=2))
raise SystemExit(0 if gate["offline_gate_pass"] else 2)
PY
      scope9_log "WAVE B P0 complete; OFFLINE_GATE_ROUND9_P0.json written"
      break
    fi
    sleep 180
  done
' >> "${LOG_DIR}/wave_b_p0_waiter.log" 2>&1 &
echo $! > "${PID_DIR}/wave_b_p0_waiter.pid"
scope9_log "P0 waiter pid=$(cat "${PID_DIR}/wave_b_p0_waiter.pid")"
scope9_log "P0 launch done. Logs: ${LOG_DIR}/wave_b_p0_*.log"
