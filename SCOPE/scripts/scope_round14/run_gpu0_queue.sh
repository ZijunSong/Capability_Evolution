#!/usr/bin/env bash
# GPU0: Dup anchor + ModuleRetirementGate calibration on R14_FRESH100
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

GPU="${1:-0}"
OUT_GPU="${OUT}/gpu0_dup_anchor"
HB="${OUT_GPU}/HEARTBEAT"
PARALLEL="${R14_PARALLEL:-16}"

r14_log "GPU${GPU} queue start"
r14_touch_hb "${HB}"

if r14_wave0_complete; then
  r14_log "GPU0 wave0 complete — aggregating"
  python training/scope_round14/aggregate_dup_anchor.py \
    --anchor-dir "${OUT_GPU}" \
    --resume \
    2>&1 | tee -a "${LOG_DIR}/gpu0_aggregate.log"
else
  r14_log "GPU0 wave0 incomplete — sequential fallback on GPU${GPU}"
  for cond_seed in "B_OFF:42" "B_ON:42" "T_OFF:42" "T_OFF:43" "T_OFF:44"; do
    cond="${cond_seed%%:*}"
    seed="${cond_seed##*:}"
    out="${OUT_GPU}/${cond}"
    if [[ "${cond}" == "T_OFF" ]]; then
      out="${OUT_GPU}/T_OFF_seed${seed}"
    fi
    r14_touch_hb "${HB}"
    python training/scope_round14/run_module_retirement_eval.py \
      --capability duplicate_evidence \
      --manifest "${R14_FRESH100}" \
      --output-dir "${out}" \
      --gpu "${GPU}" \
      --seed "${seed}" \
      --conditions "${cond}" \
      --temperature 0.0 \
      --parallel "${PARALLEL}" \
      --flat-output \
      --resume \
      --run-closed-loop \
      2>&1 | tee -a "${LOG_DIR}/gpu0_${cond}_seed${seed}.log"
  done
  python training/scope_round14/aggregate_dup_anchor.py \
    --anchor-dir "${OUT_GPU}" \
    2>&1 | tee -a "${LOG_DIR}/gpu0_aggregate.log"
fi

# GPU0-C: 830 confirmation when gate passes (best seed from aggregate)
GATE="${OUT_GPU}/DUP_RETIREMENT_GATE.json"
BEST_SEED=42
if [[ -f "${GATE}" ]]; then
  BEST_SEED="$(python -c "import json;print(json.load(open('${GATE}')).get('seed_stability',{}).get('best_seed',42))")"
fi

if [[ -f "${GATE}" ]] && [[ "$(r14_gate_pass "${GATE}" gate_c_pass)" == "True" ]]; then
  r14_touch_hb "${HB}"
  python training/scope_round14/run_module_retirement_eval.py \
    --capability duplicate_evidence \
    --manifest "${MANIFEST_DIR}/R14_HOLD_830.json" \
    --output-dir "${OUT_GPU}/confirm_830_seed${BEST_SEED}" \
    --gpu "${GPU}" \
    --seed "${BEST_SEED}" \
    --conditions B_OFF B_ON T_OFF \
    --temperature 0.0 \
    --parallel "${PARALLEL}" \
    --resume \
    --run-closed-loop \
    2>&1 | tee -a "${LOG_DIR}/gpu0_confirm830.log"
else
  r14_log "GPU0 skip 830 confirm (gate_c not pass or gate missing)"
fi

python training/scope_round14/build_capability_evidence.py \
  --capability duplicate_evidence \
  --metrics-json "${OUT_GPU}/RETIREMENT_EVAL.json" \
  --output-dir "${OUT_GPU}" \
  --gpu "${GPU}" \
  --seed "${BEST_SEED}" \
  --manifest "${R14_FRESH100}" \
  --resume

r14_touch_hb "${HB}"
echo "DONE" > "${OUT_GPU}/DONE"
r14_log "GPU${GPU} queue complete"
