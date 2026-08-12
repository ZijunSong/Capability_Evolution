#!/usr/bin/env bash
# GPU6: rollback_lite — remap R13 → local train → gate → optional closed-loop
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

GPU="${1:-6}"
OUT_GPU="${OUT}/gpu6_rollback_lite"
DS="${DATA_DIR}/rollback_lite"
HB="${OUT_GPU}/HEARTBEAT"

r14_log "GPU${GPU} rollback_lite queue"
r14_touch_hb "${HB}"

python training/scope_round14/remap_rollback_lite.py \
  --output-dir "${DS}" \
  --resume \
  2>&1 | tee "${LOG_DIR}/gpu6_remap.log"

GATE="${DS}/DATASET_GATE.json"
if [[ ! -f "${GATE}" ]]; then
  r14_log "GPU6: remap gate missing — stop"
  echo "UNRESOLVED" > "${OUT_GPU}/DONE"
  exit 1
fi

OBJECTIVE="hard_boundary"
SEEDS=(42)
r14_touch_hb "${HB}"
python training/scope_round14/train_rollback_lite.py \
  --train "${DS}/train.jsonl" \
  --valid "${DS}/valid.jsonl" \
  --seed 42 \
  --gpu "${GPU}" \
  --output-dir "${OUT_GPU}/train_seed42" \
  --objective "${OBJECTIVE}" \
  --manifest "${R14_FRESH100}" \
  --resume \
  2>&1 | tee "${LOG_DIR}/gpu6_train_seed42.log"

LOCAL_GATE="${OUT_GPU}/train_seed42/LOCAL_GATE.json"
SEED42_OK="False"
if [[ -f "${LOCAL_GATE}" ]]; then
  SEED42_OK="$(r14_gate_pass "${LOCAL_GATE}" gate_b_pass)"
fi

BAL="$(python -c "import json;print(json.load(open('${OUT_GPU}/train_seed42/METRICS.json')).get('balanced_accuracy',0) if __import__('pathlib').Path('${OUT_GPU}/train_seed42/METRICS.json').exists() else 0)")"
RR="$(python -c "
import json
from pathlib import Path
p=Path('${OUT_GPU}/train_seed42/METRICS.json')
if not p.exists(): print(0); raise SystemExit
m=json.loads(p.read_text())
r=m.get('per_class_recall') or m.get('class_recall') or {}
print(r.get('RECOVER',0))
")"

if [[ "${SEED42_OK}" != "True" ]] && python -c "exit(0 if float('${BAL}')<0.75 and float('${RR}')<0.5 else 1)"; then
  r14_log "GPU6 seed42 failed hard — trying pairwise repair once"
  python training/scope_round14/train_rollback_lite.py \
    --train "${DS}/train.jsonl" \
    --valid "${DS}/valid.jsonl" \
    --seed 42 \
    --gpu "${GPU}" \
    --output-dir "${OUT_GPU}/train_seed42_pairwise" \
    --objective pairwise_margin \
    --manifest "${R14_FRESH100}" \
    2>&1 | tee "${LOG_DIR}/gpu6_train_seed42_repair.log" || true
elif [[ "${SEED42_OK}" == "True" ]]; then
  SEEDS=(42 43 44)
fi

for s in "${SEEDS[@]}"; do
  [[ "${s}" == "42" ]] && continue
  r14_touch_hb "${HB}"
  python training/scope_round14/train_rollback_lite.py \
    --train "${DS}/train.jsonl" \
    --valid "${DS}/valid.jsonl" \
    --seed "${s}" \
    --gpu "${GPU}" \
    --output-dir "${OUT_GPU}/train_seed${s}" \
    --objective "${OBJECTIVE}" \
    --manifest "${R14_FRESH100}" \
    --resume \
    2>&1 | tee "${LOG_DIR}/gpu6_train_seed${s}.log" || break
done

python training/scope_round14/build_capability_evidence.py \
  --capability rollback_lite \
  --dataset "${DS}" \
  --metrics-json "${OUT_GPU}/train_seed42/METRICS.json" \
  --output-dir "${OUT_GPU}" \
  --gpu "${GPU}" \
  --seed 42 \
  --manifest "${R14_FRESH100}" \
  --resume

if [[ -f "${LOCAL_GATE}" ]] && [[ "$(r14_gate_pass "${LOCAL_GATE}" gate_b_pass)" == "True" ]]; then
  python training/scope_round14/run_module_retirement_eval.py \
    --capability rollback_lite \
    --manifest "${R14_FRESH100}" \
    --output-dir "${OUT_GPU}/retirement" \
    --gpu "${GPU}" \
    --seed 42 \
    --resume \
    --dry-run
else
  r14_log "GPU6 local gate not pass — retirement pending"
fi

echo "DONE" > "${OUT_GPU}/DONE"
r14_log "GPU${GPU} rollback_lite queue complete"
