#!/usr/bin/env bash
# Shared capability queue: build events → gate A → train → local gate → retirement
set -euo pipefail

r14_capability_queue() {
  local gpu="$1"
  local capability="$2"
  local out_gpu="$3"
  local ds="$4"
  local hb="$5"
  local smoke_manifest="${6:-${R14_SMOKE20}}"

  r14_log "GPU${gpu} ${capability} queue"
  r14_touch_hb "${hb}"

  python training/scope_round14/build_event_dataset.py \
    --capability "${capability}" \
    --output-dir "${ds}" \
    --gpu "${gpu}" \
    --seed 42 \
    --manifest "${R14_FRESH100}" \
    --resume \
    2>&1 | tee "${LOG_DIR}/gpu${gpu}_build_events.log"

  local gate="${ds}/DATASET_GATE.json"
  local gate_a="False"
  local status="UNRESOLVED"
  if [[ -f "${gate}" ]]; then
    gate_a="$(r14_gate_pass "${gate}" gate_a_pass)"
    status="$(python -c "import json;print(json.load(open('${gate}')).get('status','UNRESOLVED'))")"
  fi

  local local_ok="False"
  if [[ "${gate_a}" == "True" ]]; then
    r14_touch_hb "${hb}"
    python training/scope_round14/train_local_decision.py \
      --capability "${capability}" \
      --train "${ds}/train.jsonl" \
      --valid "${ds}/valid.jsonl" \
      --seed 42 \
      --gpu "${gpu}" \
      --output-dir "${out_gpu}/train_seed42" \
      --objective discriminative_ce \
      --manifest "${R14_FRESH100}" \
      --resume \
      2>&1 | tee "${LOG_DIR}/gpu${gpu}_train_seed42.log" || true

    if [[ -f "${out_gpu}/train_seed42/DONE" ]]; then
      local_ok="$(python -c "
import json
from pathlib import Path
p=Path('${out_gpu}/train_seed42/METRICS.json')
if p.exists():
  m=json.loads(p.read_text())
  bal=float(m.get('balanced_accuracy') or 0)
  rec=min((m.get('per_class_recall') or m.get('class_recall') or {'x':0}).values() or [0])
  print('True' if bal>=0.72 or rec>=0.65 else 'False')
else:
  print('False')
")"
      if [[ "${local_ok}" == "True" ]]; then
        for s in 43 44; do
          r14_touch_hb "${hb}"
          python training/scope_round14/train_local_decision.py \
            --capability "${capability}" \
            --train "${ds}/train.jsonl" \
            --valid "${ds}/valid.jsonl" \
            --seed "${s}" \
            --gpu "${gpu}" \
            --output-dir "${out_gpu}/train_seed${s}" \
            --objective discriminative_ce \
            --manifest "${R14_FRESH100}" \
            --resume \
            2>&1 | tee "${LOG_DIR}/gpu${gpu}_train_seed${s}.log" || break
        done
      else
        r14_touch_hb "${hb}"
        python training/scope_round14/train_local_decision.py \
          --capability "${capability}" \
          --train "${ds}/train.jsonl" \
          --valid "${ds}/valid.jsonl" \
          --seed 42 \
          --gpu "${gpu}" \
          --output-dir "${out_gpu}/train_seed42_pairwise" \
          --objective pairwise_margin \
          --manifest "${R14_FRESH100}" \
          --resume \
          2>&1 | tee "${LOG_DIR}/gpu${gpu}_train_seed42_repair.log" || true
      fi
    fi

    if [[ "${local_ok}" == "True" ]] || [[ -f "${out_gpu}/train_seed42/DONE" ]]; then
      python training/scope_round14/run_module_retirement_eval.py \
        --capability "${capability}" \
        --manifest "${smoke_manifest}" \
        --output-dir "${out_gpu}/smoke_retirement" \
        --gpu "${gpu}" \
        --seed 42 \
        --resume \
        --run-closed-loop \
        2>&1 | tee "${LOG_DIR}/gpu${gpu}_smoke_retire.log" || \
      python training/scope_round14/run_module_retirement_eval.py \
        --capability "${capability}" \
        --manifest "${smoke_manifest}" \
        --output-dir "${out_gpu}/smoke_retirement" \
        --gpu "${gpu}" \
        --seed 42 \
        --resume \
        --dry-run \
        2>&1 | tee -a "${LOG_DIR}/gpu${gpu}_smoke_retire.log"
    fi
  fi

  python training/scope_round14/build_capability_evidence.py \
    --capability "${capability}" \
    --dataset "${ds}" \
    --metrics-json "${out_gpu}/train_seed42/METRICS.json" \
    --output-dir "${out_gpu}" \
    --gpu "${gpu}" \
    --seed 42 \
    --manifest "${R14_FRESH100}" \
    --resume \
    2>&1 | tee "${LOG_DIR}/gpu${gpu}_evidence.log" || true

  r14_touch_hb "${hb}"
  echo "${status}" > "${out_gpu}/DONE"
  r14_log "GPU${gpu} ${capability} complete status=${status}"
}
