#!/usr/bin/env bash
# Parallel Dup 830 confirmation: 8 shards × (B_OFF, B_ON, T_OFF seed42) using free GPUs.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

OUT830="${OUT}/gpu0_dup_anchor/confirm_830_seed42"
MANIFEST="${MANIFEST_DIR}/R14_HOLD_830.json"   # 8-shard layout
PARALLEL="${R14_PARALLEL:-16}"
SEED=42
MODEL_BASE="${BASE_MODEL}"
MODEL_T="$(r14_o7_ckpt ${SEED})"
HARNESS_OFF="${OUT}/gpu0_dup_anchor/B_OFF/harness_module_off.yaml"
HARNESS_ON="${OUT}/gpu0_dup_anchor/B_ON/harness_module_on.yaml"

mkdir -p "${OUT830}"
r14_log "830 parallel confirm start"

run_shard() {
  local gpu="$1" label="$2" model="$3" harness="$4" shard="$5" use_dup="$6"
  local out="${OUT830}/${label}/${shard}"
  local expected=104
  if [[ "${shard}" == "shard6" || "${shard}" == "shard7" ]]; then expected=103; fi
  local n=0
  [[ -f "${out}/episodes.jsonl" ]] && n=$(wc -l < "${out}/episodes.jsonl" | tr -d ' ')
  if [[ "${n}" -ge "${expected}" && -f "${out}/summary.json" ]]; then
    r14_log "skip complete 830 ${label} ${shard} (${n})"
    return 0
  fi
  mkdir -p "${out}"
  local port
  port="$(r14_port_for_gpu "${gpu}")"
  local dup_flag=(--collect-states-only)
  if [[ "${use_dup}" == "1" ]]; then
    dup_flag=(--dup-operation)
  fi
  r14_log "830 GPU${gpu} ${label} ${shard} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" \
    --n-shards 8 \
    --model-path "${model}" \
    --harness-config "${harness}" \
    --temperature 0.0 \
    --vllm-port "${port}" \
    --dup-seed "${SEED}" \
    --checkpoint-label "${label}" \
    --parallel "${PARALLEL}" \
    --decision-threshold 0 \
    --resume \
    "${dup_flag[@]}" \
    >> "${LOG_DIR}/dup830_${label}_${shard}.log" 2>&1
}

# Wave schedule: for each condition, run 8 shards on GPUs 0-7
for label_model_harness_dup in \
  "B_OFF|${MODEL_BASE}|${HARNESS_OFF}|0" \
  "B_ON|${MODEL_BASE}|${HARNESS_ON}|0" \
  "T_OFF|${MODEL_T}|${HARNESS_OFF}|1"
do
  IFS='|' read -r label model harness use_dup <<< "${label_model_harness_dup}"
  r14_log "830 wave ${label}"
  pids=()
  for gpu in 0 1 2 3 4 5 6 7; do
    # Skip GPUs still training rollback if processes alive
    if [[ "${gpu}" == "5" ]] || [[ "${gpu}" == "6" ]]; then
      if pgrep -f "train_rollback_lite.py.*gpu ${gpu}|train_rollback_lite.py.*--gpu ${gpu}" >/dev/null; then
        r14_log "830 defer shard on GPU${gpu} (rollback training)"
        continue
      fi
    fi
    shard="shard${gpu}"
    run_shard "${gpu}" "${label}" "${model}" "${harness}" "${shard}" "${use_dup}" &
    pids+=($!)
    sleep 8
  done
  for pid in "${pids[@]:-}"; do
    wait "${pid}" || true
  done
done

# Aggregate shard summaries into confirm_830 RETIREMENT-like report
python - <<'PY'
import json
from pathlib import Path
from collections import defaultdict
root = Path("outputs/scope_round14/gpu0_dup_anchor/confirm_830_seed42")
report = {"schema_version": "scope.round14.dup830.v1", "conditions": {}}
for label in ["B_OFF", "B_ON", "T_OFF"]:
    eps = 0
    recalls = []
    bals = []
    drrs = []
    for shard in root.joinpath(label).glob("shard*"):
        s = shard / "summary.json"
        if not s.exists():
            continue
        sm = json.loads(s.read_text())
        eps += int(sm.get("n_completed") or 0)
        # episode recalls
        ep = shard / "episodes.jsonl"
        if ep.exists():
            for line in ep.open():
                d = json.loads(line)
                if "recall" in d:
                    recalls.append(float(d["recall"]))
        tel = sm.get("dup_telemetry") or {}
        if tel.get("balanced_accuracy") is not None:
            bals.append(float(tel["balanced_accuracy"]))
            drrs.append(float(tel.get("duplicate_reject_rate") or 0))
    report["conditions"][label] = {
        "n_episodes": eps,
        "mean_recall": sum(recalls)/len(recalls) if recalls else None,
        "mean_balanced_accuracy": sum(bals)/len(bals) if bals else None,
        "mean_duplicate_reject_rate": sum(drrs)/len(drrs) if drrs else None,
    }
(root / "CONFIRM_830_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

r14_log "830 parallel confirm complete"
