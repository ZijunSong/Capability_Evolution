#!/usr/bin/env bash
# Finish remaining T_OFF 830 shards as GPUs free; wait for B_OFF/B_ON fills first.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

OUT830="${OUT}/gpu0_dup_anchor/confirm_830_seed42/T_OFF"
MANIFEST="${MANIFEST_DIR}/R14_HOLD_830.json"
H_OFF="${OUT}/gpu0_dup_anchor/B_OFF/harness_module_off.yaml"
MODEL_T="$(r14_o7_ckpt 42)"
PARALLEL="${R14_PARALLEL:-16}"

# Wait for finishing fills
for i in $(seq 1 60); do
  if pgrep -f 'confirm_830_seed42/B_OFF/shard5' >/dev/null || pgrep -f 'confirm_830_seed42/B_ON/shard6' >/dev/null; then
    r14_log "finish_t_off: waiting for B_OFF/B_ON fills..."
    sleep 30
  else
    break
  fi
done

run_one() {
  local gpu="$1" shard="$2"
  local out="${OUT830}/${shard}"
  local expected=104
  [[ "${shard}" == "shard6" || "${shard}" == "shard7" ]] && expected=103
  local n=0
  [[ -f "${out}/episodes.jsonl" ]] && n=$(wc -l < "${out}/episodes.jsonl" | tr -d ' ')
  if [[ "${n}" -ge "${expected}" && -f "${out}/summary.json" ]]; then
    r14_log "skip T_OFF/${shard} (${n})"
    return 0
  fi
  # Skip if already running
  if pgrep -f "confirm_830_seed42/T_OFF/${shard}" >/dev/null; then
    r14_log "already running T_OFF/${shard}"
    return 0
  fi
  mkdir -p "${out}"
  local port
  port="$(r14_port_for_gpu "${gpu}")"
  r14_log "T_OFF ${shard} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" --n-shards 8 \
    --model-path "${MODEL_T}" \
    --harness-config "${H_OFF}" \
    --temperature 0.0 \
    --vllm-port "${port}" \
    --dup-seed 42 \
    --checkpoint-label T_OFF \
    --parallel "${PARALLEL}" \
    --decision-threshold 0 \
    --resume \
    --dup-operation \
    >> "${LOG_DIR}/dup830_T_OFF_${shard}.log" 2>&1 &
  echo $! > "${PID_DIR}/dup830_t_off_${shard}.pid"
}

# Prefer GPUs not in rollback train
run_one 0 shard0
sleep 5
run_one 2 shard2
sleep 5
run_one 3 shard3
sleep 5
run_one 4 shard4
sleep 5
run_one 0 shard6 || true
# shard1/5 may already be running; shard7 on GPU2 after delay
sleep 5
run_one 2 shard7 || true

r14_log "finish_t_off_830 launched; waiting"
wait || true

python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/scope_round14/gpu0_dup_anchor/confirm_830_seed42")
report = {"schema_version": "scope.round14.dup830.v1", "conditions": {}}
for label in ["B_OFF", "B_ON", "T_OFF"]:
    eps = 0
    recalls = []
    bals = []
    drrs = []
    for shard in sorted((root / label).glob("shard*")):
        s = shard / "summary.json"
        if not s.exists():
            continue
        sm = json.loads(s.read_text())
        eps += int(sm.get("n_completed") or 0)
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
        "mean_recall": (sum(recalls) / len(recalls)) if recalls else None,
        "mean_balanced_accuracy": (sum(bals) / len(bals)) if bals else None,
        "mean_duplicate_reject_rate": (sum(drrs) / len(drrs)) if drrs else None,
    }
(root / "CONFIRM_830_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
r14_log "finish_t_off_830 complete"
