#!/usr/bin/env bash
# Clean conflicting 830 jobs; finish B_OFF missing shards; then run B_ON + T_OFF waves.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

OUT830="${OUT}/gpu0_dup_anchor/confirm_830_seed42"
MANIFEST="${MANIFEST_DIR}/R14_HOLD_830.json"
PARALLEL="${R14_PARALLEL:-16}"
SEED=42

r14_log "cleanup: stop conflicting 830 supervisors/rollouts (keep rollback trains)"

# Kill 830 supervisors and hmin confirm jobs, but NOT train_rollback_lite
pkill -f 'launch_830_confirm_parallel.sh' 2>/dev/null || true
pkill -f 'confirm_830_seed42' 2>/dev/null || true
pkill -f 'fill_b_off_shards56.sh' 2>/dev/null || true
sleep 3
# Kill lingering vllm on 19400-19407 if not used by rollback (rollback doesn't use vllm ports)
for port in 19400 19401 19402 19403 19404 19405 19406 19407; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 3

run_shard() {
  local gpu="$1" label="$2" model="$3" harness="$4" shard="$5" use_dup="$6"
  local out="${OUT830}/${label}/${shard}"
  local expected=104
  if [[ "${shard}" == "shard6" || "${shard}" == "shard7" ]]; then expected=103; fi
  local n=0
  [[ -f "${out}/episodes.jsonl" ]] && n=$(wc -l < "${out}/episodes.jsonl" | tr -d ' ')
  if [[ "${n}" -ge "${expected}" && -f "${out}/summary.json" ]]; then
    r14_log "skip ${label}/${shard} (${n})"
    return 0
  fi
  mkdir -p "${out}"
  local port
  port="$(r14_port_for_gpu "${gpu}")"
  local dup_flag=(--collect-states-only)
  [[ "${use_dup}" == "1" ]] && dup_flag=(--dup-operation)
  r14_log "830 GPU${gpu} ${label} ${shard}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" --n-shards 8 \
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

# Finish B_OFF shard5/6 on GPU0/1 first (GPU5/6 may still train)
H_OFF="${OUT}/gpu0_dup_anchor/B_OFF/harness_module_off.yaml"
H_ON="${OUT}/gpu0_dup_anchor/B_ON/harness_module_on.yaml"
MODEL_T="$(r14_o7_ckpt ${SEED})"

r14_log "wave finish B_OFF shard5/6"
run_shard 0 B_OFF "${BASE_MODEL}" "${H_OFF}" shard5 0 &
p1=$!
sleep 8
run_shard 1 B_OFF "${BASE_MODEL}" "${H_OFF}" shard6 0 &
p2=$!
wait $p1 $p2 || true

# B_ON wave on GPUs not used by rollback. Prefer 0-4,7; if 5/6 free, use them too.
run_wave() {
  local label="$1" model="$2" harness="$3" use_dup="$4"
  r14_log "wave ${label}"
  local pids=()
  local free_gpus=()
  for gpu in 0 1 2 3 4 5 6 7; do
    if [[ "${gpu}" == "5" || "${gpu}" == "6" ]]; then
      if pgrep -f "train_rollback_lite.py" >/dev/null; then
        continue
      fi
    fi
    free_gpus+=("${gpu}")
  done
  # Map shards to available GPUs in rounds
  local shards=(shard0 shard1 shard2 shard3 shard4 shard5 shard6 shard7)
  local i=0
  while [[ $i -lt ${#shards[@]} ]]; do
    local batch_pids=()
    for gpu in "${free_gpus[@]}"; do
      [[ $i -ge ${#shards[@]} ]] && break
      local shard="${shards[$i]}"
      i=$((i+1))
      run_shard "${gpu}" "${label}" "${model}" "${harness}" "${shard}" "${use_dup}" &
      batch_pids+=($!)
      sleep 6
    done
    for pid in "${batch_pids[@]:-}"; do
      wait "${pid}" || true
    done
  done
}

run_wave B_ON "${BASE_MODEL}" "${H_ON}" 0
run_wave T_OFF "${MODEL_T}" "${H_OFF}" 1

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

r14_log "cleanup_and_relaunch_830 complete"
