#!/usr/bin/env bash
# Barrier 5.1: 12 micro-overfit jobs (3 datasets × 4 sizes) across 8 GPUs.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope10_setup

MICRO_JOBS=(
  "D1_live_only 2"
  "D1_live_only 8"
  "D1_live_only 32"
  "D1_live_only 128"
  "D2_mixed_aligned 2"
  "D2_mixed_aligned 8"
  "D2_mixed_aligned 32"
  "D2_mixed_aligned 128"
  "D3_mixed_hard_continue 2"
  "D3_mixed_hard_continue 8"
  "D3_mixed_hard_continue 32"
  "D3_mixed_hard_continue 128"
)

run_gpu_queue() {
  local gpu="$1"
  shift
  local -a jobs=("$@")
  for job in "${jobs[@]}"; do
    local ds="${job%% *}"
    local sz="${job##* }"
    scope10_log "micro gpu${gpu}: ${ds} d${sz}"
    CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round10/run_micro_overfit.py \
      --dataset "${ds}" --size "${sz}" --gpu cuda:0 \
      >> "${LOG_DIR}/micro_${ds}_d${sz}_gpu${gpu}.log" 2>&1
  done
}

# Round-robin assign jobs to GPUs 0-7
declare -a GPU_QUEUES
for gpu in 0 1 2 3 4 5 6 7; do
  GPU_QUEUES[gpu]=""
done
idx=0
for job in "${MICRO_JOBS[@]}"; do
  gpu=$((idx % 8))
  GPU_QUEUES[gpu]="${GPU_QUEUES[gpu]}${job}|"
  idx=$((idx + 1))
done

scope10_log "Barrier 5.1: micro-overfit 12 jobs on 8 GPUs"
for gpu in 0 1 2 3 4 5 6 7; do
  IFS='|' read -ra jobs <<< "${GPU_QUEUES[gpu]}"
  # drop trailing empty element
  local_jobs=()
  for j in "${jobs[@]}"; do
    [[ -n "${j}" ]] && local_jobs+=("${j}")
  done
  if [[ "${#local_jobs[@]}" -gt 0 ]]; then
    run_gpu_queue "${gpu}" "${local_jobs[@]}" &
  fi
done
wait

# Verify all reports exist
python - <<'PY'
import json, sys
from pathlib import Path
root = Path("outputs/scope_round10/training/micro_overfit")
datasets = ["D1_live_only", "D2_mixed_aligned", "D3_mixed_hard_continue"]
sizes = [2, 8, 32, 128]
fail = False
for ds in datasets:
    for sz in sizes:
        r = root / ds / f"d{sz}" / "MICRO_REPORT.json"
        if not r.exists():
            print(f"MISSING {r}", file=sys.stderr)
            fail = True
            continue
        if not json.loads(r.read_text()).get("pass"):
            print(f"FAIL {r}", file=sys.stderr)
            fail = True
if fail:
    sys.exit(2)
print("micro-overfit gate PASS")
PY

scope10_log "Barrier 5.1 micro-overfit complete"
