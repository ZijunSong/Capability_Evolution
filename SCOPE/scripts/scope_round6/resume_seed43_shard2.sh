#!/usr/bin/env bash
# Resume seed43/shard2 holdout (9/25 done) + finalize report + result-record
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round6/_common.sh"
scope6_setup

CL="${OUT}/closed_loop/holdout_50q/seed43/shard2"
LOG="${LOG_DIR}/phase_d"
MERGED="${R5}/merged"
CAL="${OUT}/calibration/thresholds.json"
TAU=$(python -c "import json; print(json.load(open('${CAL}'))['per_seed']['43']['tau'])")

scope6_log "Resume seed43 shard2 from $(wc -l < "${CL}/episodes.jsonl" 2>/dev/null || echo 0) episodes, tau=${TAU}"

CUDA_VISIBLE_DEVICES=2 python training/scope_round3/hmin_v2_dup_rollout.py \
  --output-dir "${CL}" \
  --manifest "${MANIFEST}" \
  --shard shard2 --n-shards 4 \
  --model-path "${MERGED}/o7_r64_seed43" \
  --vllm-port 9802 \
  --dup-operation \
  --decision-threshold "${TAU}" \
  --dup-seed 43 \
  --checkpoint-label o7_r64_seed43 \
  --parallel 1 \
  --query-timeout-s 600 \
  >> "${LOG}/holdout_43_shard2_resume.log" 2>&1

python training/scope_round6/aggregate_round6.py --run-dir "${CL}"
python training/scope_round6/build_round6_report.py
python training/scope_round6/update_result_record.py

scope6_log "seed43 shard2 resume + result-record update complete"
