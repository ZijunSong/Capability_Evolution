#!/usr/bin/env bash
# Phase D: 50q holdout (shard2+shard3) closed-loop gate
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round6/_common.sh"
scope6_setup

CL="${OUT}/closed_loop/holdout_50q"
LOG="${LOG_DIR}/phase_d"
MERGED="${R5}/merged"
CAL="${OUT}/calibration/thresholds.json"
mkdir -p "${CL}" "${LOG}"

# Use per-seed calibrated thresholds from Phase C
get_tau() {
  local seed=$1
  python -c "import json; d=json.load(open('${CAL}')); print(d['per_seed'][${seed}]['tau'])"
}

run_shard() {
  local gpu=$1 seed=$2 shard=$3 port=$4
  local tau
  tau=$(get_tau "${seed}")
  local model="${MERGED}/o7_r64_seed${seed}"
  local out="${CL}/seed${seed}/${shard}"
  if [[ -f "${out}/aggregated_metrics.json" ]]; then return 0; fi
  mkdir -p "${out}"
  scope6_log "Holdout seed${seed} ${shard} GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" --n-shards 4 \
    --model-path "${model}" \
    --vllm-port "${port}" \
    --dup-operation \
    --decision-threshold "${tau}" \
    --dup-seed "${seed}" \
    --checkpoint-label "o7_r64_seed${seed}" \
    --parallel 1 >> "${LOG}/holdout_${seed}_${shard}.log" 2>&1
  python training/scope_round6/aggregate_round6.py --run-dir "${out}"
}

# 8 GPU allocation from todo
run_shard 0 42 shard2 9800 &
sleep 25
run_shard 1 42 shard3 9801 &
sleep 25
run_shard 2 43 shard2 9802 &
sleep 25
run_shard 3 43 shard3 9803 &
sleep 25
run_shard 4 44 shard2 9804 &
sleep 25
run_shard 5 44 shard3 9805 &
sleep 25
# Base controls
CUDA_VISIBLE_DEVICES=6 python training/scope_round3/hmin_v2_dup_rollout.py \
  --output-dir "${CL}/base/shard2" \
  --manifest "${MANIFEST}" --shard shard2 --n-shards 4 \
  --model-path "${BASE_MODEL}" --vllm-port 9806 \
  --dup-operation --parallel 1 >> "${LOG}/base_shard2.log" 2>&1 &
sleep 25
CUDA_VISIBLE_DEVICES=7 python training/scope_round3/hmin_v2_dup_rollout.py \
  --output-dir "${CL}/base/shard3" \
  --manifest "${MANIFEST}" --shard shard3 --n-shards 4 \
  --model-path "${BASE_MODEL}" --vllm-port 9807 \
  --dup-operation --parallel 1 >> "${LOG}/base_shard3.log" 2>&1 &

wait
for d in "${CL}"/*/*; do
  [[ -d "$d" ]] && python training/scope_round6/aggregate_round6.py --run-dir "$d" || true
done
python training/scope_round6/build_round6_report.py
touch "${OUT}/PHASE_D_COMPLETE"
scope6_set_stage "done"
scope6_log "Phase D complete"
