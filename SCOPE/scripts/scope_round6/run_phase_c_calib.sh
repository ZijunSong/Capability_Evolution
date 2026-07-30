#!/usr/bin/env bash
# Phase C-CALIB: calibrate thresholds + 25q shard1 prospective
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round6/_common.sh"
scope6_setup

CAL="${OUT}/calibration"
CL="${OUT}/closed_loop/calib_25q"
LOG="${LOG_DIR}/phase_c_calib"
MERGED="${R5}/merged"
mkdir -p "${CAL}" "${CL}" "${LOG}"

scope6_log "Calibrating thresholds on shard0"
python training/scope_round6/calibrate_threshold.py --gpu cuda:0

THR_JSON="${CAL}/thresholds.json"
TAU42=$(python -c "import json; d=json.load(open('${THR_JSON}')); print(d['tau_seed42'])")
TAU43=$(python -c "import json; d=json.load(open('${THR_JSON}')); print(d['tau_seed43'])")
TAU44=$(python -c "import json; d=json.load(open('${THR_JSON}')); print(d['tau_seed44'])")
TAU_SHARED=$(python -c "import json; d=json.load(open('${THR_JSON}')); print(d['tau_shared'])")

run_cl() {
  local gpu=$1 seed=$2 tau=$3 tag=$4 port=$5
  local model="${MERGED}/o7_r64_seed${seed}"
  local out="${CL}/${tag}/seed${seed}"
  if [[ -f "${out}/aggregated_metrics.json" ]]; then return 0; fi
  mkdir -p "${out}"
  scope6_log "CL ${tag} seed${seed} tau=${tau} GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard shard1 --n-shards 4 \
    --model-path "${model}" \
    --vllm-port "${port}" \
    --dup-operation \
    --decision-threshold "${tau}" \
    --dup-seed "${seed}" \
    --checkpoint-label "o7_r64_seed${seed}" \
    --parallel 1 >> "${LOG}/cl_${tag}_seed${seed}.log" 2>&1
  python training/scope_round6/aggregate_round6.py --run-dir "${out}"
}

# GPU assignment per todo
run_cl 0 42 "${TAU42}" "per_seed" 9700 &
sleep 25
run_cl 1 43 "${TAU43}" "per_seed" 9701 &
sleep 25
run_cl 2 44 "${TAU44}" "per_seed" 9702 &
sleep 25
run_cl 3 42 "${TAU_SHARED}" "shared" 9703 &
sleep 25
run_cl 4 43 "${TAU_SHARED}" "shared" 9704 &
sleep 25
run_cl 5 44 "${TAU_SHARED}" "shared" 9705 &
sleep 25
# Base control GPU6
CUDA_VISIBLE_DEVICES=6 python training/scope_round3/hmin_v2_dup_rollout.py \
  --output-dir "${CL}/base_control" \
  --manifest "${MANIFEST}" \
  --shard shard1 --n-shards 4 \
  --model-path "${BASE_MODEL}" \
  --vllm-port 9706 \
  --dup-operation --parallel 1 >> "${LOG}/cl_base.log" 2>&1 &
sleep 25
# Worst seed control GPU7 (seed43 default)
run_cl 7 43 0.0 "threshold_zero" 9707 &

wait
touch "${OUT}/PHASE_C_CALIB_COMPLETE"
scope6_set_stage "phase_c_calib_done"
scope6_log "Phase C-CALIB complete"
