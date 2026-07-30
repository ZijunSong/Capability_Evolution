#!/usr/bin/env bash
# B6 — Best objective 3 seeds + compact_json + Base closed-loop 100q
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round5/_common.sh"
scope5_setup

BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
MERGED_ROOT="${OUT}/merged"
CL_ROOT="${OUT}/closed_loop/b6_100q"
LOG_B6="${LOG_DIR}/b6"
mkdir -p "${MERGED_ROOT}" "${CL_ROOT}" "${LOG_B6}"

merge_if_needed() {
  local name="$1" adapter="$2"
  local merged="${MERGED_ROOT}/${name}"
  if [[ -f "${merged}/config.json" ]]; then return; fi
  scope5_log "[B6 merge] ${name}"
  python training/merge_lora_hf.py \
    --base-model "${BASE_MODEL}" \
    --adapter "${adapter}" \
    --output "${merged}" >> "${LOG_B6}/merge.log" 2>&1
}

for seed in 42 43 44; do
  merge_if_needed "o7_r64_seed${seed}" "${OUT}/b4_full/o7_r64_seed${seed}/adapter"
done
merge_if_needed "compact_json_seed42" "${OUT}/b4_full/compact_json_seed42/adapter"

run_shard() {
  local gpu=$1 variant=$2 model=$3 port=$4 shard=$5 n_shards=$6
  local out="${CL_ROOT}/${variant}/${shard}"
  if [[ -f "${out}/summary.json" ]]; then return 0; fi
  mkdir -p "${out}"
  scope5_log "[B6 CL] GPU${gpu} ${variant}/${shard}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" \
    --n-shards "${n_shards}" \
    --model-path "${model}" \
    --vllm-port "${port}" \
    --dup-operation \
    --parallel 1 >> "${LOG_B6}/cl_${variant}_${shard}.log" 2>&1
}

launch_wave() {
  local n_shards=$1; shift
  local -a jobs=("$@")
  local pids=() delay=0
  for job in "${jobs[@]}"; do
    IFS=':' read -r gpu variant model port shard <<< "${job}"
    ( sleep "${delay}"; run_shard "${gpu}" "${variant}" "${model}" "${port}" "${shard}" "${n_shards}" ) &
    pids+=($!)
    delay=$((delay + 75))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || true; done
}

NS=4  # 4×25q = 100q

scope5_log "[B6] wave 1 Base"
launch_wave "${NS}" \
  "0:base:${BASE_MODEL}:9300:shard0" \
  "1:base:${BASE_MODEL}:9301:shard1" \
  "2:base:${BASE_MODEL}:9302:shard2" \
  "3:base:${BASE_MODEL}:9303:shard3"

for seed in 42 43 44; do
  v="best_o7_${seed}"
  m="${MERGED_ROOT}/o7_r64_seed${seed}"
  base_port=$((9400 + (seed - 42) * 10))
  scope5_log "[B6] wave Best-${seed}"
  launch_wave "${NS}" \
    "0:${v}:${m}:$((base_port)):shard0" \
    "1:${v}:${m}:$((base_port+1)):shard1" \
    "2:${v}:${m}:$((base_port+2)):shard2" \
    "3:${v}:${m}:$((base_port+3)):shard3"
done

scope5_log "[B6] wave compact_json"
launch_wave "${NS}" \
  "4:compact_json:${MERGED_ROOT}/compact_json_seed42:9500:shard0" \
  "5:compact_json:${MERGED_ROOT}/compact_json_seed42:9501:shard1" \
  "6:compact_json:${MERGED_ROOT}/compact_json_seed42:9502:shard2" \
  "7:compact_json:${MERGED_ROOT}/compact_json_seed42:9503:shard3"

scope5_log "[B6] closed-loop 100q complete"
