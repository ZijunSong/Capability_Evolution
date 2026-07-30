#!/usr/bin/env bash
# B5 — Top-2 + compact_json + Base closed-loop 50q (2×25q shards, wave launch)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round5/_common.sh"
scope5_setup

BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
MERGED_ROOT="${OUT}/merged"
CL_ROOT="${OUT}/closed_loop/b5_50q"
LOG_B5="${LOG_DIR}/b5"
mkdir -p "${MERGED_ROOT}" "${CL_ROOT}" "${LOG_B5}"

# Merge adapters for rollout
merge_if_needed() {
  local name="$1" adapter="$2"
  local merged="${MERGED_ROOT}/${name}"
  if [[ -f "${merged}/config.json" ]]; then return; fi
  scope5_log "[B5 merge] ${name}"
  python training/merge_lora_hf.py \
    --base-model "${BASE_MODEL}" \
    --adapter "${adapter}" \
    --output "${merged}" >> "${LOG_B5}/merge.log" 2>&1
}

# Read Top-2 from B4 gate
TOP1="" TOP2=""
if [[ -f "${OUT}/B4_TOP2" ]]; then
  mapfile -t tops < <(head -2 "${OUT}/B4_TOP2")
  TOP1="${tops[0]:-}"
  TOP2="${tops[1]:-}"
fi
[[ -z "${TOP1}" ]] && TOP1="o7_r64_seed42"
[[ -z "${TOP2}" ]] && TOP2="compact_json_seed42"

merge_if_needed "base" "${BASE_MODEL}"
merge_if_needed "${TOP1}" "${OUT}/b4_full/${TOP1}/adapter"
merge_if_needed "${TOP2}" "${OUT}/b4_full/${TOP2}/adapter"
merge_if_needed "compact_json_seed42" "${OUT}/b4_full/compact_json_seed42/adapter"

run_shard() {
  local gpu=$1 variant=$2 model=$3 port=$4 shard=$5
  local out="${CL_ROOT}/${variant}/${shard}"
  if [[ -f "${out}/summary.json" ]]; then
    scope5_log "[skip] ${variant}/${shard}"
    return 0
  fi
  mkdir -p "${out}"
  scope5_log "[B5 CL] GPU${gpu} ${variant}/${shard} port${port}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" \
    --n-shards 4 \
    --model-path "${model}" \
    --vllm-port "${port}" \
    --dup-operation \
    --parallel 1 >> "${LOG_B5}/cl_${variant}_${shard}.log" 2>&1
}

launch_wave() {
  local -a jobs=("$@")
  local pids=() delay=0
  for job in "${jobs[@]}"; do
    IFS=':' read -r gpu variant model port shard <<< "${job}"
    ( sleep "${delay}"; run_shard "${gpu}" "${variant}" "${model}" "${port}" "${shard}" ) &
    pids+=($!)
    delay=$((delay + 75))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || true; done
}

scope5_log "[B5] wave A (max 4 vLLM cold starts)"
launch_wave \
  "0:base:${BASE_MODEL}:9200:shard0" \
  "1:base:${BASE_MODEL}:9201:shard1" \
  "2:${TOP1}:${MERGED_ROOT}/${TOP1}:9202:shard0" \
  "3:${TOP1}:${MERGED_ROOT}/${TOP1}:9203:shard1"

scope5_log "[B5] wave B"
launch_wave \
  "4:${TOP2}:${MERGED_ROOT}/${TOP2}:9204:shard0" \
  "5:${TOP2}:${MERGED_ROOT}/${TOP2}:9205:shard1" \
  "6:compact_json:${MERGED_ROOT}/compact_json_seed42:9206:shard0" \
  "7:compact_json:${MERGED_ROOT}/compact_json_seed42:9207:shard1"

scope5_log "[B5] closed-loop 50q complete"
