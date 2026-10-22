#!/usr/bin/env bash
# Submit TRIM Harness-1 bcplus_full eval: Qwen3-4B-Instruct-2507 + step100 LoRA (h3eb8 reward=recall-only rl/zero 200step).
# INDEPENDENT EVAL SEED=2024 (multi-seed averaging; seed42 already done as group 1, this is group 3 of 3).
# component=zero, 8gpu. 8 single-GPU vLLM actors (TP=1) + base+LoRA, local_bm25.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_adapter_h3eb8_recall_only_transfer_eval_8gpu_seed2024.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name trim-harness1-bcplus_full-qwen3-4b-adapter-step100-h3eb8-recall-only-eval-8gpu-seed2024-20261015 \
  --description "TRIM bcplus_full eval: Qwen3-4B-Instruct-2507 step100 LoRA (h3eb8 reward=recall-only rl/zero 200step). component=zero 8gpu. Independent eval seed=2024 (2nd of 3-seed avg; seed42 done, seed123/2024 new). 8 single-GPU vLLM actors base+LoRA local_bm25." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 5 \
  --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* \
  --env CHROMA_API_KEY=local-offline \
  --env CHROMA_DATABASE=local-offline \
  --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env ADAPTER_PATH=/mnt/songzijun/Capability_Evolution/TRIM/outputs/qwen3-4B-rl-only-recall-32gpu_20260928/checkpoints/rl/step_000100/adapter \
  --env BENCHMARK=bcplus_full \
  --env EVAL_SEED=2024 \
  --env RUN_ID=qwen3_4b_adapter_step100_h3eb8_recall_only_bcplus_full_8gpu_20261015_seed2024_memqa_v5 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
