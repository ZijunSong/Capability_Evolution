#!/usr/bin/env bash
# Submit Harness-1 BC+ full RL (CISPO) Qwen3-4B-Instruct-2507 via torch_ddp_lora
# backend on one 8-GPU gm402-80g node (200 steps). Pure CLI flags (no --filename
# YAML); JuiceFS mount passed explicitly via --juicefs_mount_configs JSON with
# juiceFsCluster=bj9-cloudml (endpoint). image_command read verbatim from the
# entry script via "$(cat ...)" to avoid CLI nested-quote escaping bugs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_rl_ddp_200step_8gpu.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-qwen3-4b-rl-ddp-200step-8gpu \
  --description "Harness-1 BC+ full RL (CISPO) Qwen3-4B-Instruct-2507 on one 8-GPU gm402-80g node via torch_ddp_lora backend (per-rank vLLM TP=1 rollout + DDP LoRA CISPO step). 200 steps, 32 groups/step, group-size 8, max-turns 40, max-model-len 8192, gpu-mem-util 0.90, local_legacy BM25, SEC data under TRIM/data, offline tiktoken, four-pack runtime locked in preflight. CloudML WORLD_SIZE=1 skip fixed in 6e8dea4; 8 ranks truly start (world_size=8)." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 8 \
  --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env SCAPE_FORCE_LOCAL_HARMONY=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
