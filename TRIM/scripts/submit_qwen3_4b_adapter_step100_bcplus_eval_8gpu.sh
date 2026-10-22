#!/usr/bin/env bash
# Submit TRIM Harness-1 BC+ full ADAPTER eval (Qwen3-4B + step100 LoRA) on one 8-GPU gm402-80g node.
# Evaluates the trained adapter from t-20260920095812-lxb0w (100step torch_ddp_lora RL, completed 9/22 05:04).
# Pure CLI flags (no --filename); JuiceFS via --juicefs_mount_configs JSON (juiceFsCluster=bj9-cloudml).
# image_command read verbatim from entry script via "$(cat ...)" to avoid CLI nested-quote escaping.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_adapter_step100_bcplus_eval_8gpu.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-qwen3-4b-adapter-step100-eval-8gpu-recov2 \
  --description "TRIM Harness-1 BC+ full ADAPTER eval: Qwen3-4B-Instruct-2507 + step100 LoRA (t-...lxb0w torch_ddp_lora RL 100step, completed 9/22). 8 single-GPU vLLM actors (TP=1) load base+LoRA (enable_lora + lora-modules), 32 API-client workers, local_bm25, COMPONENT=zero no verifier, max-turns 40, gpu-mem-util 0.90. Compare vs zero baseline (Recall 7.51%)." \
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
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* \
  --env CHROMA_API_KEY=local-offline \
  --env CHROMA_DATABASE=local-offline \
  --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
