#!/usr/bin/env bash
# Submit TRIM Harness-1 transfer-benchmark ADAPTER eval (Qwen3-4B + 9ixnl LoRA) on one 8-GPU gm402-80g node.
# Parameterized by env: BENCHMARK (longsealqa|frames|hotpotqa), ADAPTER (step100|step200), RUN_ID, JOB_NAME.
# Usage: BENCHMARK=longsealqa ADAPTER=step200 bash submit_qwen3_4b_transfer_9ixnl_eval_8gpu.sh
set -euo pipefail

BENCHMARK="${BENCHMARK:?set BENCHMARK=longsealqa|frames|hotpotqa}"
ADAPTER="${ADAPTER:?set ADAPTER=step100|step200}"
STEP=$([ "$ADAPTER" = "step200" ] && echo step_000200 || echo step_000100)
ADAPTER_PATH_ENV="/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_qwen3_4b_rl_ddp_200step_zero_32gpu_4node_20260925/checkpoints/rl/${STEP}/adapter"
RUN_ID_ENV="${RUN_ID:-qwen3_4b_transfer_9ixnl_${ADAPTER}_${BENCHMARK}_8gpu_20260928_memqa_v5}"
JOB_NAME="${JOB_NAME:-trim-harness1-bcplus-full-qwen3-4b-transfer-${ADAPTER}-${BENCHMARK}-eval-8gpu-20260928}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_transfer_9ixnl_eval_8gpu.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name "${JOB_NAME}" \
  --description "TRIM Harness-1 transfer eval: Qwen3-4B-Instruct-2507 + 9ixnl ${ADAPTER} LoRA on ${BENCHMARK} (local_bm25 transfer_local corpus, COMPONENT=zero, max-turns 40, gpu-mem 0.90). 8 single-GPU vLLM actors + 32 API workers." \
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
  --env BENCHMARK="${BENCHMARK}" \
  --env ADAPTER_PATH="${ADAPTER_PATH_ENV}" \
  --env RUN_ID="${RUN_ID_ENV}" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
