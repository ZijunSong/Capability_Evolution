#!/usr/bin/env bash
# Submit TRIM Harness-1 bcplus_full eval: Qwen3-4B-Instruct-2507 + step100 LoRA (iowty trim/all 产物).
# COMPONENT=subtractive_curation,importance_tagging,adaptive_rerank_instruction (Evidence Selection & Prioritization
#   三件套：减性筛选+重要性标注+自适应重排). 非 all 不启 verifier, 8 卡全当 LoRA actor, gpu-mem0.90, temp=1.0, max-turns40.
# BENCHMARK=bcplus_full — wrapper 内置 transfer benchmark 路由. priority=2.
# Pure CLI flags; JuiceFS via --juicefs_mount_configs JSON (juiceFsCluster=bj9-cloudml).
# image_command read verbatim from entry via "$(cat ...)".
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_adapter_step100_iowty_esp_8gpu.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name trim-harness1-bcplus_full-qwen3-4b-adapter-step100-iowty-esp-eval-8gpu-20261003 \
  --description "TRIM Harness-1 bcplus_full eval: Qwen3-4B + step100 LoRA (iowty trim/all 200step 4node). COMPONENT=subtractive_curation,importance_tagging,adaptive_rerank_instruction (ESP 三件套), 8gpu, gpu-mem0.90, temp=1.0, max-turns40. local_bm25. priority=2." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 2 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* --env CHROMA_API_KEY=local-offline --env CHROMA_DATABASE=local-offline --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env BENCHMARK=bcplus_full \
  --env RUN_ID=qwen3_4b_adapter_step100_iowty_esp_bcplus_full_8gpu_20261003_memqa_v5 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
