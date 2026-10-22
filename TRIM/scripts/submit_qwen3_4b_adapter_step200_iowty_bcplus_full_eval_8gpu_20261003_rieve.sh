#!/usr/bin/env bash
# Submit TRIM Harness-1 BCPLUS_FULL eval: Qwen3-4B + step200 LoRA (iowty trim/all).
# COMPONENT=auto_populate_first_search,chunk_neighbors (Retrieval Initiation & Expansion) — 精确 v8d mask
#   (这两个 ON，其余 8 个高级组件 OFF)。与 zero student-口径 baseline 对比组件策略效果。
# 8gpu, gpu-mem0.90, temp=1.0, max-turns40. Patches: worker inject gold_docids + strip ::c (fix3 inherited).
# priority=2 (用户降级). RUN_ID _rieve 后缀避 eval 脏目录复用坑 (assert_fresh_eval_dir).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_adapter_step200_iowty_bcplus_transfer_eval_8gpu_rieve.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name trim-harness1-bcplus-full-qwen3-4b-adapter-step200-iowty-eval-8gpu-20261003-rieve \
  --description "TRIM BCPLUS_FULL eval: Qwen3-4B step200 LoRA iowty trim/all. COMPONENT=auto_populate_first_search,chunk_neighbors (Retrieval Initiation & Expansion) 8gpu 830q temp1.0 max-turns40 gpu-mem0.90. Compare vs zero component baseline." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 2 \
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
  --env BENCHMARK=bcplus_full \
  --env RUN_ID=qwen3_4b_adapter_step200_iowty_bcplus_full_8gpu_20261003_rieve_memqa_v5 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
