#!/usr/bin/env bash
# Submit TRIM Harness-1 HotpotQA eval (FIX2): Qwen3-4B + step100 LoRA (iowty trim/all).
# component=zero, 8gpu. FIX2: also inject gold_docids into dataset._gold_document_ids so
# evaluate_results_final_answer_recall resolves (BrowseCompPlusDataset reads a separate
# _gold_document_ids dict, not _query_index). Builds on FIX1 (query_index + ::c strip).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_adapter_step100_9ixnl_bcplus_transfer_eval_8gpu.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name trim-harness1-hotpotqa-qwen3-4b-adapter-step100-9ixnl-eval-8gpu-20260930 \
  --description "TRIM hotpotqa eval FIX2: Qwen3-4B step100 LoRA 9ixnl rl/zero. component=zero 8gpu 493q. PATCH inject _gold_document_ids (fix2 in JuiceFS code tree) so final_answer_recall resolves."
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
  --env BENCHMARK=hotpotqa \
  --env RUN_ID=qwen3_4b_adapter_step100_9ixnl_hotpotqa_8gpu_20260930_fix2_memqa_v5 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
