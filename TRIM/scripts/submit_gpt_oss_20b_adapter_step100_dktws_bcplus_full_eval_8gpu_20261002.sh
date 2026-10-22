#!/usr/bin/env bash
# Eval GPT-OSS-20B step100 LoRA (dktws trim/all 200step 4node run2 bf16, 100-ckpt)
# on benchmark BrowseComp+. bf16-dequant base + LoRA adapter, 8gpu, local_bm25,
# COMPONENT=zero, fix3 gold_document_ids patch (harness1_api_worker.py:144-176)
# so final_answer_recall resolves. Fresh jobName/RUN_ID (7nzzn eval preserved).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_adapter_step100_bcplus_transfer_eval_8gpu.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name trim-harness1-bcplus_full-gpt-oss-20b-adapter-step100-dktws-eval-8gpu-20261002 \
  --description "TRIM BrowseComp+ eval: GPT-OSS-20B step100 LoRA (dktws trim/all 200step 4node run2 bf16 step100). bf16-dequant base+LoRA. 8gpu TP1 actors enable_lora. local_bm25 COMPONENT=zero. fix3 gold_document_ids patch. Avoid mxfp4+LoRA PTX." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* \
  --env CHROMA_API_KEY=local-offline \
  --env CHROMA_DATABASE=local-offline \
  --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env ADAPTER_PATH=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_ddp_200step_all_32gpu_4node_20260925_run2_bf16/checkpoints/scape_seed/step_000100/adapter \
  --env BENCHMARK=bcplus_full \
  --env RUN_ID=gpt_oss_20b_adapter_step100_dktws_bcplus_full_8gpu_20261002_memqa_v5 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
