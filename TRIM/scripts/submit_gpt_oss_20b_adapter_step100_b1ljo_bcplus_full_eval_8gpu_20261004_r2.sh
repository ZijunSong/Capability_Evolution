#!/usr/bin/env bash
# Eval GPT-OSS-20B step100 LoRA (b1ljo rl/zero recall-only 200step 4node bf16, 100-ckpt)
# on benchmark BrowseComp+. RECOVER from rank19 transport-error (GPU3 actor port 8046 died
# mid-eval, 26 queries all format_error_final FA recall=0). Fresh RUN_ID + output dir to
# avoid assert_fresh_eval_dir. Same config otherwise.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_adapter_step100_bcplus_transfer_eval_8gpu.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name trim-harness1-bcplus_full-gpt-oss-20b-adapter-step100-b1ljo-eval-8gpu-20261004-r2 \
  --description "TRIM BrowseComp+ eval r2 (recover rank19 transport error): GPT-OSS-20B step100 LoRA (b1ljo rl/zero recall-only bf16 step100). bf16-dequant base+LoRA. 8gpu TP1 actors. local_bm25 COMPONENT=zero. fix3 gold_document_ids patch. Fresh RUN_ID/output dir." \
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
  --env ADAPTER_PATH=/mnt/songzijun/outputs/gpt-oss-20b-rl-recall-only-bf16-32gpu_20261002/checkpoints/rl/step_000100/adapter \
  --env BENCHMARK=bcplus_full \
  --env RUN_ID=gpt_oss_20b_adapter_step100_b1ljo_bcplus_full_8gpu_20261004_memqa_v5_r2 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
