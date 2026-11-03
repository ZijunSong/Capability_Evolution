#!/usr/bin/env bash
# Eval GPT-OSS-20B step100 LoRA (no-project recall 200step 4node bf16,
# Job t-20261015163939-2xsbm, only_opd cell step_000100) on benchmark bcplus_full.
# bf16-dequant base + LoRA adapter, 8gpu, local_bm25, COMPONENT=zero,
# EVAL_SEED=42 (3-seed independent eval, h3eb8 口径 42/123/2024).
# Fork of submit_gpt_oss_20b_adapter_step100_pncda_*_seed123.sh template.
# fix3 gold_document_ids patch (harness1_api_worker.py:144-176) so
# final_answer_recall resolves. Fresh jobName/RUN_ID per seed×bench.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_adapter_step100_bcplus_transfer_eval_8gpu_seed.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name trim-harness1-bcplus_full-gpt-oss-20b-adapter-step100-projected-teacher-eval-hotpotqa-seed42 \
  --description "TRIM bcplus_full eval: GPT-OSS-20B step100 LoRA (projected teacher SFT v2 twcdt r4 step100 zero component cell=zero). bf16-dequant base+LoRA. 8gpu TP1 actors enable_lora. local_bm25 COMPONENT=zero EVAL_SEED=42. fix3 gold_document_ids patch. Avoid mxfp4+LoRA PTX." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 5 --preemptible=false \
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
  --env EVAL_SEED=42 \
  --env ADAPTER_PATH=/mnt/songzijun/outputs/gpt-oss-20b-projected-teacher-sft-bf16-32gpu_20261021_v2_train_r4/checkpoints/step-000100/adapter \
  --env BENCHMARK=hotpotqa \
  --env RUN_ID=gpt_oss_20b_adapter_step100_projected_teacher_hotpotqa_8gpu_20261023_seed42 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
