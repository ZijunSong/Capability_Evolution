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
  --job_name trim-harness1-gpt-oss-20b-teacher-sft-r8-step200-longsealqa-eval-8gpu-20261022-seed42 \
  --description "TRIM longsealqa eval: GPT-OSS-20B Teacher SFT r8 step200 LoRA (accelerate_ddp 200step grad-ckpt, Job g8xxa). bf16-dequant base+LoRA. 8gpu TP1 COMPONENT=zero EVAL_SEED=42. fix3 gold_document_ids. 1-seed." \
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
  --env ADAPTER_PATH=/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-accelerate-ddp_20261021_r8/checkpoints/step-000200/adapter \
  --env BENCHMARK=longsealqa \
  --env RUN_ID=gpt_oss_20b_teacher_sft_r8_step200_longsealqa_8gpu_20261022_memqa_v5_seed42 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
