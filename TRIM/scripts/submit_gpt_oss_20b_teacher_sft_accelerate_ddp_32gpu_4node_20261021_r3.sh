#!/usr/bin/env bash
# Submit Harness-1 BC+ GPT-OSS-20B Teacher SFT — accelerate_ddp train-only.
#
# Reuses the already-collected Teacher SFT rows (t-20261020164236-7rk9c, after
# the _action_of codec fix; 192261 legal rows sealed into sft_sealed/ by
# scripts/seal_teacher_sft_dataset.py). Trains with the new accelerate_ddp CE
# backend (trim.training.sft_engine.fixed_runtime.run_fixed_sft): DDP 32 ranks
# step together, replacing the legacy mb1 single-rank serial ce_update that
# got SIGTERM-killed by the GPU-util watchdog mid first CE step.
#
# init-adapter = the fresh LoRA the Student on-policy leg saved at step 0
# (true_p5/adapters/current) — untrained LoRA init keeps the ablation's
# fresh-LoRA-shared-seed-42 contract (only variable = supervision signal source).
#
# Fork of submit_gpt_oss_20b_teacher_sft_32gpu_4node_20261019_true.sh. Only changes:
#   * --image_command points at the train-only accelerate_ddp entry.
#   * fresh jobName + fresh OUT dir.
#   * description reflects accelerate_ddp + sealed data.
# Everything else identical: bf16-dequant (NOT mxfp4), mb1, GUARANTEED_PUBLIC,
# queue 2334, priority 8, tag MemQA-General-V5, alert P2 on FAILED, no retry,
# HangDetection off.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_teacher_sft_accelerate_ddp_32gpu_4node_20261021.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-teacher-sft-bf16-32gpu-4node-20261021-accelerate-ddp-r3 \
  --description "TRIM GPT-OSS-20B bf16 4node32gpu Teacher SFT accelerate_ddp train-only r3. Reuses sealed dataset (192261 rows). Fix r3: max_model_len 8192->6144 (data max 5757<6144, 0 loss; reduce forward activations to avoid single-card OOM at 73.93GB/80GB). r2 OOM at first step forward. global_step bug fixed (r2 reached microbatch_start step=1). DDP 32 ranks. init-adapter fresh LoRA Student step0. mb1 bf16. priority8 MemQA-V5." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 8 \
  --preemptible=false \
  --rdma \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":4,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1/tinker-cookbook:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env OUT=/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-accelerate-ddp_20261021_r3 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
