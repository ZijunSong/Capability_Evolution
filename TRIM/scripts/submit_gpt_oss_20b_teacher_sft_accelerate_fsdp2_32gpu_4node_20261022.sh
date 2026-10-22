#!/usr/bin/env bash
# Submit the FORMAL Teacher SFT accelerate_fsdp2 run (4 nodes x 8 GPUs, 200 steps).
#
# PREREQUISITE: the single-node smoke (t-20261022110759-szfhz or its recovery)
# must have passed — 5 optimizer steps, DCP checkpoint, LoRA export verified.
# Do NOT submit this before that gate; the smoke exists to fail cheaply.
#
# Fresh OUT; grad-ckpt starts `off` (export SFT_CKPT_MODE=non_reentrant for a
# new contract only after the off path is measured). SFT_MAX_MODEL_LEN defaults
# to 4096 to match the sealed-row subset the DDP attempts trained on; switch to
# 8192 only with an explicit new run contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_teacher_sft_accelerate_fsdp2_32gpu_4node.sh"
test -f "${ENTRY}"

OUT_DIR=/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-fsdp2-32gpu-4node-20261022

cml custom_train submit \
  --job_name trim-teacher-sft-fsdp2-32gpu-4node-20261022 \
  --description "TRIM GPT-OSS-20B Teacher SFT accelerate_fsdp2 formal. 4 node 32 gpu, 200 steps, mb1 action-only CE, sealed teacher dataset, common init adapter. FSDP2 shard base + DCP ckpt. priority8 MemQA-V5." \
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
  --env OUT="${OUT_DIR}" \
  --env NNODES=4 \
  --env SFT_RUN_TAG=teacher-fsdp2-20261022 \
  --env SFT_CKPT_MODE="${SFT_CKPT_MODE:-off}" \
  --env SFT_NUM_UPDATES=200 \
  --env SFT_SAVE_EVERY=10 \
  --env SFT_MAX_MODEL_LEN="${SFT_MAX_MODEL_LEN:-4096}" \
  --env SFT_BATCH_SIZE=32 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
