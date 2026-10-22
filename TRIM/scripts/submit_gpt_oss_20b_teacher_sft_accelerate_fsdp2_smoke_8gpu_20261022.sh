#!/usr/bin/env bash
# Submit the FIRST accelerate_fsdp2 smoke for Teacher SFT (2026-10-22).
#
# Scope per TRIM_SFT_最新运行代码诊断与FSDP2改造指导.md §2/§14/§17 batch 1:
# single node x 8 GPUs (NOT 4x8 yet), checkpointing off, 5 optimizer steps on
# the existing sealed Teacher dataset. Proves: FSDP2 load + shard assertion,
# weighted action CE backward, DCP sharded checkpoint, LoRA export, one resume
# step. Only after this passes, the formal 4x8 200-step run goes to a NEW OUT.
#
# Fresh OUT every attempt (assert_fresh / stale-ckpt rules): OUT name embeds
# the date and smoke tag. Failures keep their logs for audit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_teacher_sft_accelerate_fsdp2_32gpu_4node.sh"
test -f "${ENTRY}"

OUT_DIR=/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-fsdp2-smoke-8gpu-20261022_r3

cml custom_train submit \
  --job_name trim-teacher-sft-fsdp2-smoke-8gpu-20261022-r3 \
  --description "TRIM GPT-OSS-20B Teacher SFT accelerate_fsdp2 smoke. 1 node 8 gpu, 5 steps, grad-ckpt off, mb1 action-only CE on sealed teacher dataset. Verifies FSDP2 shard load + DCP ckpt + LoRA export before 4x8 formal. priority8 MemQA-V5." \
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
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1/tinker-cookbook:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env OUT="${OUT_DIR}" \
  --env NNODES=1 \
  --env NODE_RANK=0 \
  --env MASTER_ADDR=127.0.0.1 \
  --env MASTER_PORT=29631 \
  --env SFT_RUN_TAG=teacher-fsdp2-smoke-20261022 \
  --env SFT_CKPT_MODE=non_reentrant \
  --env SFT_NUM_UPDATES=5 \
  --env SFT_SAVE_EVERY=1 \
  --env SFT_MAX_MODEL_LEN=4096 \
  --env SFT_BATCH_SIZE=32 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
