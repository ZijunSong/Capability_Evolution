#!/usr/bin/env bash
# Submit GPT-OSS-20B Student on-policy SFT v2 DDP train-phase (2026-10-21).
#
# Recovery/validation run: consume the sealed r000000 collected by 14ztx
# (255 rows, policy digest 23e4175f...), apply ONE 32-card Accelerate DDP
# action-position CE update (global batch=128, mb=1), publish adapter/optimizer/
# scheduler/selection, profile load+CE time + peak VRAM, then exit.
#
# This is NOT the resident online loop — one update + profile, the unit for
# first-round validation and each future step once the loop ships. cf. 14ztx
# which got SIGTERM'd because single rank0 ran 128 MoE backward serially (2.5h,
# 31 idle cards → low utilization killed). The DDP backend splits 128 rows /
# 32 ranks ≈ 4 backward slots/rank, all 32 cards active.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_student_onpolicy_sft_v2_ddp_train_phase_32gpu_4node_20261021.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-student-onpolicy-sft-v2-ddp-trainphase-bf16-32gpu-4node-20261021-r9-pri8 \
  --description "GPT-OSS-20B bf16 4node32gpu Student on-policy SFT v2 DDP train-phase (recovery+profile). Accelerate DDP (c8640e4): ONE 32-rank action-position CE update from sealed r000000 (255 rows, 14ztx collect). gbatch=32 (one row/rank) mb1 bf16. --resume-round 14ztx r000000 + --init-adapter 14ztx adapters/current (digest 23e4175f match). accelerate launch 4x8=32 DDP. Validates DDP backend + profiles load/CE/VRAM. NOT resident online loop. pri8 MemQA-General-V5." \
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
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1/tinker-cookbook:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env OUT=/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-v2-ddp-trainphase-bf16-32gpu_20261021_r9 \
  --env SEALED_ROUND=/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/rounds/r000000 \
  --env INIT_ADAPTER=/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/adapters/current \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
