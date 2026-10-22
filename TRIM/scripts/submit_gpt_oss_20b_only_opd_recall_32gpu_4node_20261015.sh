#!/usr/bin/env bash
# Submit Harness-1 BC+ GPT-OSS-20B only-opd (final-answer-recall reward) via
# torch_ddp_lora on FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards = 32 GPUs,
# 200 steps, --train_method only-opd --component all = teacher_full student_zero,
# rl_enabled=False so ONLY the OPD projection distillation cell runs).
#
# Fork of submit_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261002_recall_only_bf16.sh
# (the verified gpt-oss bf16-dequant recall-only 32gpu baseline). Only changes:
#   * --image_command points at the only-opd entry (forked from the 20261002 bf16
#     entry, --train_method only-opd instead of trim/all).
#   * --priority 8 (user-requested; baseline was 2).
#   * --tag MemQA-General-V5 (user-requested).
#   * fresh jobName + fresh OUT dir (do NOT overwrite any prior attempt).
# Everything else is identical: bf16-dequant (NOT mxfp4), mb1, gpu-mem0.85,
# rdma, GUARANTEED_PUBLIC, queue 2334, alert P2 on FAILED, no retry, HangDetection off.
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. NOTE: the
# CloudML JuiceFS schema field is `juiceFsCluster` (there is no literal
# `endpoint` key) — its value "bj9-cloudml" IS the cluster/endpoint identifier.
# image_command read verbatim from the entry script via "$(cat ...)" to avoid
# CLI nested-quote escaping bugs.
#
# NOTE on tensorboard: NOT enabled (--enable_tensorboard is absent), matching
# the 20261002 baseline. The entry's TENSORBOARD_PROBE will gracefully report
# log_path_env=None; if you want the TB UI populated, add --enable_tensorboard
# here and the platform injects $TENSORBOARD_LOG_PATH (the entry already wires
# SummaryWriter + the 5 scalar groups).
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has 32
# optimizer.rank0000..0031.pt; (3) vllm_sessions/ has 32 _r0.._r31 sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_only_opd_recall_32gpu_4node_20261015.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-only-opd-recall-bf16-32gpu-4node-20261015 \
  --description "TRIM Harness-1 BC+ GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora. 200step only-opd (teacher_full student_zero, rl_enabled=False, OPD distill only). REWARD=final-answer-recall EXPLICIT (score_explicit_reward, no env, no logprobs). mb1 gpu-mem0.85 (bf16 40GB weights, CLAUDE.md 20261015 铁律). priority8 MemQA-General-V5. bf16 fixes mxfp4 Marlin PTX+NaN. world_size=32 audit." \
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
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env OUT=/mnt/songzijun/outputs/gpt-oss-20b-only-opd-recall-bf16-32gpu_20261015 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
