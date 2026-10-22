#!/usr/bin/env bash
# Submit Harness-1 BC+ full RL (CISPO, H_zero baseline, --train_method rl
# --component zero) Qwen3.5-27B (dense bf16 27B, Qwen3_5ForConditionalGeneration)
# via torch_ddp_lora backend on FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards =
# 32 GPUs, 200 steps) with --rl-reward final-answer-recall (EXPLICIT path,
# score_explicit_reward, pure fa-recall metric, no env/no logprobs).
#
# Fork of submit_qwen3_4b_rl_ddp_200step_zero_32gpu_4node_20260925.sh (the
# verified 9ixnl Qwen3-4B rl/zero multi-node baseline). Changes vs baseline:
#   * --image_command points at the 27B entry (forked from 9ixnl, model=
#     Qwen3.5-27B + --rl-reward final-answer-recall + mb2 + gpu-mem0.80 +
#     ARCH_OK Qwen3_5 ModelRegistry probe).
#   * fresh jobName + fresh OUT dir (do NOT overwrite any prior attempt).
#   * --priority 8 (same as 9ixnl training tasks; higher than wufak pri5 to
#     schedule the 32-GPU gang ahead of it).
# Everything else is identical: dense bf16 (NOT mxfp4), torch_ddp_lora,
# rdma, GUARANTEED_PUBLIC, queue 2334, alert P2 on FAILED, no retry,
# HangDetection off.
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. NOTE: the
# CloudML JuiceFS schema field is `juiceFsCluster` (there is no literal
# `endpoint` key) — its value "bj9-cloudml" IS the cluster/endpoint identifier.
# image_command read verbatim from the entry script via "$(cat ...)" to avoid
# CLI nested-quote escaping bugs.
#
# === GPU contention (2026-10-19) — check at submit time ===
# Queue 2334 cloudml.ng1gm402-8-8.20-224 RESOURCEFREE=36 at submit time, but
# wufak (pri5 lambda_opd=1.0 r2) holds 32 cards. A 32-GPU gang (4 nodes x 8
# cards) may queue if the pool is busy. If queue RESOURCEFREE < 32, expect
# FailedQueueScheduling "insufficient resource ... left X, request 32"; it
# dequeues once a gang releases. Submit is fine; the job waits in queue.
# Verify with: cml custom_train describe <jobId> (state + events
# Scheduled/Unschedulable/FailedQueueScheduling).
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has 32
# optimizer.rank0000..0031.pt; (3) vllm_sessions/ has 32 _r0.._r31 sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen35_27b_rl_ddp_200step_zero_recall_32gpu_4node_20261019.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-qwen35-27b-rl-ddp-200step-zero-recall-32gpu-4node-20261019 \
  --description "TRIM BC+ rl/zero (H_zero RL) Qwen3.5-27B dense bf16 27B 4node32gpu torch_ddp_lora 200step. REWARD=final-answer-recall EXPLICIT (score_explicit_reward, pure metric, no env/logprobs, fed to CISPO backward). mb2 gpu-mem0.90 max-turns40 max-model-len8192 (27B dense grad-ckpt, peak~76GB; 0.90=giveKV20GB rollout faster, vLLM shutdown in backward so 0-helpsOOM). SCAPE_FORCE_LOCAL_HARMONY=1. rl/zero no curate projection so 3 DDP patches. world_size=32 audit." \
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
  --env SCAPE_FORCE_LOCAL_HARMONY=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env OUT=/mnt/songzijun/outputs/qwen35-27b-rl-ddp-200step-zero-recall-32gpu-4node_20261019 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
