#!/usr/bin/env bash
# Submit Harness-1 BC+ GPT-OSS-20B Student on-policy SFT (on-policy-data vs
# OPD+RL gain control) via the independent SFT v1 stack (NOT verl torch_ddp_lora).
# 4 nodes x 8 rollout replicas = 32 cards. Each round rolls the CURRENT student
# (component=zero) on fresh states, labels decision points with TRIM-legal
# branches, one action-level CE update. No teacher at all. Critical control:
# if this alone matches only-opd/trim gains, the lift is on-policy data, not
# the OPD+RL objective. Action-only CE, NO RL/OPD/gap/gate.
#
# Fork of submit_gpt_oss_20b_only_opd_recall_32gpu_4node_20261015.sh. Only
# changes:
#   * --image_command points at the Student on-policy SFT entry (forked from
#     the only-opd entry; calls run_student_onpolicy_sft_multinode.py).
#   * fresh jobName + fresh OUT dir (do NOT overwrite any prior attempt).
#   * description reflects the on-policy vs OPD+RL control purpose.
# Everything else identical to only-opd: bf16-dequant (NOT mxfp4), mb1,
# gpu-mem0.85 rollout, rdma, GUARANTEED_PUBLIC, queue 2334, priority 8,
# tag MemQA-General-V5, alert P2 on FAILED, no retry, HangDetection off.
#
# Ablation alignment: same base (gpt-oss-20b-bf16-dequant), same sec corpus,
# same rl-data, same seed=42, same eval as the only-opd/no-gate/no-project
# suite. NO --init-adapter (fresh lora-rank=32 LoRA, shared seed across the
# three SFT legs). NO teacher (student self-rollout each round).
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. image_command
# read verbatim from the entry script via "$(cat ...)" to avoid nested-quote
# escaping bugs.
#
# NOTE: tensorboard NOT enabled (matching only-opd baseline). The entry's
# TENSORBOARD_PROBE will report log_path_env=None gracefully.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_student_onpolicy_sft_32gpu_4node_20261019.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-student-onpolicy-sft-bf16-32gpu-4node-20261019-r3 \
  --description "TRIM Harness-1 BC+ GPT-OSS-20B bf16-dequant 4node 32gpu Student on-policy SFT (on-policy vs OPD+RL gain control). Independent SFT v1 stack (HF+PEFT LoRA+action-only CE, NOT verl/RL/OPD). Each round student=zero self-rollout fresh states, TRIM-legal labels, one CE update. No teacher. mb1 micro-batch gpu-mem0.85 (bf16 40GB weights + no grad-ckpt, CLAUDE.md 铁律). fresh lora-rank=32 seed=42 (shared across 3 SFT legs, only variable=supervision source). priority8 MemQA-General-V5. world_size=32 audit." \
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
  --env OUT=/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_r3 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
