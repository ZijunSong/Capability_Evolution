#!/usr/bin/env bash
# Submit Harness-1 BC+ GPT-OSS-20B Projected Teacher SFT (Harness-Zero-style
# baseline) via the independent SFT v1 stack (NOT verl torch_ddp_lora). 4
# nodes x 8 rollout replicas = 32 cards. Convert Teacher guidance into
# H_min-legal Student behavior: replay the same Full-Harness teacher
# trajectories in a Zero env, keep only student-realizable projected actions,
# SFT on those. Same teacher signal as the Teacher SFT leg but supervised on
# the Zero-legal projection. Action-only CE, NO RL/OPD/gap/gate.
#
# Fork of submit_gpt_oss_20b_only_opd_recall_32gpu_4node_20261015.sh. Only
# changes:
#   * --image_command points at the Projected Teacher SFT entry (forked from
#     the only-opd entry; calls run_projected_teacher_sft_multinode.py).
#   * fresh jobName + fresh OUT dir (do NOT overwrite any prior attempt).
#   * description reflects the Harness-Zero-style baseline purpose.
# Everything else identical to only-opd: bf16-dequant (NOT mxfp4), mb1,
# gpu-mem0.85 rollout, rdma, GUARANTEED_PUBLIC, queue 2334, priority 8,
# tag MemQA-General-V5, alert P2 on FAILED, no retry, HangDetection off.
#
# Ablation alignment: same base (gpt-oss-20b-bf16-dequant), same sec corpus,
# same rl-data, same seed=42, same eval as the only-opd/no-gate/no-project
# suite. NO --init-adapter (fresh lora-rank=32 LoRA, shared seed across the
# three SFT legs). Teacher = base itself.
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
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_projected_teacher_sft_32gpu_4node_20261019.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-projected-teacher-sft-bf16-32gpu-4node-20261019-r9 \
  --description "TRIM BC+ GPT-OSS-20B bf16 4node 32gpu Projected Teacher SFT (Harness-Zero-style baseline). SFT v1 stack (HF+PEFT LoRA+action-only CE, NOT verl/RL/OPD). Replay Full-Harness teacher trajectories in Zero env, keep student-realizable projected actions, SFT on those. teacher=base. mb1 gpu-mem0.85 (bf16 40GB weights+no grad-ckpt 铁律). fresh lora-rank=32 seed=42 (shared across 3 SFT legs). priority8 MemQA-General-V5. Reuses Teacher SFT archive (waits for it). world_size=32 audit." \
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
  --env OUT=/mnt/songzijun/outputs/gpt-oss-20b-projected-teacher-sft-bf16-32gpu_20261019_r9 \
  --env TEACHER_OUT=/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-bf16-32gpu_20261019_r9 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
