#!/usr/bin/env bash
# Submit Harness-1 BC+ trim (SR-OPD/CISPO distill, --train_method trim
# --component all) Qwen3-4B-Instruct-2507 (dense bf16 4B) via torch_ddp_lora
# backend on FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards = 32 GPUs, 200 steps).
# REWARD = final-answer recall ONLY + TENSORBOARD ON.
#
# Fork of submit_qwen3_4b_trim_ddp_200step_all_32gpu_4node_20260925.sh (iowty,
# the verified Qwen3-4B trim/all 32gpu 4node baseline) merged with the three
# h3eb8 differences: (1) reward in trim/training/four_cell_runtime.py's
# terminal_reward_breakdown is `total = rec` (final-answer recall ONLY; legacy
# 0.80*recall + 0.10*legal + <=0.10*shaping + -0.2 invalid penalty dropped;
# diagnostic fields still emitted); (2) --enable_tensorboard is passed so the
# platform injects $TENSORBOARD_LOG_PATH and trainer_adapter.py's
# _make_tensorboard_writer writes per-step RL metrics (reward/loss/cispo/tokens
# /timing/mem + 5 scalar groups + jsonl mirror fallback) to a tensorboard UI;
# (3) new optimization/audit metrics (entropy/approx_kl/grad_norm/advantage_std
# /success_rate/selected_ratio) asserted in entry PATCH_CHECK. Everything else
# is byte-identical to iowty (mb4/gpu-mem0.90/max-turns40/max-model-len8192/
# SCAPE_FORCE_LOCAL_HARMONY/4 DDP patches incl curate-clipping).
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. The CloudML
# JuiceFS schema field is `juiceFsCluster` (no literal `endpoint` key); its
# value "bj9-cloudml" IS the cluster/endpoint identifier (verified by 7nzzn
# describe juiceFSCluster field). image_command read verbatim from the entry
# script via "$(cat ...)" to avoid CLI nested-quote escaping bugs.
#
# === GPU contention — check at submit time ===
# Queue 2334 cloudml.ng1gm402-8-8.20-224 RESOURCEFREE varies as 32-GPU gangs
# start/stop. A 32-GPU gang (4 nodes x 8 cards) may queue if the pool is busy.
# If queue RESOURCEFREE < 32, expect FailedQueueScheduling "insufficient
# resource ... left X, request 32"; it dequeues once a gang releases. Submit
# is fine; the job waits in queue. Verify with: cml custom_train describe <jobId>.
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes
# 4 --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has the DDP rank0
# merged optimizer.pt (DDP path = single file, NOT FSDP2 shards); (3)
# vllm_sessions/ has 32 _r0.._r31 sessions.
#
# Usage:
#   bash /mnt/songzijun/Capability_Evolution/TRIM/scripts/submit_qwen3_4b_trim_all_recall_only_32gpu_4node_20260930.sh
# To dry-run (echo the cml command without submitting), pass DRY=1:
#   DRY=1 bash submit_qwen3_4b_trim_all_recall_only_32gpu_4node_20260930.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_trim_all_recall_only_32gpu_4node_20260930.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness1-bcplus-full-qwen3-4b-trim-all-recall-only-32gpu-4node-20260930"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/qwen3-4B-trim-all-recall-only-32gpu_20260930"
# Description MUST be <=500 chars (CloudML rejects >500 with
# "描述信息不能超过500字符"; h3eb8 hit this). Keep it terse.
DESCRIPTION="TRIM Harness-1 BC+ trim (train_method=trim component=all) Qwen3-4B-Instruct-2507 dense bf16 4node 32gpu torch_ddp_lora 200step. REWARD=final-answer-recall ONLY (terminal_reward_breakdown total=curated_recall). TENSORBOARD ON (--enable_tensorboard; trainer writes 5 groups + jsonl mirror). mb4 gpu-mem0.90 max-turns40 max-model-len8192. curate-clipping patch asserted. SCAPE_FORCE_LOCAL_HARMONY=1. 4 DDP patches. world_size=32 audit."

# Defensive: refuse to submit if description exceeds 500 chars.
DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description is ${DESC_LEN} chars (>500). CloudML will reject it. Trim it." >&2
  exit 1
fi
echo "DESCRIPTION_LEN ${DESC_LEN} (<=500 ok)"

CMD=(
  cml custom_train submit
  --job_name "${JOB_NAME}"
  --description "${DESCRIPTION}"
  --access_type PRIVATE
  --tag MemQA-General-V5
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
  --framework pytorch
  --queue_id 2334
  --priority 8
  --preemptible=false
  --enable_tensorboard
  --rdma
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'
  --resource_configs '[{"nodeRole":"worker","nodeNumber":4,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]'
  --env PYTHONUNBUFFERED=1
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  --env TRIM_GPU_KEEPALIVE=0
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
  --env OPENAI_API_KEY=**-****-***********-***-*-****-***
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
  --env SCAPE_FORCE_LOCAL_HARMONY=1
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
  --env OUT="${OUT}"
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}'
  --diag_options '[{"name":"HangDetection","enable":false}]'
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}'
  --image_command "$(cat "${ENTRY}")"
)

if [ "${DRY:-0}" = "1" ]; then
  echo "[dry-run] would execute:"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

exec "${CMD[@]}"
