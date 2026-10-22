#!/usr/bin/env bash
# Submit Harness-1 BC+ TRIM (OPD) GPT-OSS-20B (original mxfp4, non-dequant per
# user) via torch_ddp_lora backend on FOUR 8-GPU gm402-80g nodes (4 nodes x 8
# cards = 32 GPUs, 100 steps, --component all = H_full, all advanced components
# ON). --train_method trim (NOT rl).
#
# This is the TRIM/ALL 100-step MULTI-NODE variant. Derived from
# submit_gpt_oss_20b_rl_ddp_100step_zero_32gpu_4node_20260925.sh (the 100-step
# RL/ZERO 4-node job). Deltas vs that submit:
#   * job_name ...rl-ddp-100step-zero... -> ...trim-ddp-100step-all...
#   * --description: RL/ZERO -> TRIM/ALL, note curate-clipping patch required
#   * --env OUT: points at the trim/all output dir (not the rl/zero one)
#   * ENTRY points at the trim/all entry (with curate-clip patch assertions)
# Everything else identical: 4 nodes x 8 gm402-80g, RDMA on, priority 8,
# tag MemQA-General-V5, queue 2334, framework pytorch, JuiceFS juiceFsCluster
# bj9-cloudml, image_command via $(cat entry).
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. NOTE: the
# CloudML JuiceFS schema field is `juiceFsCluster` (there is no literal
# `endpoint` key) — its value "bj9-cloudml" IS the cluster/endpoint identifier.
# image_command read verbatim from the entry script via "$(cat ...)" to avoid
# CLI nested-quote escaping bugs.
#
# === Submit-time resource reality (verified 2026-09-25 ~13:20) ===
# queue 2334 gm402-80g: RESOURCEFREE=16 / RESOURCETOTAL=368, PENDING=0 RUNNING=68.
# TWO 32-GPU 4-node jobs are RUNNING right now:
#   * 7nzzn (200step rl/zero, runTime ~3.7h)   — occupies 32 gm402 cards
#   * ghynp (100step rl/zero, just started)     — occupies 32 gm402 cards
# Together that is 64 gm402 cards already in RUNNING, so the queue's
# RESOURCEFREE=16 is the currently-UNALLOCATED quota, NOT physical free; the 64
# running cards do NOT appear in FREE. The cluster-wide idle (cml resource idle
# list) shows gm402 GUARANTEED_PUBLIC FREE=30, also < 32. So a 3rd 32-GPU gang
# submitted now will hit `FailedQueueScheduling: insufficient resource
# cloudml.ng1gm402-8-8.20-224, left N, request 32` and QUEUE until 7nzzn or
# ghynp releases cards. workspace quota gm402 GUARANTEED_PUBLIC APPLIED=1024
# LEFT=0 (public pool historically NOT a hard ceiling; 7nzzn's 32-GPU gang was
# scheduled despite LEFT=0, so it is a latent risk not a hard block).
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has 32
# optimizer.rank0000..0031.pt; (3) vllm_sessions/ has 32 _r0.._r31 sessions.
# For TRIM/ALL additionally: the curate-clipping tolerant-skip patch MUST be
# asserted in preflight (PATCH_CHECK_OK includes "skip_clipped_doc": True) or
# step0 datum will abort on doc 0000950103-25-007493_26 (same as single-node
# x2j3u FAILED).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_ddp_100step_all_32gpu_4node_20260925.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-trim-ddp-100step-all-32gpu-4node-20260925 \
  --description "TRIM Harness-1 BC+ TRIM GPT-OSS-20B mxfp4 4node 32gpu torch_ddp_lora. 100step all (H_full) mb1 gpu-mem0.90 max-turns40. train_method=trim (OPD, NOT rl). curate-clipping tolerant-skip patch REQUIRED (x2j3u abort fix). Shared --out via --env OUT. RDMA on. world_size=32 audit required." \
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
  --env OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_ddp_100step_all_32gpu_4node_20260925 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
