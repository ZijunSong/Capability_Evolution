#!/usr/bin/env bash
# Submit Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4, non-dequant
# per user) via torch_ddp_lora backend on FOUR 8-GPU gm402-80g nodes (4 nodes x
# 8 cards = 32 GPUs, 100 steps, --component zero = H_zero baseline).
#
# This is the 100-step MULTI-NODE variant. Derived from
# submit_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20260925.sh (the 200-step
# 4-node job t-20260925093542-7nzzn, RUNNING 2026-09-25). Deltas vs that submit:
#   * job_name ...200step...-recov1 -> ...100step...
#   * --description: 200step -> 100step, mb=1 explicit note
#   * --env OUT: points at the 100step output dir (not the 200step one)
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
# === Submit-time resource reality (verified 2026-09-25 ~12:40) ===
# queue 2334 gm402-80g: RESOURCEFREE=32 / RESOURCETOTAL=368, PENDING=0 RUNNING=69.
# BUT the 200-step job 7nzzn (t-20260925093542-7nzzn) is RUNNING and occupies
# exactly 32 gm402 cards (4 nodes x 8). The queue RESOURCEFREE=32 figure is the
# queue's currently-UNALLOCATED quota, NOT the cluster's physical free; 7nzzn's
# 32 cards are already counted in RUNNING and do NOT appear in FREE. The
# cluster-wide idle (cml resource idle list) shows gm402 GUARANTEED_PUBLIC
# FREE=54, which IS physical and INCLUDES 7nzzn's freed... actually idle FREE is
# a separate snapshot. Bottom line: a 2nd 32-GPU gang submitted while 7nzzn is
# still running will likely hit `FailedQueueScheduling: insufficient resource
# ... left 8, request 32` (exactly the event 7nzzn itself hit at 09:37 before
# resources freed up). Expect queueing. workspace quota gm402 GUARANTEED_PUBLIC
# APPLIED=1024 LEFT=0 (public pool historically NOT a hard ceiling; historical
# single-node 8-GPU jobs kept getting scheduled, and 7nzzn's 32-GPU gang was
# scheduled too, so LEFT=0 did not block — but it is a latent risk).
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has 32
# optimizer.rank0000..0031.pt; (3) vllm_sessions/ has 32 _r0.._r31 sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_rl_ddp_100step_zero_32gpu_4node_20260925.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-rl-ddp-100step-zero-32gpu-4node-20260925 \
  --description "TRIM Harness-1 BC+ RL GPT-OSS-20B mxfp4 4node 32gpu torch_ddp_lora. 100step zero mb1 gpu-mem0.90 max-turns40. 100-step sibling of 200-step 7nzzn (RUNNING). Shared --out via --env OUT (korur out-bug fix). RDMA on. world_size=32 audit required." \
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
  --env OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_100step_zero_32gpu_4node_20260925 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
