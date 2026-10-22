#!/usr/bin/env bash
# Submit Harness-1 BC+ trim (SR-OPD/CISPO distill, --train_method trim
# --component all) GPT-OSS-20B (original mxfp4, non-dequant per user) via
# torch_ddp_lora backend on FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards =
# 32 GPUs, 200 steps).
#
# FIRST multi-node trim/all submission. The two running 32-GPU jobs (7nzzn
# 200step zero, ghynp 100step zero) are --train_method rl --component zero
# (H_zero RL baseline); this one is --train_method trim --component all
# (SR-OPD + RL joint distill, the CISPO teacher-student path). Single-node
# trim/all is the ethqp/axztt path; this is its multi-node debut.
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. NOTE: the
# CloudML JuiceFS schema field is `juiceFsCluster` (there is no literal
# `endpoint` key) — its value "bj9-cloudml" IS the cluster/endpoint
# identifier (verified by 7nzzn describe juiceFSCluster field). image_command
# read verbatim from the entry script via "$(cat ...)" to avoid CLI
# nested-quote escaping bugs.
#
# === GPU contention (2026-09-25) — EXPECT TO QUEUE ===
# At submit time: 7nzzn (200step zero, RUNNING, ~3.7h) + ghynp (100step zero,
# RUNNING) occupy 64 cards. queue 2334 RESOURCEFREE=16, idle FREE=30, quota
# LEFT=0. This third 32-GPU gang CANNOT schedule until one of them releases;
# expect FailedQueueScheduling "insufficient resource ... left X, request 32"
# (exactly as 7nzzn itself did at 09:37 before resources freed at 09:38). It
# will dequeue once 7nzzn or ghynp finishes/stops. Submit is fine; the job
# waits in queue. Verify with: cml custom_train describe <jobId> (state +
# events Scheduled/Unschedulable/FailedQueueScheduling).
#
# === Multi-node risks (most retired by 7nzzn 2026-09-25) ===
#   * PyTorchJob CRD injection: custom_train framework=pytorch nodeNumber=4
#     creates a PyTorchJob CRD with master-0/worker pods + init-pytorch that
#     injects MASTER_ADDR/NODE_RANK (NOT RANK/WORLD_SIZE, verified by 7nzzn
#     PREFLIGHT rank_env=None world_size_env=None). The entry unsets the rank
#     env defensively so run_train_multinode.py self-launches torchrun.
#   * starfs-user-pv FailedMount: the 2026-09-15 probe was killed at deploying
#     because CloudML multi-node auto-injects a starfs-user-pv volume that
#     failed to mount. 7nzzn (2026-09-25) did NOT hit this (platform issue
#     resolved). Low risk now, still watch events.
#   * cross-node nccl object collectives: 7nzzn verified Gloo rendezvous +
#     object ops work cross-4-node (PyTorch auto-uses Gloo for
#     broadcast_object_list/all_gather_object). Risk retired by 7nzzn.
#   * workspace quota: kscn gm402 GUARANTEED_PUBLIC LEFT=0; 7nzzn (32-GPU gang)
#     was scheduled despite this, so the public pool does NOT strictly block
#     at LEFT=0. Still, 32-GPU gang may queue if pool is busy.
#   * trim/all-specific curate-clipping patch (ethqp fix): without
#     action_is_visible + tinker_opd_datum skip path, step0 datum build aborts
#     on "projected curate refs docs not visible after clipping" (x2j3u
#     FAILED@step0). Patch is in the mounted code tree and asserted by the
#     entry's PATCH_CHECK. This is the ONLY trim/all-specific risk not covered
#     by 7nzzn's rl/zero run; watch step0 datum build closely.
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has 32
# optimizer.rank0000..0031.pt; (3) vllm_sessions/ has 32 _r0.._r31 sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_ddp_200step_all_32gpu_4node_20260925.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-trim-ddp-200step-all-32gpu-4node-20260925 \
  --description "TRIM Harness-1 BC+ trim (SR-OPD/CISPO distill, train_method=trim component=all) GPT-OSS-20B mxfp4 4node 32gpu torch_ddp_lora. FIRST multi-node trim/all (7nzzn/ghynp are rl/zero). mb1 gpu-mem0.90 max-turns40 max-model-len8192. curate-clipping patch (action_is_visible) asserted. Shared --out via --env OUT (multi-node铁律). RDMA on. world_size=32 audit." \
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
  --env OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_ddp_200step_all_32gpu_4node_20260925 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
