#!/usr/bin/env bash
# Submit Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4, non-dequant
# per user) via torch_ddp_lora backend on EIGHT 8-GPU gm402-80g nodes (8 nodes x
# 8 cards = 64 GPUs, 200 steps, --component zero = H_zero baseline).
#
# This is the 8-NODE / 64-GPU scale-out of
# submit_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20260925.sh (4-node 32-GPU,
# the 7nzzn baseline: RUNNING stable through step12, runTime 10.8h, zero errors,
# world_size=32 provenance audit fully passed). 8-node 64-GPU is a 2x scale of
# the verified 4-node path. Structural changes vs the 4-node submit:
#   * nodeNumber 4 -> 8
#   * --rdzv-id gptoss20b-rl-32 -> gptoss20b-rl-64 (distinct rendezvous id so
#     this job does not collide with the still-running 7nzzn)
#   * --train-groups-per-step 32 -> 64 (recov1 FIX: mdkyx first attempt used
#     32 groups with world_size=64, round-robin left rank32..63 with 0 queries
#     and 32/64 GPUs idle. 64 restores the 1:1 ratio proven on 7nzzn.)
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. NOTE: the
# CloudML JuiceFS schema field is `juiceFsCluster` (there is no literal
# `endpoint` key) — its value "bj9-cloudml" IS the cluster/endpoint identifier.
# image_command read verbatim from the entry script via "$(cat ...)" to avoid
# CLI nested-quote escaping bugs.
#
# === FIRST 8-node submission risks (do NOT assume success, audit after) ===
#   * PyTorchJob CRD injection: custom_train framework=pytorch nodeNumber=8
#     creates a PyTorchJob CRD with master-0/worker pods + init-pytorch that
#     injects MASTER_ADDR + NODE_RANK (7nzzn verified it does NOT inject
#     RANK/WORLD_SIZE/PET_* on custom_train). The entry unsets the rank env
#     defensively and self-launches torchrun (deterministic).
#   * 64-rank Gloo rendezvous: 7nzzn verified Gloo sidegroup handles 32 ranks
#     for rendezvous + object collectives; 64 ranks is 2x but within practical
#     range. Highest multi-node-specific code risk; flag for post-submit audit.
#   * 64-rank gang scheduling: queue 2334 gm402-80g RESOURCEFREE=112 / cluster
#     idle FREE=155 at submit time, so a 64-GPU gang (8 nodes x 8 cards) is
#     schedulable; but 7nzzn + kxsb3 + iowty + 9ixnl (4 x 32-GPU jobs) are
#     already running (128 cards occupied), and 64-GPU gang is a larger ask
#     than the 32-GPU gangs that succeeded. May queue; audit state + events.
#   * workspace quota: kscn gm402 GUARANTEED_PUBLIC LEFT=0 (public pool does
#     NOT strictly block at LEFT=0 — 7nzzn 32-GPU gang scheduled fine).
#
# world_size=64 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 64 by the three-tuple — (1) stdout has torchrun --nnodes 8
# --nproc_per_node 8 + RESOLVED_CONFIG dist world_size=64; (2) checkpoints/rl/
# step_000001/ has the DDP rank0-merged optimizer.pt + STEP_COMPLETE (DDP path
# saves rank0-merged single optimizer.pt, NOT 64 rank-sharded files — this is the
# correct DDP vs FSDP2 difference, not a defect); (3) vllm_sessions/ has 64
# _r0.._r63 sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_rl_ddp_200step_zero_64gpu_8node_20260925.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-rl-ddp-200step-zero-64gpu-8node-20260925-recov1 \
  --description "TRIM Harness-1 BC+ RL GPT-OSS-20B mxfp4 8node 64gpu torch_ddp_lora 200step zero. RECOV1 of mdkyx(killed@step0): mdkyx used groups-per-step=32 with world_size=64, round-robin left rank32-63 local_queries=0 (32/64 GPU idle). FIX: groups-per-step 32->64 restores 1:1 ratio (7nzzn 32group/32rank), all 64 ranks active. mb1 gpu-mem0.90 max-turns40. --rdzv-id gptoss20b-rl-64. Shared --out via --env OUT. RDMA on. world_size=64 audit." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 8 \
  --preemptible=false \
  --rdma \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":8,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
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
  --env OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_200step_zero_64gpu_8node_20260925_recov1 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
