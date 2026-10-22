#!/usr/bin/env bash
# Submit Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4, non-dequant
# per user) via torch_ddp_lora backend on FOUR 8-GPU gm402-80g nodes (4 nodes x
# 8 cards = 32 GPUs, 200 steps, --component zero = H_zero baseline).
#
# This is the MULTI-NODE variant of submit_gpt_oss_20b_rl_ddp_200step_zero_8gpu_20260924.sh
# (single-node 8-GPU). First 4-node 32-GPU submission in this workspace.
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. NOTE: the
# CloudML JuiceFS schema field is `juiceFsCluster` (there is no literal
# `endpoint` key) — its value "bj9-cloudml" IS the cluster/endpoint identifier.
# image_command read verbatim from the entry script via "$(cat ...)" to avoid
# CLI nested-quote escaping bugs.
#
# === FIRST 4-node submission risks (do NOT assume success, audit after) ===
#   * PyTorchJob CRD injection: custom_train framework=pytorch nodeNumber=4
#     creates a PyTorchJob CRD with master-0/worker pods + init-pytorch that
#     injects RANK/WORLD_SIZE/PET_*/MASTER_ADDR. The entry unsets the rank env
#     so run_train_multinode.py self-launches torchrun (deterministic).
#   * starfs-user-pv FailedMount: the 2026-09-15 mnode_env_probe (ic07h) was
#     killed at deploying because CloudML multi-node auto-injects a
#     starfs-user-pv volume that failed to mount. Platform issue, not our
#     JuiceFS. If it persists, this job stalls at deploying too.
#   * cross-node nccl object collectives: trainer_adapter.py:181 hardcodes
#     nccl; broadcast_object_list/all_gather_object across 4 nodes is unproven
#     (single-node world_size=8 worked on PyTorch 2.10). Highest code risk.
#   * workspace quota: kscn gm402 GUARANTEED_PUBLIC LEFT=0; 32-GPU gang may
#     queue or be rejected despite cluster idle FREE=40.
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has 32
# optimizer.rank0000..0031.pt; (3) vllm_sessions/ has 32 _r0.._r31 sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20260925.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-rl-ddp-200step-zero-32gpu-4node-20260925-recov1 \
  --description "TRIM Harness-1 BC+ RL GPT-OSS-20B mxfp4 4node 32gpu torch_ddp_lora. 200step zero mb1 gpu-mem0.90. recov1 fix: shared --out via --env OUT (korur used per-pod $(date) timestamp -> 4 nodes wrote 2 different out dirs -> step0 gather would crash). RDMA on. world_size=32 audit." \
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
  --env OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20260925 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
