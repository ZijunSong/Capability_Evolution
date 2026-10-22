#!/usr/bin/env bash
# Submit Harness-1 BC+ trim (SR-OPD/CISPO distill, --train_method trim
# --component all) Qwen3-4B-Instruct-2507 (dense bf16 4B) via torch_ddp_lora
# backend on FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards = 32 GPUs, 200 steps).
#
# === THIS VARIANT: trim/all + final-answer recall + turn-count penalty reward (GATED) + TensorBoard ===
# Built on the iowty-verified Qwen3-4B trim/all 32gpu path. Two deltas vs iowty:
#   1. Reward formula: the active path
#      (trim.training.four_cell_runtime.terminal_reward_breakdown, called by
#      trainer_adapter via four_cell_runtime — NOT harness-1's external
#      compute_reward) is pure final-answer recall; this job enables the
#      turn-count penalty via TRIM_REWARD_MODE=recall_turn (total = recall -
#      0.15*(n_turns/40)*[rec>0], GATED so penalty is zeroed when rec=0, fix
#      for jf8fc reward collapse to n_turns=1). Env-gated, so leaving it unset
#      reproduces iowty.
#   2. TensorBoard + metrics: --enable_tensorboard --enable_metrics flip the
#      platform to inject $TENSORBOARD_LOG_PATH, which
#      trainer_adapter._make_tensorboard_writer picks up (it writes reward/
#      loss/ cispo/ tokens/ timing_s/ mem/ scalar blocks every optimizer step;
#      turn_penalty_mean / n_turns_mean are in the reward block). JSONL
#      metrics.jsonl / step_timing.jsonl / turn_samples.jsonl are also written
#      under <OUT>/rl/ as a fallback for offline plotting.
#   3. Independent jobName + OUT (-recall-turn-punish-gated) so this run does
#      NOT collide with iowty's provenance; distinct --rdzv-id.
#
# Everything else identical to iowty: dense bf16 4B (no mxfp4/triton/Marlin),
# mb=4, gpu-mem 0.90, SCAPE_FORCE_LOCAL_HARMONY=1 (harmless on Qwen), model cp
# to /tmp per node, PATCH_CHECK asserts all 4 patches (3 DDP-generic +
# trim/all-specific curate-clipping). Shared --out via --env OUT (multi-node铁律).
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. The CloudML
# JuiceFS schema field is `juiceFsCluster` (no literal "endpoint" key); its
# value "bj9-cloudml" IS the cluster/endpoint identifier (verified by 7nzzn
# describe juiceFSCluster field). image_command read verbatim from the entry
# script via "$(cat ...)" to avoid CLI nested-quote escaping bugs.
#
# === GPU contention (check at submit time) ===
# Queue 2334 cloudml.ng1gm402-8-8.20-224 RESOURCEFREE varies as 32-GPU gangs
# start/stop. A 32-GPU gang (4 nodes x 8 cards) may queue if the pool is busy
# (7nzzn hit "left 8, request 32" before resources freed). If queue
# RESOURCEFREE < 32, expect FailedQueueScheduling "insufficient resource ...
# left X, request 32"; it dequeues once a gang releases. workspace quota
# kscn gm402 GUARANTEED_PUBLIC LEFT=0 does NOT strictly block (7nzzn/iowty
# scheduled despite it). Verify with: cml custom_train describe <jobId>
# (state + events Scheduled/Unschedulable/FailedQueueScheduling).
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has optimizer.pt
# (DDP rank0-merged single file, NOT FSDP2 sharded); (3) vllm_sessions/ has 32
# _r0.._r31 sessions. Plus reward: stdout PREFLIGHT must show
# reward_mode=recall_turn; metrics.jsonl reward_parts_last must show
# protocol=final_answer_recall_gated_turn_penalty + turn_penalty>0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_trim_ddp_200step_all_recall_turn_punish_gated_32gpu_4node_20260930.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-qwen3-4b-trim-ddp-200step-all-recall-turn-punish-gated-32gpu-4node-20260930 \
  --description "TRIM Harness-1 BC+ trim (train_method=trim component=all) Qwen3-4B-Instruct-2507 dense bf16 4B 4node 32gpu torch_ddp_lora. GATED turn penalty: total=recall-0.15*(n_turns/40)*[rec>0] (turn penalty zeroed when rec=0, fixes jf8fc reward collapse to n_turns=1). tensorboard on. mb4 gpu-mem0.90 max-turns40. curate-clipping patch asserted. world_size=32 audit." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 8 \
  --preemptible=false \
  --rdma \
  --enable_tensorboard \
  --enable_metrics \
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
  --env TRIM_REWARD_MODE=recall_turn \
  --env TRIM_TURN_PENALTY=0.15 \
  --env TRIM_TURN_NORM=40 \
  --env OUT=/mnt/songzijun/outputs/qwen3-4b-trim-recall-turn-punish-gated-32gpu_20260930 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
