#!/usr/bin/env bash
# Submit Harness-1 BC+ GPT-OSS-20B (--train_method no-gate --component all,
# --rl-reward final-answer-recall) via torch_ddp_lora backend on FOUR 8-GPU
# gm402-80g nodes (4 nodes x 8 cards = 32 GPUs, 200 steps).
#
# Fork of submit_gpt_oss_20b_trim_ddp_200step_all_32gpu_4node_20260925.sh with
# two changes (matching the entry): train_method=no-gate (was trim), and
# --rl-reward final-answer-recall via the NATIVE flag path (the recall-only
# entry hard-edited four_cell_runtime.py; this job does NOT touch the code tree
# — reward comes from --rl-reward -> reward_policy.score_explicit_reward).
# Model is bf16-dequant (NOT original mxfp4): avoids the mxfp4 Marlin PTX crash
# (v0dhh FAILED@step1) + NaN (txgli/zxkvl FAILED@step0); bf16 is the verified
# safe path. mb4 is safe on bf16 (no Marlin/Triton backend, no mxfp4 OOM floor).
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. NOTE: the
# CloudML JuiceFS schema field is `juiceFsCluster` (there is no literal
# `endpoint` key) — its value "bj9-cloudml" IS the cluster/endpoint identifier
# (verified by 7nzzn describe juiceFSCluster field). image_command read verbatim
# from the entry script via "$(cat ...)" to avoid CLI nested-quote escaping bugs.
#
# === GPU availability at submit-time snapshot (2026-10-15) ===
# queue 2334 "记忆对话组-训练队列" kscn-hd-cloudml-prod(青海) gm402-80g:
#   RESOURCEFREE=174 / RESOURCETOTAL=368, AVAILABLESPEC=1,2,4,8.
# 32-GPU (4x8) gang fits comfortably in 174 free cards. No contention from
# prior jobs at this snapshot. Watch FailedQueueScheduling only if pool drains
# between submit and schedule.
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has 32
# optimizer.rank0000..0031.pt; (3) vllm_sessions/ has 32 _r0.._r31 sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_no_gate_final_answer_recall_32gpu_4node_20261015.sh"
test -f "${ENTRY}"

OUT_DIR=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_no_gate_final_answer_recall_32gpu_4node_20261015

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-no-gate-fa-recall-32gpu-4node-20261015 \
  --description "TRIM Harness-1 BC+ GPT-OSS-20B no-gate (CISPO+projected-gap g=1) component=all rl-reward=final-answer-recall (native flag, pure-fa reward). bf16-dequant (NOT mxfp4, avoids Marlin PTX+NaN). 4node32gpu torch_ddp_lora mb1 (GPT-OSS-20B MoE no-grad-ckpt铁律) gpu-mem0.85 (bf16-dequant实测验证值,非0.90) max-turns40 max-model-len8192. curate-clipping patch asserted. Shared --out via --env OUT. RDMA on. world_size=32 audit." \
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
  --env OUT="${OUT_DIR}" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
