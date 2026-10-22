#!/usr/bin/env bash
# Submit Harness-1 BC+ full TRIM (CISPO + projected-gap, no-project method)
# GPT-OSS-20B (bf16-dequant) via torch_ddp_lora backend on FOUR 8-GPU gm402-80g
# nodes (4 nodes x 8 cards = 32 GPUs, 200 steps, --component all = Teacher Full
# / Student Zero). REWARD = final-answer recall ONLY (TRIM_REWARD_MODE=recall).
#
# Fork of submit_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261002_recall_only_bf16.sh
# (same gpt-oss-20B bf16-dequant + 32gpu 4node + final-answer-recall reward),
# changing the train axis from (rl, zero) to (no-project, all), priority 2 -> 8.
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. NOTE: the
# CloudML JuiceFS schema field is `juiceFsCluster` (there is no literal
# `endpoint` key) — its value "bj9-cloudml" IS the cluster/endpoint identifier.
# image_command read verbatim from the entry script via "$(cat ...)" to avoid
# CLI nested-quote escaping bugs.
#
# === Model choice: bf16-dequant (NOT the mxfp4 original) ===
# CLAUDE.md铁律: GPT-OSS-20B 训练禁用原版 mxfp4 (mxfp4+LoRA+PTX 必崩 +
# recall-reward 路径 mxfp4 NaN logprob). 用 JuiceFS 上的 bf16-dequant 目录.
# 镜像 micr...songzijun-cto 是容器, 模型权重仍走 JuiceFS bf16-dequant.
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has 32
# optimizer.rank0000..0031.pt; (3) vllm_sessions/ has 32 _r0.._r31 sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_ddp_200step_no_project_all_recall_32gpu_4node_20261015.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-trim-ddp-200step-no-project-all-32gpu-4node-20261015-resume \
  --description "RESUME of akvlv (FAILED @ step1 backward, JVM pyjnius SIGSEGV occasional). --resume from step_000001 (latest, cell=no_project, STEP_COMPLETE+optimizer+adapter intact). Same config: GPT-OSS-20B bf16-dequant 4node32gpu torch_ddp_lora, no-project+all, final-answer-recall, mb1 gpu-mem0.85. rdzv-id -resume suffix. Skips step0 (60min saved). If step1 backward crashes again at same point, pyjnius/JVM hardening needed." \
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
  --env TRIM_REWARD_MODE=recall \
  --env OUT=/mnt/songzijun/outputs/gpt-oss-20b-trim-no-project-all-recall-32gpu_20261015 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
