#!/usr/bin/env bash
# Submit Harness-1 BC+ trim/all GPT-OSS-20B (bf16-dequant) on EIGHT 8-GPU
# gm402-80g nodes (8 nodes x 8 cards = 64 GPUs, 200 steps).
#
# === 64-GPU scale-up of kxsb3 (bf16-dequant 32-GPU trim/all, verified) ===
# kxsb3 (t-20260925162650-kxsb3, 32-GPU bf16-dequant trim/all) verified
# step0 -> step1 -> step2 (terminated by user at step2-in-progress, step0+1
# fully preserved): bf16-dequant checkpoint (quant_method=NONE, 39GB) runs the
# standard bf16+LoRA path, vLLM uses "Using TRITON backend for Unquantized MoE"
# (unquantized.py, NOT Marlin) — definitively bypassing the mxfp4 Marlin PTX
# crash (坑1) and triton_kernels CPU-OOM pit (坑4). This 64-GPU job is a 2x
# scale-up of kxsb3's 32-rank rendezvous; per-rank memory budget UNCHANGED
# (each rank still TP=1 single-GPU), mb1 + gpu-mem 0.90 still fits 80GB.
#
# === Why bf16-dequant (NOT original mxfp4) ===
# v0dhh (original mxfp4, 32-GPU trim/all) FAILED at step1 vLLM+LoRA reload on
# cudaErrorUnsupportedPtxVersion (Marlin repack). The PTX crash is an
# INTERMITTENT race; 64 ranks concurrent LoRA reload is far more likely to
# trigger it than 32. kxsb3 bf16-dequant eliminates the Marlin path entirely.
# The user reference command's original-mxfp4 path = v0dhh's failure path and
# would almost certainly crash at step1+ on 64 ranks. User decision: bf16-dequant.
#
# Fresh jobName + fresh OUT dir (kxsb3 attempt preserved, not overwritten).
# Pure CLI flags; JuiceFS via --juicefs_mount_configs JSON (juiceFsCluster=bj9-cloudml).
#
# GPU availability (2026-09-25 ~15:30 at submit-prep): queue 2334 gm402
# RESOURCEFREE=112 / idle FREE=155 / quota LEFT=0 (public pool does NOT strictly
# block at LEFT=0, proven by 7nzzn/iowty 32-GPU gangs scheduling). 64-GPU gang
# (8 nodes x 8 GPU) needs 64 <= 112 queue free and 64 <= 155 idle — both hold,
# so it should schedule promptly. Still watch describe state + events
# Scheduled/Unschedulable/FailedQueueScheduling.
#
# world_size=64 audit post-submit (CLAUDE.md): (1) stdout torchrun --nnodes 8
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ 64 optimizer.pt (DDP
# rank0 single-file or 32/64 rank shards); (3) vllm_sessions/ 64 _r0.._r63
# sessions. PLUS bf16-dequant-specific: MODEL_PROBE must show quant_method=NONE
# (no mxfp4), and step1 vLLM+LoRA reload must NOT show "Using Marlin backend
# for mxfp4 lora" (bf16 path has no Marlin) — this is the v0dhh fix verification.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_ddp_200step_all_64gpu_8node_20260925.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-trim-ddp-200step-all-64gpu-8node-20260925-bf16 \
  --description "TRIM BC+ trim/all GPT-OSS-20B bf16-dequant 8node 64gpu torch_ddp_lora 200step. 64-GPU scale-up of kxsb3 (bf16 32gpu verified step0-1). bf16 standard LoRA path no Marlin/Triton no PTX. mb1 gpu-mem0.90. curate-clipping patch. Shared --out via --env OUT. RDMA. world_size=64 audit." \
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
  --env OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_ddp_200step_all_64gpu_8node_20260925_bf16 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
