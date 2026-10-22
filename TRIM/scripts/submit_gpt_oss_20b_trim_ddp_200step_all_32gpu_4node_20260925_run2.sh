#!/usr/bin/env bash
# Submit RECOV1 of Harness-1 BC+ trim/all GPT-OSS-20B on FOUR 8-GPU gm402-80g
# nodes (32 GPUs, 200 steps). This is the recovery of failed v0dhh
# (t-20260925132550-v0dhh), switching from original mxfp4 to the bf16-dequant
# checkpoint to definitively bypass the step1 mxfp4+LoRA Marlin PTX crash
# (cudaErrorUnsupportedPtxVersion @ repack_weight).
#
# v0dhh step0 was fully successful (adapter landed); only step1 vLLM+LoRA reload
# crashed on the mxfp4 Marlin MoE weight repack. Per user decision, recov1 uses
# /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant (same GptOssForCausalLM,
# quant_method=NONE, 39GB) running the standard bf16+LoRA path — NO Marlin,
# NO Triton, NO PTX. vLLM was dequantizing mxfp4→bf16 at runtime anyway, so
# bf16-dequant uses the SAME compute path with pre-converted weights.
#
# Fresh jobName + fresh OUT dir (v0dhh failed attempt preserved, not overwritten).
# Pure CLI flags; JuiceFS via --juicefs_mount_configs JSON (juiceFsCluster=bj9-cloudml).
#
# GPU contention: 7nzzn still RUNNING (200step rl/zero, ~3.7h+). ghynp stopped.
# queue 2334 FREE was 48 at v0dhh submit; a 32-GPU gang may queue until 7nzzn
# releases, but v0dhh scheduled in 12s so the pool has capacity. Watch describe.
#
# world_size=32 audit post-submit (CLAUDE.md): (1) stdout torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ 32 optimizer.pt;
# (3) vllm_sessions/ 32 _r0.._r31. PLUS bf16-dequant-specific: MODEL_PROBE must
# show quant_method=NONE (no mxfp4), and step1 vLLM+LoRA reload must NOT show
# "Using Marlin backend for mxfp4 lora" (bf16 path has no Marlin) — this is the
# v0dhh fix verification.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_ddp_200step_all_32gpu_4node_20260925_run2.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-trim-ddp-200step-all-32gpu-4node-20260925-run2-bf16 \
  --description "TRIM BC+ trim/all GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora 200step. RECOV1 of v0dhh (mxfp4 step1 Marlin PTX crash). bf16 standard LoRA path, no Marlin/Triton. mb1 gpu-mem0.90. curate-clipping patch. Shared --out via --env OUT. RDMA. world_size=32 audit." \
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
  --env OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_ddp_200step_all_32gpu_4node_20260925_run2_bf16 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
