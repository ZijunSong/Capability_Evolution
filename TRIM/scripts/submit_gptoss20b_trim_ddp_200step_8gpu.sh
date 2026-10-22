#!/usr/bin/env bash
# Submit Harness-1 BC+ full trim GPT-OSS-20B (original mxfp4, non-dequant
# per user instruction) via torch_ddp_lora backend on one 8-GPU gm402-80g node
# (200 steps). Pure CLI flags (no --filename YAML); JuiceFS mount passed
# explicitly via --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml
# (endpoint). image_command read verbatim from the entry script via "$(cat ...)"
# to avoid CLI nested-quote escaping bugs.
#
# world_size=8 is REQUIRED (single node, 8 ranks via torchrun --nproc_per_node 8).
# Post-submit provenance audit (CLAUDE.md): verify world_size is truly 8 by the
# three-tuple — (1) stdout has torchrun --nproc_per_node 8; (2)
# checkpoints/<cell>/step_000001/ has 8 optimizer.rank0000..0007.pt; (3)
# vllm_sessions/ has 8 _r0.._r7 sessions. Any one missing => actually
# single-GPU run, result is NOT a valid "8-GPU DDP" artifact.
#
# Known risk (user-accepted): original mxfp4 + vLLM enable_lora at step1+
# vLLM+LoRA reload may crash cudaErrorUnsupportedPtxVersion (Marlin LoRA backend
# PTX vs gm402 sm_90). The RL ddp recov2 (j7bje) proved mxfp4+LoRA is safe on
# this torch_ddp_lora TP=1 path (it auto-dequantizes to bf16 in vLLM 0.19.1 when
# triton_kernels is absent, NOT the Marlin path). trim path reuses the same
# engine; if step1 vLLM+LoRA crashes with PTX, switch --model_name to
# /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant and resubmit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gptoss20b_trim_ddp_200step_8gpu.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gptoss20b-trim-ddp-200step-8gpu \
  --description "Harness-1 BC+ trim GPT-OSS-20B (original mxfp4, non-dequant) via torch_ddp_lora, 8x gm402-80g single node, 200 steps, 32 grp/step, gs8, mb1, max-turns 40, max-model-len 8192, gpu-mem 0.90, local_legacy BM25. Four-pack torch2.10.0/vllm0.19.1/tf4.57.6/pyserini2.0.0. Code patches in place: startup_timeout_s=3600, 4-tuple unpack, EADDRINUSE retry. SCAPE_FORCE_LOCAL_HARMONY unset. Model cp to /tmp. world_size=8." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 8 \
  --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
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
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
