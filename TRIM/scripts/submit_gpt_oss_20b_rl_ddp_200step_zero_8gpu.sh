#!/usr/bin/env bash
# Submit Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4, non-dequant
# per user instruction) via torch_ddp_lora backend on one 8-GPU gm402-80g node
# (200 steps). Pure CLI flags (no --filename YAML); JuiceFS mount passed
# explicitly via --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml
# (endpoint). image_command read verbatim from the entry script via "$(cat ...)"
# to avoid CLI nested-quote escaping bugs.
#
# RECOVERY1 (of failed t-20260919074148-ybdr3): removed SCAPE_FORCE_LOCAL_HARMONY=1
# from envConfigs. That env forced harness.ultra_core to use _LocalConversation
# (which has no to_json), while load_harmony_enc() returned the openai_harmony
# Rust encoder whose render_conversation_for_completion() hard-calls
# conversation.to_json() → AttributeError at step0 first-turn prompt (only
# gpt-oss triggers the Harmony path; Qwen3-4B uses hf_chat and was unscathed).
# aemum (9/14, same gpt-oss mxfp4 local_legacy) never set this env and ran fine.
#
# RECOVERY2 (of failed t-20260919093125-gsqvw, recovery1): patched vendored
# trim/integrations/verl/trainer_adapter.py:389 to pass startup_timeout_s=3600.0
# to VLLMGenerateClient (was default 900s). gpt-oss-20b mxfp4 has 3 safetensors
# shards; 8 DDP ranks each spin a TP=1 vLLM engine reading the 13GB model from
# JuiceFS concurrently → ~7min/shard × 3 ≈ 20min/engine, exceeding 900s default.
# 3600s matches four_cell_runtime.open_vllm. py_compile of trainer_adapter.py
# added to preflight. This is a JuiceFS-code-tree patch (read from mounted
# /mnt/songzijun), no image rebuild needed.
#
# world_size=8 is required (single node, 8 ranks via torchrun --nproc_per_node 8).
# Post-submit provenance audit (per CLAUDE.md): verify world_size is truly 8 by
# the three-tuple — (1) stdout has [verl-fsdp2-launch] banner + torchrun
# --nproc_per_node 8; (2) checkpoints/rl/step_000001/ has 8 optimizer.rank0000
# ..0007.pt; (3) vllm_sessions/ has 8 _r0 .. _r7 sessions. Any one missing =>
# actually single-GPU run, result is NOT a valid "8-GPU DDP" artifact.
#
# Known risk (user-accepted): original mxfp4 + vLLM enable_lora at step1+
# vLLM+LoRA reload may crash cudaErrorUnsupportedPtxVersion (Marlin LoRA backend
# PTX vs gm402 sm_90). local_legacy + DISABLE_CUSTOM_ALL_REDUCE=1 ran 8 rollout
# sessions ~55h without PTX (aemum), but torch_ddp_lora is TP=1 with NO
# DISABLE_CUSTOM_ALL_REDUCE — no direct precedent. If step1 vLLM+LoRA crashes
# with PTX, switch --model_name to /mnt/songzijun/models/openai/gpt-oss-20b-bf16
# -dequant and resubmit (bf16 fully avoids mxfp4 Marlin LoRA path).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_rl_ddp_200step_zero_8gpu.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-rl-ddp-200step-zero-8gpu \
  --description "GPT-OSS-20B mxfp4 RL ZERO-component(H_zero, all advanced components OFF, vs j7bje all) via torch_ddp_lora, 8x gm402-80g, 200 steps, gs8, max-turns 40, max-model-len 8192, mb1, gpu-mem 0.90. Original mxfp4 (vLLM auto-dequant bf16, no VLLM_MXFP4_USE_MARLIN). Inherit j7bje 2 fixes: no SCAPE_FORCE_LOCAL_HARMONY + trainer_adapter startup_timeout_s=3600. world_size=8." \
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
