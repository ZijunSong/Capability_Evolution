#!/usr/bin/env bash
# Submit Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4, non-dequant
# per user instruction) on 8-GPU gm402-80g via torch_ddp_lora backend, 100 steps,
# component=zero, mb=4, gpu-mem 0.90 (fully utilize 8x80GB KV cache; backward
# runs after close_vllm so high watermark does not squeeze it). Queue 2334,
# priority 8, tag MemQA-General-V5.
#
# JuiceFS is mounted via the CLI --juicefs_mount_configs JSON (endpoint/
# juiceFsCluster=bj9-cloudml), NOT via the YAML --filename, per request.
# The YAML file deliberately omits juiceFsMountConfigs for this reason.
# The YAML imageCommand is a placeholder; the real entry is injected via
# --image_command "$(cat <entry>.sh)".
set -euxo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/scripts/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_100step_zero_8gpu_mb2_20260925.yaml"
ENTRY="/mnt/songzijun/Capability_Evolution/TRIM/scripts/entry_gpt_oss_20b_rl_ddp_100step_zero_8gpu_mb2_20260925.sh"

# JuiceFS mount JSON: volume=songzijun, cluster endpoint=bj9-cloudml.
# Per CLAUDE.md / request: endpoint passed here as juiceFsCluster=bj9-cloudml,
# NOT via YAML --filename.
JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

cml custom_train submit \
  --filename "$YAML" \
  --job_name "trim-harness1-bcplus-full-gpt-oss-20b-rl-ddp-100step-zero-8gpu-mb2" \
  --access_type "PRIVATE" \
  --framework "pytorch" \
  --priority 8 \
  --queue_id "2334" \
  --tag "MemQA-General-V5" \
  --image_url "micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto" \
  --image_command "$(cat "$ENTRY")" \
  --juicefs_mount_configs "$JUICEFS_JSON"
