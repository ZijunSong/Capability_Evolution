#!/usr/bin/env bash
# Submit Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4) on
# 8-GPU gm402-80g via torch_ddp_lora backend, queue 2334, priority 8,
# tag MemQA-General-V5.
#
# JuiceFS is mounted via the CLI --juicefs_mount_configs JSON (endpoint/
# juiceFsCluster=bj9-cloudml), NOT via the YAML --filename, per request.
# The YAML file deliberately omits juiceFsMountConfigs for this reason.
set -euxo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/scripts/cloudml_custom_train_harness1_bcplus_full_gptoss20b_rl_ddp_100step_8gpu_20260918.yaml"
ENTRY="/mnt/songzijun/Capability_Evolution/TRIM/scripts/entry_gptoss20b_rl_ddp_100step_8gpu.sh"

# JuiceFS mount JSON: volume=songzijun, cluster endpoint=bj9-cloudml.
JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

cml custom_train submit \
  --filename "$YAML" \
  --job_name "trim-harness1-bcplus-full-gptoss20b-rl-ddp-100step-8gpu" \
  --access_type "PRIVATE" \
  --framework "pytorch" \
  --priority 8 \
  --queue_id "2334" \
  --tag "MemQA-General-V5" \
  --image_url "micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto" \
  --image_command "$(cat "$ENTRY")" \
  --juicefs_mount_configs "$JUICEFS_JSON"
