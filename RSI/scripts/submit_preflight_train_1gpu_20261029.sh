#!/usr/bin/env bash
# Submit the 1-GPU training preflight (cheap gate before 32-GPU jobs).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_preflight_train_1gpu_20261029.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name rsi-preflight-train-opd-9b-1gpu-20261029 \
  --description "RSI training-path preflight (1 GPU, ~15min): real Qwen3.5-9B bf16 + LoRA through generate/episode-collection(teacher fork)/joint RL+OPD step/export. Cheap gate to run BEFORE queueing 32-GPU HarnessOPD jobs; catches GPU-config divergences (dtype/device/wrappers) that CPU tests miss." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":1},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
