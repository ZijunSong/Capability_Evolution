#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_h0v3_appworld_9B_4gpu_20261109_r1.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-h0v3-appworld-9b-4gpu-20261109-r1 \
  --description "AppWorld H0 v3 re-measure (guide 4.5 step1): Qwen3.5-9B 4gpu, appworld_v3.md contract (PR-01: verified API discovery, print rule, typed Finish, structured errors). SAME 16-task e0 panel as r3 (baseline partial 34.74% strict 0/16) + work_train/harness_select splits for the bounded retry. Measures the H0-interface-fix effect, NOT an evolution gain. OUT runs/h0v3_appworld_9b_r1g4." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 2 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":4},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
