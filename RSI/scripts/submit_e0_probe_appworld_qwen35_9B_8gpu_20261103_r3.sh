#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_e0_probe_appworld_qwen35_9B_8gpu_20261103_r3.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-e0-probe-appworld-qwen35-9b-8gpu-20261103-r3 \
  --description "E0 measurement rebuild r2 (pri8 resubmit of fznew, identical entry): 4 frozen AppWorld scenarios x 3 phases (no-op control / fixture check / 9B real H0) through the fixed runtime+scorer. GATE: all tasks measurement_status=VALID, scorer never mutates business DB, real-H0 phase shows n_decisions>0 and nonzero tokens (clock-freeze regression). strict success MAY be 0 - that is an E1 difficulty finding. OUT runs/e0_probe_r3 + runs/e0_noop_r3." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
