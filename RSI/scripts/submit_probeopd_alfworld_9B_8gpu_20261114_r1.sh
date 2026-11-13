#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_probeopd_alfworld_9B_8gpu_20261114_r1.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-probeopd-alfworld-9b-8gpu-20261114-r1 \
  --description "ALFWorld probe-opd (guide 5.3 gates 2/3): real distillability probe with the CONFIRMED teacher 622625de (confirm delta_h_strict +45.8pp, 11W/0L/13T on 24 fresh groups). Student H0 decision points -> teacher forks at the decision snapshot with exact public prefix (PR-06) -> native anchors -> legal targets -> OPD gradient. Serial topology. OUT runs/alfworld_probe_r1." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
