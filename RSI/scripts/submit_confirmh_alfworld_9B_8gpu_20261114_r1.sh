#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_confirmh_alfworld_9B_8gpu_20261114_r1.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-confirmh-alfworld-9b-8gpu-20261114-r1 \
  --description "ALFWorld fresh confirm_h (guide 5.2/5.3): frozen candidate 622625de (c31e0e88, pre-registered) vs H0 on v2 manifest confirm_h panel (24 groups), Qwen3.5-9B 8gpu SERIAL topology (5-hang-proof). Gates: delta_h_strict>0 with cost/regression info = GAIN_CONFIRMED_ALIGNMENT_PENDING; CI crossing zero = UNRESOLVED_INTERVAL reported honestly. OUT runs/alfworld_confirm_r1." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
