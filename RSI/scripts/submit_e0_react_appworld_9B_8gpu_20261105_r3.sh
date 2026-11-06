#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_e0_react_appworld_9B_8gpu_20261105_r3.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-e0-react-appworld-9b-8gpu-20261105-r3 \
  --description "E0 r3 answer-fix control: same 16 official-gated dev tasks, H0-ReAct vs H0-Act paired, Qwen3.5-9B. vs r1 (hf4ax, TGC 0 with 11/16 answer=NOT_GIVEN): adapter now submits Finish.answer via apis.supervisor.complete_task; numeric finish.answer accepted by parse_action; env_doc teaches respond-before-finish. Expect TGC >> 0. OUT runs/e0_react_appworld_9b_r3 + runs/e0_act_appworld_9b_r3." \
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
