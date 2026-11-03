#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_e0_react_appworld_9B_8gpu_20261104_r1.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-e0-react-appworld-9b-8gpu-20261104-r1 \
  --description "E0 interface+measurement acceptance: AppWorld 16 official-gated dev tasks, H0-ReAct vs H0-Act paired, Qwen3.5-9B, 8xM402. Gates: G1 all VALID via official evaluate_task; G2 scorer never mutates DB; G3 both arms n_decisions>0+tokens>0; G4 official split listings. Strict MAY be 0. OUT runs/e0_react_appworld_9b_r1 + runs/e0_act_appworld_9b_r1." \
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
