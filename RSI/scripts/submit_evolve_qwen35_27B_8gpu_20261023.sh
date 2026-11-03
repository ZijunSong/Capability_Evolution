#!/usr/bin/env bash
# Submit RSI phase-1 EVOLVE — Qwen3.5-27B, single node 8xM402, priority 5.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_evolve_qwen35_27B_8gpu_20261023.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name rsi-evolve-qwen35-27b-p1-20261023 \
  --description "RSI phase-1 EVOLVE (Qwen3.5-27B): model as actor+self-proposer generates harness component code from empty bank; 7xM402 panel workers (GPU0-6)+proposer (GPU7); bf16 ~54GB/GPU; toy_file/toy_kv protocol panel (real domains await data prepare). Artifacts runs/evolve_qwen35_27b_p1/evolve/." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 5 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
