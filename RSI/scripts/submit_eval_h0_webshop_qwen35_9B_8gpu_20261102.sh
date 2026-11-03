#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_eval_h0_webshop_qwen35_9B_8gpu_20261102.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-eval-h0-webshop-qwen35-9b-8gpu-20261102-r4 \
  --description "H0 minimal-harness WebShop evaluation (Qwen3.5-9B) r2: official get_reward on frozen_goals 12087 (final_test=official 500-goal human panel), 8 role pools, 8xM402 224GB node. r2 fixes r9mla: pip() helper caught TimeoutExpired (causal-conv1d sdist build blew 600s and killed the entry; it is best-effort, fla is the hard assert). Shared SimServer cache per worker; 8-task smoke gate before full 7-pool run. Fresh OUT runs/eval_h0_webshop_qwen35_9b_bench/eval/." \
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
