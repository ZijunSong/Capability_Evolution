#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_eval_hk_appworld_qwen35_9B_8gpu_20261102_r1.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-eval-hk-appworld-qwen35-9b-8gpu-20261102-r1 \
  --description "exp_hk complete-benchmark eval (Qwen3.5-9B x self-evolved harness H_k) full AppWorld r1: 7 role pools / 325 tasks, official TGC, 8xM402 panel. H_k = fh46b champion bank 73ddc436 (ContextCompressor). Paired vs exp_h0 dhlol r2 (TGC 23.51% valid): same seed 17 / same zero-sampling / same dep pins / same budgets; only harness differs (bank vs minimal). OUT runs/eval_hk_appworld_qwen35_9b_bench_r1/eval/." \
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
