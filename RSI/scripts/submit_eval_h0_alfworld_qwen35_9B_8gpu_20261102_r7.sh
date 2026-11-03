#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_eval_h0_alfworld_qwen35_9B_8gpu_20261102_r7.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-eval-h0-alfworld-qwen35-9b-8gpu-20261102-r7 \
  --description "H0 minimal-harness complete-benchmark evaluation (Qwen3.5-9B) full ALFWorld r7: 7 role pools / 5266 tasks, official engine-won scoring, 8xM402 panel. r6(zqpvg) hit a new bug at 48/736: alfworld_env Finish branch passed done=True to Observation (no such field) so every task whose model emitted Finish died with TypeError (482/664 in work_train). Fixed in rsi pkg (done belongs to StepResult only) + 2 unit tests. Fresh OUT runs/eval_h0_alfworld_qwen35_9b_bench_r7/eval/." \
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
