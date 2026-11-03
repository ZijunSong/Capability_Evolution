#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_eval_h0_alfworld_qwen35_9B_8gpu_20261102_r5.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-eval-h0-alfworld-qwen35-9b-8gpu-20261102-r5 \
  --description "H0 minimal-harness complete-benchmark evaluation (Qwen3.5-9B) full ALFWorld r5: 7 role pools / 5266 tasks, official engine-won scoring, 8xM402 panel. r4(y9qbm) died at data prepare: provider cache log print polluted stdout JSON piped to json.load gate. Fix in rsi pkg: cache prints now go to stderr, stdout stays pure JSON. Same prewarm+cache hang fix as r4. Fresh OUT runs/eval_h0_alfworld_qwen35_9b_bench_r5/eval/." \
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
