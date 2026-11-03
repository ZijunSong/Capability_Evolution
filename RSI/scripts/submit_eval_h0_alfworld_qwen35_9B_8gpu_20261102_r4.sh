#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_eval_h0_alfworld_qwen35_9B_8gpu_20261102_r4.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-eval-h0-alfworld-qwen35-9b-8gpu-20261102-r4 \
  --description "H0 minimal-harness complete-benchmark evaluation (Qwen3.5-9B) full ALFWorld r4: 7 role pools / 5266 tasks, official engine-won scoring, 8xM402 panel. r3(bwozi) ran but all 8 spawn workers hung in AlfWorldTaskProvider._load: concurrent JuiceFS tree walks amplified metadata ops (GPU 0%, 0 tasks in 1h). r4 adds /tmp inter-process load cache keyed by tree digest + single-process prewarm before workers start. Fresh OUT runs/eval_h0_alfworld_qwen35_9b_bench_r4/eval/." \
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
