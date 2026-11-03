#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_eval_h0_appworld_qwen35_27B_8gpu_20261102_r2.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-eval-h0-appworld-qwen35-27b-8gpu-20261102-r2 \
  --description "H0 minimal-harness full AppWorld eval (Qwen3.5-27B) r2: recovers kygt9 hang (8 workers OOM-killed in from_pretrained full-RAM load + Pool.map stuck on dead children). Fixes: low_cpu_mem_usage/device_map load, parent skips build_policy, staggered worker loads, ProcessPoolExecutor fail-loud, per-task prints. 7 role pools (325 tasks) via rsi.cli eval --harness minimal, official TGC scoring, 8xM402. Artifacts runs/eval_h0_appworld_qwen35_27b_bench/eval/." \
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
