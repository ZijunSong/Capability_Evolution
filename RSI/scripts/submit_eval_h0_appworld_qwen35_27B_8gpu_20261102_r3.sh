#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_eval_h0_appworld_qwen35_27B_8gpu_20261102_r3.sh"
test -f "${ENTRY}"
cml custom_train submit \
  --job_name rsi-eval-h0-appworld-qwen35-27b-8gpu-20261102-r3 \
  --description "H0 minimal-harness full AppWorld eval (Qwen3.5-27B) r3: recovers 1nvje BrokenProcessPool (unpinned pip pulled torch 2.15.0+cu132 on cu128-driver gm402 + unpinned fla ABI mismatch -> workers killed at first generate; 7/7 splits 0 tasks done). Fixes: torch==2.10.0+cu128 pin + CUDA gate, fla==0.6.0 + is_flash_linear_attention assert, aligned proven dhlol(9B r2) dep block. 7 role pools 325 tasks, runs/eval_h0_appworld_qwen35_27b_bench_r3/eval/." \
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
