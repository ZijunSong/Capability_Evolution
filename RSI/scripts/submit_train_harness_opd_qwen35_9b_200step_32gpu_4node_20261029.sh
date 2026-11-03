#!/usr/bin/env bash
# Submit RSI HarnessOPD internalization — Qwen3.5-9B, all-teacher/zero-student,
# 200-step, 4node32gpu, priority 8.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_train_harness_opd_qwen35_9b_200step_32gpu_4node_20261029.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name rsi-opd-qwen35-9b-teacher-all-student-zero-200step-32gpu-4node-20261029 \
  --description "RSI HarnessOPD internalization (Qwen3.5-9B): all-teacher/zero-student. Student=H0 only, Teacher=H0+r4 champion bank (FinishGuard) isolated forks; joint CISPO + gated projected-gap (lambda 0.01, gate_beta 5); LoRA r32 cumulative, base frozen; 200 optimizer steps, 64 episodes/step over 32 ranks (torchrun data-parallel, global normalizers + SUM grad sync). Probe gate skipped per request. Runs runs/train_opd_qwen35_9b_teacher_all_student_zero_200step/." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":4,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
