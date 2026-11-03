#!/usr/bin/env bash
# Submit Probe C2 闭环评测：Qwen3.5-9B 1node8gpu，priority 8。
# 等 3 个 8-step adapter 后跑 5 条件 C_select 完整 episode + G/E 聚合。
# DRY=1 干跑。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_probe_c_c2_eval_8gpu_20261102.sh"
test -f "${ENTRY}"
ENTRY_BYTES=$(wc -c < "${ENTRY}")
[ "${ENTRY_BYTES}" -le 22000 ] || { echo "FATAL: entry ${ENTRY_BYTES}B >22000" >&2; exit 1; }

JOB_NAME="probe-c-c2-9b-eval-8gpu-20261102-r2"
OUT="/mnt/songzijun/outputs/probe-c-c2-9b-eval-8gpu_20261102_r2"
PRIORITY="${PRIORITY:-8}"
DESCRIPTION="Probe C2 closed-loop eval: Qwen3.5-9B 1node8gpu. 5 conditions on C_select full episodes (final-answer-recall J): no_update/runtime_m/harnessopd_m/rl_only/same_view; G_m^8/E_m^8 paired bootstrap. Waits for 3x 8-step adapters (12h cap)."

DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description ${DESC_LEN} chars (>500)" >&2
  exit 1
fi
echo "[submit] JOB_NAME=${JOB_NAME} OUT=${OUT} (entry=${ENTRY_BYTES}B, desc=${DESC_LEN}, priority=${PRIORITY})"

CMD=(
  cml custom_train submit
  --job_name "${JOB_NAME}"
  --description "${DESCRIPTION}"
  --access_type PRIVATE
  --tag MemQA-General-V5
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
  --framework pytorch
  --queue_id 2334
  --priority "${PRIORITY}"
  --preemptible=false
  --enable_tensorboard
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]'
  --env PYTHONUNBUFFERED=1
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
  --env OPENAI_API_KEY=**-****-***********-***-*-****-***
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
  --env SCAPE_FORCE_LOCAL_HARMONY=1
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
  --env OUT="${OUT}"
  --env STATES_OUT=/mnt/songzijun/outputs/probe-c-qwen35-9b-coalitions-8gpu_20261101_r3
  --env OUT_HM=/mnt/songzijun/outputs/probe-c-c2-9b-harnessopd_m-8step-16gpu_20261102_r2
  --env OUT_RL=/mnt/songzijun/outputs/probe-c-c2-9b-rl_only-8step-16gpu_20261102_r2
  --env OUT_SV=/mnt/songzijun/outputs/probe-c-c2-9b-same_view-8step-16gpu_20261102_r2
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}'
  --diag_options '[{"name":"HangDetection","enable":false}]'
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}'
  --image_command "$(cat "${ENTRY}")"
)

if [ "${DRY:-0}" = "1" ]; then
  echo "[dry-run] would execute:"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

exec "${CMD[@]}"
