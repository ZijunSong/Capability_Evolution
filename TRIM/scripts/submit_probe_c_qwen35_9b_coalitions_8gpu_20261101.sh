#!/usr/bin/env bash
# Submit Probe C 可学习性实验：Qwen3.5-9B 四组件联盟 1node8gpu，priority 8（用户要求）。
# 链路 audit→collect(真实 H0 状态,SEC train pool)→cache(C0 漏斗+冻结 Teacher)→
# screen(C1 4→8 微更新)。状态一次采集四联盟共用；Student=H0，Teacher=联盟精确 add pair。
# 用法: submit_probe_c_qwen35_9b_coalitions_8gpu_20261101.sh   （DRY=1 干跑）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_probe_c_qwen35_9b_coalitions_8gpu_20261101.sh"
test -f "${ENTRY}"
ENTRY_BYTES=$(wc -c < "${ENTRY}")
[ "${ENTRY_BYTES}" -le 22000 ] || { echo "FATAL: entry ${ENTRY_BYTES}B >22000（imageCommand SQL 铁律）" >&2; exit 1; }

# r8：r7 已出 C1@4（trieve：R_N≈+0.04 双 seed、R_D≈-0.001、DEPRIORITIZE）；按契约
# READY 候选默认 4→8 同轨迹全跑（16 才按 borderline 门控），r8 补全 8 步轨迹。
JOB_NAME="probe-c-qwen35-9b-coalitions-8gpu-20261101-r8"
OUT="/mnt/songzijun/outputs/probe-c-qwen35-9b-coalitions-8gpu_20261101_r3"
PRIORITY="${PRIORITY:-8}"
DESCRIPTION="Probe C learnability: Qwen3.5-9B 4 coalitions (esp/ccso/trieve/msc) 1node8gpu. Real H0 states (SEC train pool, split-before-sample), C0 funnel + frozen-teacher cache + C1 4-8 micro-updates (frozen projected-gap, per-candidate theta0 reset). Student=H0, Teacher=exact add pair. Unregistered members reported UNREGISTERED."

DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description is ${DESC_LEN} chars (>500)." >&2
  exit 1
fi
echo "[submit] JOB_NAME=${JOB_NAME} OUT=${OUT} (desc=${DESC_LEN} chars, entry=${ENTRY_BYTES}B, priority=${PRIORITY})"

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
  --env TRIM_GPU_KEEPALIVE=0
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
  --env OPENAI_API_KEY=**-****-***********-***-*-****-***
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
  --env SCAPE_FORCE_LOCAL_HARMONY=1
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
  --env OUT="${OUT}"
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
