#!/usr/bin/env bash
# Submit Probe C R1 measurement smoke（guide §16 R1）：1node8gpu pri5。
# 复用旧 r3 真实 H0 状态（OLD_RUN），不重采集；只跑双单成员测量链验证。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_probe_c_r1_smoke_8gpu_20261104.sh"
test -f "${ENTRY}"
ENTRY_BYTES=$(wc -c < "${ENTRY}")
[ "${ENTRY_BYTES}" -le 22000 ] || { echo "FATAL: entry ${ENTRY_BYTES}B >22000" >&2; exit 1; }

OLD_RUN="${OLD_RUN:-/mnt/songzijun/outputs/probe-c-qwen35-9b-coalitions-8gpu_20261101_r3}"
OUT="${OUT:-/mnt/songzijun/outputs/probe-c-r1-smoke-8gpu_20261104}"
PRIORITY="${PRIORITY:-5}"
JOB_NAME="probe-c-r1-smoke-qwen35-9b-8gpu-20261104"
DESCRIPTION="Probe C R1 measurement smoke (guide R1): reuse r3 real-H0 states, 16+16 queries stratified, two single members (auto_populate_first_search + sentence_compress). Checks: same-state target in K, repeat-score consistency, teacher view provenance, 128vs2048 budget parity, zero cross-split leakage. No training. Gate: PREFLIGHT_OK.json"

DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then echo "FATAL: desc ${DESC_LEN}>500" >&2; exit 1; fi
echo "[submit] R1 smoke OUT=${OUT} OLD_RUN=${OLD_RUN} entry=${ENTRY_BYTES}B pri=${PRIORITY}"

cml custom_train submit \
  --job_name "$JOB_NAME" --description "$DESCRIPTION" \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority "${PRIORITY}" --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env SCAPE_FORCE_LOCAL_HARMONY=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env OUT="${OUT}" \
  --env OLD_RUN="${OLD_RUN}" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
