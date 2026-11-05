#!/usr/bin/env bash
# Submit Probe C R2 C1（guide §16 R2）：1node8gpu pri5。
# 复用 R1 cache + 旧 r3 状态；双 seed 4→8 真实 checkpoint。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_probe_c_r2_c1_8gpu_20261104.sh"
test -f "${ENTRY}"
ENTRY_BYTES=$(wc -c < "${ENTRY}")
[ "${ENTRY_BYTES}" -le 22000 ] || { echo "FATAL: entry ${ENTRY_BYTES}B >22000" >&2; exit 1; }

OLD_RUN="${OLD_RUN:-/mnt/songzijun/outputs/probe-c-qwen35-9b-coalitions-8gpu_20261101_r3}"
R1_DIR="${R1_DIR:-/mnt/songzijun/outputs/probe-c-r1-smoke-8gpu_20261104_r3/Qwen_Qwen3.5-9B}"
OUT="${OUT:-/mnt/songzijun/outputs/probe-c-r2-c1-8gpu_20261104}"
PRIORITY="${PRIORITY:-5}"
JOB_NAME="probe-c-r2-c1-qwen35-9b-8gpu-20261104"
DESCRIPTION="Probe C R2 repaired C1 (guide R2): auto_populate_first_search, seeds 11/23, 4->8 via persistent sampler session (4+4==8 contract). Real atomic checkpoints pre/4/8 (weights+optimizer+RNG). Panel metrics pre/4/8 with full generation records. Fixed-budget summary at 8 across seeds. No sentence_compress (support too narrow; diagnose/Commit D first)."

DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then echo "FATAL: desc ${DESC_LEN}>500" >&2; exit 1; fi
echo "[submit] R2 C1 OUT=${OUT} R1=${R1_DIR} entry=${ENTRY_BYTES}B pri=${PRIORITY}"

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
  --env R1_DIR="${R1_DIR}" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
