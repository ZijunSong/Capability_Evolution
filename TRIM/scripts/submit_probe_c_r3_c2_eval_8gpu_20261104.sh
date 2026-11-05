#!/usr/bin/env bash
# Submit Probe C R3 C2 eval：1node8gpu pri5。
# 4 条件评测（R3 首轮省略 same_view）：no_update/runtime_m 先跑，等 2 个 adapter。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_probe_c_r3_c2_eval_8gpu_20261104.sh"
test -f "${ENTRY}"
ENTRY_BYTES=$(wc -c < "${ENTRY}")
[ "${ENTRY_BYTES}" -le 22000 ] || { echo "FATAL: entry ${ENTRY_BYTES}B >22000" >&2; exit 1; }

STATES_OUT="${STATES_OUT:-/mnt/songzijun/outputs/probe-c-qwen35-9b-coalitions-8gpu_20261101_r3}"
OUT_HM="${OUT_HM:-/mnt/songzijun/outputs/probe-c-c2-9b-harnessopd_m-8step-16gpu_20261104_r3}"
OUT_RL="${OUT_RL:-/mnt/songzijun/outputs/probe-c-c2-9b-rl_only-8step-16gpu_20261104_r3}"
OUT="${OUT:-/mnt/songzijun/outputs/probe-c-r3-c2-eval-8gpu_20261104}"
PRIORITY="${PRIORITY:-5}"
JOB_NAME="probe-c-r3-c2-eval-qwen35-9b-8gpu-20261104"
DESCRIPTION="Probe C R3 C2 eval (guide R3 minimal matrix): C_select 64-query full episodes, 4 conditions (no_update/runtime_m now; harnessopd_m/rl_only after READY_MANIFEST adapters; same_view omitted). G_m^8/E_m^8 + runtime_gain/rl_gain, paired bootstrap 2000, strict panel identity (no silent intersection). H_m mask from audit manifest."

DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then echo "FATAL: desc ${DESC_LEN}>500" >&2; exit 1; fi
echo "[submit] R3 C2 eval OUT=${OUT} HM=${OUT_HM} RL=${OUT_RL} entry=${ENTRY_BYTES}B"

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
  --env STATES_OUT="${STATES_OUT}" \
  --env OUT_HM="${OUT_HM}" \
  --env OUT_RL="${OUT_RL}" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
