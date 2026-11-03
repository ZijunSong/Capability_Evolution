#!/usr/bin/env bash
# Submit Probe C2 在线确认训练：Qwen3.5-9B 8-step 2node16gpu，priority 8。
# 用法: submit_probe_c_c2_train_16gpu_2node_20261102.sh <COND>
#   COND ∈ harnessopd_m | rl_only | same_view
# DRY=1 干跑。
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "用法: $0 <harnessopd_m|rl_only|same_view>" >&2
  exit 1
fi
COND="$1"
case "${COND}" in harnessopd_m|rl_only|same_view) ;; *) echo "FATAL: COND=${COND}" >&2; exit 1 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_probe_c_c2_train_16gpu_2node_20261102.sh"
test -f "${ENTRY}"
ENTRY_BYTES=$(wc -c < "${ENTRY}")
[ "${ENTRY_BYTES}" -le 22000 ] || { echo "FATAL: entry ${ENTRY_BYTES}B >22000" >&2; exit 1; }

JOB_NAME="probe-c-c2-9b-${COND}-8step-16gpu-2node-20261102-r2"
OUT="/mnt/songzijun/outputs/probe-c-c2-9b-${COND}-8step-16gpu_20261102_r2"
STATES_OUT="/mnt/songzijun/outputs/probe-c-qwen35-9b-coalitions-8gpu_20261101_r3"
PRIORITY="${PRIORITY:-8}"
DESCRIPTION="Probe C2 online confirmation ${COND}: Qwen3.5-9B 8-step 2node16gpu torch_ddp_lora, formal HarnessOPD semantics (rollout refresh/old logp/CISPO/projected-gap). From theta0; train queries exclude C_select/C_audit. harnessopd_m=trim+trieve teacher; rl_only=rl; same_view=zero-placebo. final-answer-recall."

DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description ${DESC_LEN} chars (>500)" >&2
  exit 1
fi
echo "[submit] COND=${COND} JOB_NAME=${JOB_NAME} OUT=${OUT} (entry=${ENTRY_BYTES}B, desc=${DESC_LEN}, priority=${PRIORITY})"

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
  --rdma
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'
  --resource_configs '[{"nodeRole":"worker","nodeNumber":2,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]'
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
  --env COND="${COND}"
  --env STATES_OUT="${STATES_OUT}"
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
