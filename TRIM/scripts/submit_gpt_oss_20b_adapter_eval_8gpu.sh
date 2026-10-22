#!/usr/bin/env bash
# 通用 adapter eval submit：给定任意 adapter 路径 + component，跑 4 benchmark eval。
# 用于补充历史实验的重复 eval（如 RL recall-only bf16, all-trim run2_bf16 的 step100/200 adapter）。
# 用法: submit_gpt_oss_20b_adapter_eval_8gpu.sh <ADAPTER_PATH> <COMPONENT> <BENCHMARK> <RUN_TAG>
#   ADAPTER_PATH  adapter 目录绝对路径（含 adapter_config.json）
#   COMPONENT     zero|all（zero=瘦harness, all=完整harness+verifier）
#   BENCHMARK     bcplus_full|longsealqa|frames|hotpotqa
#   RUN_TAG       任意短标签（注入 RUN_ID 保证 fresh 目录）
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "用法: $0 <ADAPTER_PATH> <COMPONENT> <BENCHMARK> <RUN_TAG>" >&2
  echo "  COMPONENT ∈ {zero,all}; BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}" >&2
  exit 1
fi
ADAPTER_PATH="$1"
COMPONENT="$2"
BENCHMARK="$3"
RUN_TAG="$4"

case "$COMPONENT" in
  zero|all) ;;
  *) echo "FATAL: COMPONENT=$COMPONENT 不在 {zero,all}" >&2; exit 1 ;;
esac
case "$BENCHMARK" in
  bcplus_full|longsealqa|frames|hotpotqa) ;;
  *) echo "FATAL: BENCHMARK=$BENCHMARK 不在 {bcplus_full,longsealqa,frames,hotpotqa}" >&2; exit 1 ;;
esac

test -f "${ADAPTER_PATH}/adapter_config.json" || { echo "FATAL: adapter_config.json not found at ${ADAPTER_PATH}" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y%m%d)

# 用 adapter eval entry（COMPONENT 由 env 注入，默认 zero）
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_adapter_step100_bcplus_transfer_eval_8gpu.sh"
test -f "${ENTRY}"

# 从 adapter 路径提取一个短标签做 job_name（取最后两段）
ADAPTER_TAG=$(echo "${ADAPTER_PATH}" | awk -F/ '{print $(NF-1)"_"$NF}')
JOB_NAME="trim-adapter-eval-${ADAPTER_TAG}-${COMPONENT}-${BENCHMARK}-8gpu-${DATE}-${RUN_TAG}"
RUN_ID="gpt_oss_20b_adapter_${ADAPTER_TAG}_${COMPONENT}_${BENCHMARK}_${DATE}_${RUN_TAG}"

echo "[submit] ADAPTER_PATH=${ADAPTER_PATH}"
echo "[submit] COMPONENT=${COMPONENT} BENCHMARK=${BENCHMARK} RUN_TAG=${RUN_TAG}"
echo "[submit] ENTRY=${ENTRY}"
echo "[submit] JOB_NAME=${JOB_NAME}"
echo "[submit] RUN_ID=${RUN_ID}"

cml custom_train submit \
  --job_name "$JOB_NAME" \
  --description "TRIM adapter eval: adapter=${ADAPTER_PATH} COMPONENT=${COMPONENT} BENCHMARK=${BENCHMARK}. bf16-dequant base. 8gpu. 重复实验 RUN_TAG=${RUN_TAG}." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 5 \
  --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* \
  --env CHROMA_API_KEY=local-offline --env CHROMA_DATABASE=local-offline --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env ADAPTER_PATH="${ADAPTER_PATH}" \
  --env COMPONENT="${COMPONENT}" \
  --env BENCHMARK="${BENCHMARK}" \
  --env RUN_ID="${RUN_ID}" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
