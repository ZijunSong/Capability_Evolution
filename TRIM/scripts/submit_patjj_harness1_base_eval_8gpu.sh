#!/usr/bin/env bash
# 参数化 submit：pat-jj/harness-1 base model eval（被测 model = pat-jj_harness-1-full/harness-1）。
# 用法: submit_patjj_harness1_base_eval_8gpu.sh <EVAL_MODE> <BENCHMARK> <RUN_TAG>
#   EVAL_MODE  ∈ {zero,all}（zero=瘦 harness，all=完整 harness + verifier）
#   BENCHMARK  ∈ {bcplus_full,longsealqa,frames,hotpotqa}
#   RUN_TAG    任意短标签（注入 RUN_ID 保证 fresh 目录，避 assert_fresh_eval_dir）
# 与旧 2026-09-14 patjj eval 同口径（model_name=pat-jj_harness-1-full/harness-1, api_model=harness-1）。
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "用法: $0 <EVAL_MODE> <BENCHMARK> <RUN_TAG>" >&2
  echo "  EVAL_MODE ∈ {zero,all}; BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}" >&2
  exit 1
fi
EVAL_MODE="$1"
BENCHMARK="$2"
RUN_TAG="$3"

case "$EVAL_MODE" in
  zero|all) ;;
  *) echo "FATAL: EVAL_MODE=$EVAL_MODE 不在 {zero,all}" >&2; exit 1 ;;
esac
case "$BENCHMARK" in
  bcplus_full|longsealqa|frames|hotpotqa) ;;
  *) echo "FATAL: BENCHMARK=$BENCHMARK 不在 {bcplus_full,longsealqa,frames,hotpotqa}" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_patjj_harness1_zero_all_bcplus_transfer_eval_8gpu.sh"
test -f "${ENTRY}"

DATE=$(date +%Y%m%d)
JOB_NAME="trim-patjj-harness1-${EVAL_MODE}-${BENCHMARK}-8gpu-${DATE}-${RUN_TAG}"
RUN_ID="patjj_harness1_${EVAL_MODE}_${BENCHMARK}_${DATE}_${RUN_TAG}"

echo "[submit] EVAL_MODE=${EVAL_MODE} BENCHMARK=${BENCHMARK} RUN_TAG=${RUN_TAG}"
echo "[submit] ENTRY=${ENTRY}"
echo "[submit] JOB_NAME=${JOB_NAME}"
echo "[submit] RUN_ID=${RUN_ID}"

cml custom_train submit \
  --job_name "$JOB_NAME" \
  --description "TRIM pat-jj/harness-1 base ${EVAL_MODE} eval: pat-jj_harness-1-full/harness-1 (gpt-oss-20b arch bf16 dequant). ${BENCHMARK} 8gpu. 与旧 2026-09-14 eval 同口径, zero/all 各 3 次重复求均值." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 8 \
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
  --env EVAL_MODE="$EVAL_MODE" \
  --env BENCHMARK="$BENCHMARK" \
  --env RUN_ID="$RUN_ID" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
