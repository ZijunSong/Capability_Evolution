#!/usr/bin/env bash
# 参数化 eval submit：all-rl retrain1 的 5 个 checkpoint × 4 benchmark eval。
# 与 submit_gpt_oss_20b_7nzzn_retrain_eval_8gpu.sh 同构，区别：
#   - TRAIN_OUT 指向 all-rl 训练目录（..._all_..._retrain1）
#   - step50/100/150/200 的 adapter eval 传 COMPONENT=all（启用 harness-1 verifier，与训练 component=all 对齐）
#   - step0 bare eval 用 EVAL_MODE=all（harness-1 verifier + all 组件）
# 用法: submit_gpt_oss_20b_all_rl_retrain_eval_8gpu.sh <STEP> <BENCHMARK> [RUN_TAG]
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "用法: $0 <STEP> <BENCHMARK> [RUN_TAG]" >&2
  echo "  STEP ∈ {0,50,100,150,200}; BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}" >&2
  exit 1
fi
STEP="$1"
BENCHMARK="$2"
RUN_TAG="${3:-allretrain1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y%m%d)
RETRAIN_OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_200step_all_32gpu_4node_20261003_retrain1"

case "$STEP" in
  0|50|100|150|200) ;;
  *) echo "FATAL: STEP=$STEP 不在 {0,50,100,150,200}" >&2; exit 1 ;;
esac
case "$BENCHMARK" in
  bcplus_full|longsealqa|frames|hotpotqa) ;;
  *) echo "FATAL: BENCHMARK=$BENCHMARK 不在 {bcplus_full,longsealqa,frames,hotpotqa}" >&2; exit 1 ;;
esac

if [ "$STEP" = "0" ]; then
  # bare：无 adapter，用 harness zero/all entry，EVAL_MODE=all 启用 verifier
  ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_harness_zero_all_bcplus_transfer_eval_8gpu.sh"
  JOB_NAME="trim-allretrain1-gpt-oss-20b-bare-all-${BENCHMARK}-8gpu-${DATE}"
  EXTRA_ENV="--env EVAL_MODE=all"
  DESC="TRIM all-rl retrain1 ${BENCHMARK} eval: GPT-OSS-20B bare (no adapter) step0 baseline. bf16-dequant base. EVAL_MODE=all (harness-1 verifier)."
else
  # adapter：step50/100/150/200，ADAPTER_PATH 指向 all-rl 的 step_N/adapter，COMPONENT=all 启用 verifier
  ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_adapter_step100_bcplus_transfer_eval_8gpu.sh"
  JOB_NAME="trim-allretrain1-gpt-oss-20b-adapter-step${STEP}-${BENCHMARK}-8gpu-${DATE}"
  STEP_PAD=$(printf '%03d' "$STEP")
  ADAPTER_PATH="${RETRAIN_OUT}/checkpoints/rl/step_000${STEP_PAD}/adapter"
  EXTRA_ENV="--env ADAPTER_PATH=${ADAPTER_PATH} --env COMPONENT=all"
  DESC="TRIM all-rl retrain1 ${BENCHMARK} eval: GPT-OSS-20B step${STEP} adapter (all-rl retrain1). bf16-dequant base+step${STEP} LoRA. COMPONENT=all (harness-1 verifier)."
fi
test -f "${ENTRY}"

RUN_ID="gpt_oss_20b_allretrain1_step${STEP}_${BENCHMARK}_${DATE}_${RUN_TAG}"

echo "[submit] STEP=${STEP} BENCHMARK=${BENCHMARK} RUN_TAG=${RUN_TAG}"
echo "[submit] ENTRY=${ENTRY}"
echo "[submit] JOB_NAME=${JOB_NAME}"
echo "[submit] RUN_ID=${RUN_ID}"
[ "$STEP" != "0" ] && echo "[submit] ADAPTER_PATH=${ADAPTER_PATH}"

cml custom_train submit \
  --job_name "$JOB_NAME" \
  --description "$DESC" \
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
  --env BENCHMARK="$BENCHMARK" \
  --env RUN_ID="$RUN_ID" \
  $EXTRA_ENV \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
