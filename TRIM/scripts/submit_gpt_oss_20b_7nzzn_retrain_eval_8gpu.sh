#!/usr/bin/env bash
# 参数化 eval submit：重训 7nzzn retrain1 的 5 个 checkpoint × 4 benchmark eval。
# 用法: submit_gpt_oss_20b_7nzzn_retrain_eval_8gpu.sh <STEP> <BENCHMARK> <RUN_TAG>
#   STEP      ∈ {0,50,100,150,200}（0=bare 裸模型，无 adapter）
#   BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}
#   RUN_TAG   任意短标签（注入 RUN_ID 保证 fresh 目录，避 assert_fresh_eval_dir 坑）
#
# 设计：
#   - step 0（bare）用 entry_gpt_oss_20b_harness_zero_all_bcplus_transfer_eval_8gpu.sh（EVAL_MODE=zero，无 adapter，bf16-dequant base）。
#   - step 50/100/150/200 用 entry_gpt_oss_20b_adapter_step100_bcplus_transfer_eval_8gpu.sh（ADAPTER_PATH env 覆盖指向 step_N/adapter）。
#   - 4 benchmark 共用同一 entry，BENCHMARK env 路由 index/corpus（entry 内 case 已支持）。
#   - priority=2（eval 不抢训练资源），queue 2334，gm402-80g 8 卡单节点。
# 复用：与原 4×4 eval（submit_gpt_oss_20b_adapter_step{100,200}_7nzzn_*_eval_8gpu_*fix3.sh）同口径，
#      只是参数化了 STEP/BENCHMARK，并指向 retrain1 的新 OUT 目录。
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "用法: $0 <STEP> <BENCHMARK> [RUN_TAG]" >&2
  echo "  STEP ∈ {0,50,100,150,200}; BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}" >&2
  exit 1
fi
STEP="$1"
BENCHMARK="$2"
RUN_TAG="${3:-retrain1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y%m%d)
RETRAIN_OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261003_retrain1"

# 校验 STEP
case "$STEP" in
  0|50|100|150|200) ;;
  *) echo "FATAL: STEP=$STEP 不在 {0,50,100,150,200}" >&2; exit 1 ;;
esac
# 校验 BENCHMARK
case "$BENCHMARK" in
  bcplus_full|longsealqa|frames|hotpotqa) ;;
  *) echo "FATAL: BENCHMARK=$BENCHMARK 不在 {bcplus_full,longsealqa,frames,hotpotqa}" >&2; exit 1 ;;
esac

if [ "$STEP" = "0" ]; then
  # bare：无 adapter，用 harness zero entry（multi-8-actor layout，EVAL_MODE=zero）
  ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_harness_zero_all_bcplus_transfer_eval_8gpu.sh"
  JOB_NAME="trim-retrain1-gpt-oss-20b-bare-zero-${BENCHMARK}-8gpu-${DATE}"
  EXTRA_ENV="--env EVAL_MODE=zero"
  DESC="TRIM retrain1 ${BENCHMARK} eval: GPT-OSS-20B bare (no adapter) step0 baseline. bf16-dequant base. 8gpu. EVAL_MODE=zero multi-8-actor."
else
  # adapter：step50/100/150/200，ADAPTER_PATH 指向 retrain1 的 step_N/adapter
  ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_adapter_step100_bcplus_transfer_eval_8gpu.sh"
  JOB_NAME="trim-retrain1-gpt-oss-20b-adapter-step${STEP}-${BENCHMARK}-8gpu-${DATE}"
  STEP_PAD=$(printf '%03d' "$STEP")
  ADAPTER_PATH="${RETRAIN_OUT}/checkpoints/rl/step_000${STEP_PAD}/adapter"
  EXTRA_ENV="--env ADAPTER_PATH=${ADAPTER_PATH}"
  DESC="TRIM retrain1 ${BENCHMARK} eval: GPT-OSS-20B step${STEP} adapter (7nzzn retrain1). bf16-dequant base+step${STEP} LoRA. 8gpu. COMPONENT=zero."
fi
test -f "${ENTRY}"

# fresh RUN_ID（含 DATE+RUN_TAG），避 assert_fresh_eval_dir 坑
RUN_ID="gpt_oss_20b_retrain1_step${STEP}_${BENCHMARK}_${DATE}_${RUN_TAG}"

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
