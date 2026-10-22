#!/usr/bin/env bash
# 参数化 eval submit：Harness-G / Qwen3-4B-Instruct-2507 trim/all 训练的 5 ckpt × 4 benchmark eval。
# 用法: submit_harness_g_qwen3_4b_instruct_trim_all_eval_8gpu_20261008.sh <STEP> <BENCHMARK> [RUN_TAG]
#   STEP ∈ {0,50,100,150,200}；BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}
# EVAL_MODE=all（Harness-G all = 8 graph-menu components，无 verifier）。priority=2，queue 2334，8 卡。
set -euo pipefail
if [ "$#" -lt 2 ]; then echo "用法: $0 <STEP> <BENCHMARK> [RUN_TAG]" >&2; exit 1; fi
STEP="$1"; BENCHMARK="$2"; RUN_TAG="${3:-trimall}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y%m%d)
TRAIN_OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness_g_qwen3_4b_instruct_trim_all_32gpu_4node_20261008"
case "$STEP" in 0|50|100|150|200) ;; *) echo "FATAL: STEP=$STEP" >&2; exit 1 ;; esac
case "$BENCHMARK" in bcplus_full|longsealqa|frames|hotpotqa) ;; *) echo "FATAL: BENCHMARK=$BENCHMARK" >&2; exit 1 ;; esac

ENTRY="${SCRIPT_DIR}/entry_harness_g_eval_adapter_8gpu_20261008.sh"
if [ "$STEP" = "0" ]; then
  JOB_NAME="trim-harness-g-eval-qwen3-4b-instruct-trim-all-bare-${BENCHMARK}-8gpu-${DATE}"
  ADAPTER_PATH=""
  DESC="Harness-G eval: Qwen3-4B-Instruct-2507 bare (step0, no adapter) trim/all baseline. 8gpu. EVAL_MODE=all."
else
  JOB_NAME="trim-harness-g-eval-qwen3-4b-instruct-trim-all-step${STEP}-${BENCHMARK}-8gpu-${DATE}"
  STEP_PAD=$(printf '%03d' "$STEP")
  ADAPTER_PATH="${TRAIN_OUT}/checkpoints/rl/step_000${STEP_PAD}/adapter"
  DESC="Harness-G eval: Qwen3-4B-Instruct-2507 step${STEP} adapter (trim/all). base+step${STEP} LoRA. 8gpu. EVAL_MODE=all."
fi
test -f "${ENTRY}"
RUN_ID="qwen3_4b_instruct_trim_all_step${STEP}_${BENCHMARK}_${DATE}_${RUN_TAG}"

echo "[submit] STEP=${STEP} BENCHMARK=${BENCHMARK} RUN_TAG=${RUN_TAG} ENTRY=${ENTRY} JOB_NAME=${JOB_NAME} RUN_ID=${RUN_ID}"
[ "$STEP" != "0" ] && echo "[submit] ADAPTER_PATH=${ADAPTER_PATH}"

cml custom_train submit \
  --job_name "$JOB_NAME" --description "$DESC" \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 2 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env MODEL_KIND=qwen3_4b_instruct \
  --env EVAL_MODE=all \
  --env BENCHMARK="$BENCHMARK" \
  --env RUN_ID="$RUN_ID" \
  ${ADAPTER_PATH:+--env ADAPTER_PATH="$ADAPTER_PATH"} \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
