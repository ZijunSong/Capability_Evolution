#!/usr/bin/env bash
# 提交 retrain1 的训练段（4node 32gpu）。每次只训到 EVAL_TARGET_STEP，配合 evalhook entry。
# 用法: submit_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261003_retrain1.sh <TARGET_STEP> <RESUME>
#   TARGET_STEP ∈ {50,100,150,200}：本段训到第几步
#   RESUME      ∈ {0,1}：0=首段 fresh start，1=续段从 latest ckpt resume
# orchestrator 顺序调用：50 0 → 100 1 → 150 1 → 200 1，每段后跑 4 benchmark eval。
# fresh OUT 目录 ..._20261003_retrain1，与原 7nzzn ..._20260925 隔离。
# checkpoint 在 JuiceFS 上持久，kill job 不丢；--resume 自动找 latest symlink。
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "用法: $0 <TARGET_STEP> <RESUME>" >&2
  echo "  TARGET_STEP ∈ {50,100,150,200}; RESUME ∈ {0,1}" >&2
  exit 1
fi
TARGET="$1"
RESUME="$2"

case "$TARGET" in
  50|100|150|200) ;;
  *) echo "FATAL: TARGET_STEP=$TARGET 不在 {50,100,150,200}" >&2; exit 1 ;;
esac
case "$RESUME" in
  0|1) ;;
  *) echo "FATAL: RESUME=$RESUME 不在 {0,1}" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261003_evalhook.sh"
test -f "${ENTRY}"

OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261003_retrain1
if [ "$RESUME" = "1" ]; then
  SUFFIX="-resume-to-${TARGET}"
  DESC_TAG="resume from latest to step ${TARGET}"
else
  SUFFIX="-to-${TARGET}"
  DESC_TAG="fresh train to step ${TARGET}"
fi

JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-rl-ddp-200step-zero-32gpu-4node-20261003-retrain1${SUFFIX}"

echo "[submit] TARGET=${TARGET} RESUME=${RESUME}"
echo "[submit] ENTRY=${ENTRY}"
echo "[submit] OUT=${OUT}"
echo "[submit] JOB_NAME=${JOB_NAME}"

cml custom_train submit \
  --job_name "$JOB_NAME" \
  --description "TRIM retrain1: GPT-OSS-20B zero-RL 4node32gpu torch_ddp_lora. ${DESC_TAG}. mb1 gpu-mem0.90. Eval hooks at 0/50/100/150/200 (step0 eval runs before training)." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 2 \
  --preemptible=false \
  --rdma \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":4,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
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
  --env OUT="$OUT" \
  --env EVAL_TARGET_STEP="$TARGET" \
  --env RESUME="$RESUME" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
