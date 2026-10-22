#!/usr/bin/env bash
# 提交 Harness-G / Qwen3-4B-Instruct-2507 / rl / zero 训练段（4node 32gpu）。
# 用法: submit_harness_g_qwen3_4b_instruct_rl_zero_32gpu_4node_20261008.sh <TARGET_STEP> <RESUME>
#   TARGET_STEP ∈ {50,100,150,200}；RESUME ∈ {0,1}
# orchestrator 顺序调用：50 0 → 100 1 → 150 1 → 200 1，每段后跑 4 benchmark eval。
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "用法: $0 <TARGET_STEP> <RESUME>" >&2; echo "  TARGET_STEP ∈ {50,100,150,200}; RESUME ∈ {0,1}" >&2; exit 1
fi
TARGET="$1"; RESUME="$2"
case "$TARGET" in 50|100|150|200) ;; *) echo "FATAL: TARGET_STEP=$TARGET 不在 {50,100,150,200}" >&2; exit 1 ;; esac
case "$RESUME" in 0|1) ;; *) echo "FATAL: RESUME=$RESUME 不在 {0,1}" >&2; exit 1 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_harness_g_qwen3_4b_instruct_rl_zero_32gpu_4node_20261008_evalhook.sh"
test -f "${ENTRY}"

OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness_g_qwen3_4b_instruct_rl_zero_32gpu_4node_20261008
if [ "$RESUME" = "1" ]; then
  SUFFIX="-resume-to-${TARGET}"; DESC_TAG="resume from latest to step ${TARGET}"
else
  SUFFIX="-to-${TARGET}"; DESC_TAG="fresh train to step ${TARGET}"
fi
JOB_NAME="trim-harness-g-qwen3-4b-instruct-rl-zero-32gpu-4node-20261008${SUFFIX}"

echo "[submit] TARGET=${TARGET} RESUME=${RESUME} ENTRY=${ENTRY} OUT=${OUT} JOB_NAME=${JOB_NAME}"

cml custom_train submit \
  --job_name "$JOB_NAME" \
  --description "Harness-G / Qwen3-4B-Instruct-2507 rl/zero 4node32gpu torch_ddp_lora. ${DESC_TAG}. mb4 gpu-mem0.90. Harness-G legacy_local graph runtime, bcplus_train_664, corpus graph. Eval hooks at 0/50/100/150/200." \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch \
  --queue_id 2334 \
  --priority 5 \
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
