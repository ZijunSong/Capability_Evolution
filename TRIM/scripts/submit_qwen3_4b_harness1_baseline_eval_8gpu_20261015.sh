#!/usr/bin/env bash
# Submit TRIM Harness-1 / Qwen3-4B-Instruct-2507 (bare base, no adapter) baseline eval.
# 复刻 2026-10-02 那组（group1=用户既有表），用于凑三组独立实验均值。
#   用法: submit_qwen3_4b_harness1_baseline_eval_8gpu_20261015.sh <EVAL_MODE> <BENCHMARK> <GROUP>
#     EVAL_MODE  ∈ {zero, all}         # zero=瘦harness(exp_b), all=完整harness(exp_a)
#     BENCHMARK  ∈ {bcplus_full, longsealqa, frames, hotpotqa}
#     GROUP      ∈ {2, 3}               # 第2、3组（第1组=2026-10-02 既有结果表）
# priority=5（用户要求），queue 2334，gm402-80g 8 卡单节点，MemQA-General-V5。
# 与 group1 同一 entry / runtime / 配置，仅 job_name/RUN_ID 加 GROUP+DATE 后缀（fresh 输出目录，避 assert_fresh_eval_dir 坑）。
# all 模式 entry 内部已启 harness-1 verifier（GPU6）+ multi-actor layout（ALL_EVAL_TP=24）；zero 用 multi-8-actor layout（TP=32）。
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "用法: $0 <EVAL_MODE> <BENCHMARK> <GROUP>" >&2
  echo "  EVAL_MODE ∈ {zero, all}; BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}; GROUP ∈ {2,3}" >&2
  exit 1
fi
EVAL_MODE="$1"; BENCHMARK="$2"; GROUP="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y%m%d)

case "$EVAL_MODE" in zero|all) ;; *) echo "FATAL: EVAL_MODE=$EVAL_MODE" >&2; exit 1 ;; esac
case "$BENCHMARK" in bcplus_full|longsealqa|frames|hotpotqa) ;; *) echo "FATAL: BENCHMARK=$BENCHMARK" >&2; exit 1 ;; esac
case "$GROUP" in 2|3) ;; *) echo "FATAL: GROUP=$GROUP (only 2,3; group1=2026-10-02 existing table)" >&2; exit 1 ;; esac

ENTRY="${SCRIPT_DIR}/entry_qwen3_4b_harness_zero_all_bcplus_transfer_eval_8gpu.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness1-qwen3-4b-harness-${EVAL_MODE}-${BENCHMARK}-eval-8gpu-g${GROUP}-${DATE}"
RUN_ID="qwen3_4b_harness_${EVAL_MODE}_${BENCHMARK}_8gpu_g${GROUP}_${DATE}_memqa_v5"
DESC="Harness-1 Qwen3-4B-Instruct baseline (no adapter) ${EVAL_MODE} ${BENCHMARK} group${GROUP}. 8gpu. base model. 1 of 3 groups (group1=20261002 existing table)."

echo "[submit] EVAL_MODE=${EVAL_MODE} BENCHMARK=${BENCHMARK} GROUP=${GROUP} JOB_NAME=${JOB_NAME} RUN_ID=${RUN_ID}"

cml custom_train submit \
  --job_name "$JOB_NAME" --description "$DESC" \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 5 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* \
  --env CHROMA_API_KEY=local-offline --env CHROMA_DATABASE=local-offline --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env EVAL_MODE="$EVAL_MODE" --env BENCHMARK="$BENCHMARK" --env RUN_ID="$RUN_ID" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
