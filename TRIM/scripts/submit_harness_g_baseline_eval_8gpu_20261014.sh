#!/usr/bin/env bash
# Harness-G baseline 测评提交脚本（bare 模型，无 adapter，base model baseline）。
# 用法: submit_harness_g_baseline_eval_8gpu_20261014.sh <MODEL_KIND> <EVAL_MODE> <BENCHMARK> <GROUP>
#   MODEL_KIND ∈ {qwen3_4b_instruct, qwen3_4b_thinking, gpt_oss_20b}
#   EVAL_MODE  ∈ {zero, all}         # zero=瘦harness(exp_b), all=完整harness(exp_a)
#   BENCHMARK  ∈ {bcplus_full, longsealqa, frames, hotpotqa}
#   GROUP      ∈ {2, 3}               # 第 2、3 组独立实验（第 1 组用既有 baseline 表）
# priority=8（baseline 优先调度），queue 2334，gm402-80g 8 卡单节点，MemQA-General-V5。
# 每个 eval 用 fresh RUN_ID（含 GROUP+DATE），避 assert_fresh_eval_dir 坑。
# gpt_oss_20b eval 用 bf16-dequant base（CLAUDE.md 铁律避 mxfp4+eval PTX）；qwen3 用原版 dense。
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo "用法: $0 <MODEL_KIND> <EVAL_MODE> <BENCHMARK> <GROUP>" >&2
  echo "  MODEL_KIND ∈ {qwen3_4b_instruct, qwen3_4b_thinking, gpt_oss_20b}" >&2
  echo "  EVAL_MODE ∈ {zero, all}; BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}; GROUP ∈ {2,3}" >&2
  exit 1
fi
MODEL_KIND="$1"; EVAL_MODE="$2"; BENCHMARK="$3"; GROUP="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y%m%d)

case "$MODEL_KIND" in qwen3_4b_instruct|qwen3_4b_thinking|gpt_oss_20b) ;; *) echo "FATAL: MODEL_KIND=$MODEL_KIND" >&2; exit 1 ;; esac
case "$EVAL_MODE" in zero|all) ;; *) echo "FATAL: EVAL_MODE=$EVAL_MODE" >&2; exit 1 ;; esac
case "$BENCHMARK" in bcplus_full|longsealqa|frames|hotpotqa) ;; *) echo "FATAL: BENCHMARK=$BENCHMARK" >&2; exit 1 ;; esac
case "$GROUP" in 2|3|4) ;; *) echo "FATAL: GROUP=$GROUP (only 2,3,4; group 1 was the old baseline table, now ignored)" >&2; exit 1 ;; esac

ENTRY="${SCRIPT_DIR}/entry_harness_g_eval_adapter_8gpu_20261008.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness-g-baseline-${MODEL_KIND}-${EVAL_MODE}-${BENCHMARK}-8gpu-g${GROUP}-${DATE}-v2"
RUN_ID="baseline_${MODEL_KIND}_${EVAL_MODE}_${BENCHMARK}_g${GROUP}_${DATE}_v2"
DESC="Harness-G baseline eval group${GROUP}: ${MODEL_KIND} bare (no adapter) ${EVAL_MODE}. base model. 8gpu. 1 of 3 groups (group1=existing table)."

# baseline 不带 adapter：entry 的 if [ -n ADAPTER_PATH ] 为 false 跳过 adapter preflight。
# ADAPTER_PATH 不设 env（entry 默认空），不传空值 env（cml 拒空 env value）。

echo "[submit] MODEL_KIND=${MODEL_KIND} EVAL_MODE=${EVAL_MODE} BENCHMARK=${BENCHMARK} GROUP=${GROUP} JOB_NAME=${JOB_NAME} RUN_ID=${RUN_ID}"

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
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env MODEL_KIND="$MODEL_KIND" \
  --env EVAL_MODE="$EVAL_MODE" \
  --env BENCHMARK="$BENCHMARK" \
  --env RUN_ID="$RUN_ID" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
