#!/usr/bin/env bash
# Submit Qwen3.5-9B + teacher-ablation 50step LoRA adapter eval (student=zero).
# Usage: bash submit_qwen35_9b_teacher_adapter_eval_8gpu_20261101.sh <MODEL_KEY> <BENCHMARK> <SEED>
#   MODEL_KEY ∈ esp|ccso|trieve|msc   （trieve/trieve 双拼写归一；ccso 训完自动可用）
#   BENCHMARK ∈ bcplus_full|longsealqa|frames|hotpotqa
#   SEED ∈ 1|2|3  → EVAL_SEED 42|123|2024（与 base 3-seed 同口径）
# 1 任务 = 8gpu 单节点，priority 5。3model × 4bench × 3seed = 36 任务（esp/msc/trieve）。
# ADAPTER = outputs/qwen35-9b-trim-teacher-<KEY>-50step-16gpu_20261027/checkpoints/scape_seed/step_000050/adapter
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <MODEL_KEY> <BENCHMARK> <SEED>" >&2
  echo "  MODEL_KEY ∈ {esp,ccso,trieve,msc}; BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}; SEED ∈ {1,2,3}" >&2
  exit 1
fi
MODEL_KEY="$1"; BENCHMARK="$2"; SEED="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=20261101
JFS_ROOT=/mnt/songzijun

# key 归一化（trieve/trieve 同义 → 规范 5 字符形；esp/ccso/msc 原样）
case "${MODEL_KEY}" in
  esp|ccso|msc) ;;
  rieve|trieve) MODEL_KEY="trieve" ;;   # 5 字符规范形（拼写坑：与 6 字符肉眼不可辨）
  *) echo "FATAL: MODEL_KEY=${MODEL_KEY}" >&2; exit 1 ;;
esac
case "$BENCHMARK" in bcplus_full|longsealqa|frames|hotpotqa) ;; *) echo "FATAL: BENCHMARK=$BENCHMARK" >&2; exit 1 ;; esac
case "$SEED" in
  1) EVAL_SEED=42 ;;
  2) EVAL_SEED=123 ;;
  3) EVAL_SEED=2024 ;;
  *) echo "FATAL: SEED=$SEED (want 1|2|3)" >&2; exit 1 ;;
esac

# adapter 路径解析（trieve 目录拼写坑：glob *-50step 精确后缀 + 排除 _failed_rN/_stopped_rN 归档）
OUT_DIR=$(ls -d "${JFS_ROOT}"/outputs/qwen35-9b-trim-teacher-*"-50step-16gpu_20261027" 2>/dev/null \
  | grep -vE '_r[0-9]+$' \
  | grep -F "${MODEL_KEY}" | head -1 || true)
if [ -z "${OUT_DIR}" ]; then
  # trieve 拼写双形兜底：按 *ieve* 取唯一非归档目录
  OUT_DIR=$(ls -d "${JFS_ROOT}"/outputs/qwen35-9b-trim-teacher-*ieve"-50step-16gpu_20261027" 2>/dev/null \
    | grep -vE '_r[0-9]+$' | head -1 || true)
fi
ADAPTER_PATH="${OUT_DIR}/checkpoints/scape_seed/step_000050/adapter"
test -f "${ADAPTER_PATH}/adapter_config.json" || { echo "FATAL: no adapter at ${ADAPTER_PATH}" >&2; exit 1; }
test -f "${ADAPTER_PATH}/adapter_model.safetensors"

ENTRY="${SCRIPT_DIR}/entry_qwen35_9b_teacher_adapter_zero_bcplus_transfer_eval_8gpu_seed.sh"
test -f "${ENTRY}"

RUN_TAG="${RUN_TAG:-}"
JOB_NAME="trim-harness1-qwen35-9b-teacher50-${MODEL_KEY}-zero-${BENCHMARK}-eval-8gpu-s${SEED}-${DATE}${RUN_TAG}"
RUN_ID="qwen35_9b_teacher50_${MODEL_KEY}_zero_${BENCHMARK}_8gpu_s${SEED}_${DATE}${RUN_TAG}"
DESC="Harness-1 Qwen3.5-9B + teacher-ablation 50step LoRA (${MODEL_KEY} teacher coalition, student=zero) zero ${BENCHMARK} seed${SEED}(EVAL_SEED=${EVAL_SEED}). 8gpu upstream_api, vLLM enable-lora. 3-seed mean vs 9B zero/all baseline. eval-lock tf4.57.6, no-think parser strip kept."

DESC_LEN=$(printf '%s' "${DESC}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description is ${DESC_LEN} chars (>500)." >&2
  exit 1
fi
echo "[submit] KEY=${MODEL_KEY} BENCH=${BENCHMARK} SEED=${SEED} JOB=${JOB_NAME} ADAPTER=${ADAPTER_PATH} (desc=${DESC_LEN})"

cml custom_train submit \
  --job_name "$JOB_NAME" --description "$DESC" \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority "${PRIORITY:-5}" --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* \
  --env CHROMA_API_KEY=local-offline --env CHROMA_DATABASE=local-offline --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env EVAL_MODE=zero --env BENCHMARK="$BENCHMARK" --env EVAL_SEED="$EVAL_SEED" --env RUN_ID="$RUN_ID" \
  --env ADAPTER_PATH="$ADAPTER_PATH" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
