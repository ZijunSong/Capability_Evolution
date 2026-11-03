#!/usr/bin/env bash
# Submit Qwen3.5-27B + LoRA adapter (zero-rl 200-step wmimn) zero-component eval.
# Usage: bash submit_qwen35_27b_adapter_zr200_zero_eval_8gpu_20261030.sh <BENCHMARK> <SEED>
#   BENCHMARK ∈ bcplus_full|longsealqa|frames|hotpotqa
#   SEED ∈ 1|2|3  → EVAL_SEED 42|123|2024 (与 base 3-seed / h3eb8 同口径)
# 12 任务 = 4 bench x 3 seed, priority 5, 8gpu 单节点。
# 与 base 3-seed（submit_qwen35_27b_harness1_baseline_eval_8gpu_20261015.sh）同结构，
# 仅 entry 换成 adapter 薄入口（VLLM_EXTRA 注入 --enable-lora，其余复用已验证链）。
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <BENCHMARK> <SEED>" >&2
  echo "  BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}; SEED ∈ {1,2,3}" >&2
  exit 1
fi
BENCHMARK="$1"
SEED="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=20261030

case "$BENCHMARK" in bcplus_full|longsealqa|frames|hotpotqa) ;; *) echo "FATAL: BENCHMARK=$BENCHMARK" >&2; exit 1 ;; esac
case "$SEED" in
  1) EVAL_SEED=42 ;;
  2) EVAL_SEED=123 ;;
  3) EVAL_SEED=2024 ;;
  *) echo "FATAL: SEED=$SEED (want 1|2|3)" >&2; exit 1 ;;
esac

ENTRY="${SCRIPT_DIR}/entry_qwen35_27b_adapter_zr200_zero_bcplus_transfer_eval_8gpu_seed.sh"
test -f "${ENTRY}"

RUN_TAG="${RUN_TAG:-}"
JOB_NAME="trim-harness1-qwen35-27b-adapter-zr200-zero-${BENCHMARK}-eval-8gpu-s${SEED}-${DATE}${RUN_TAG}"
RUN_ID="qwen35_27b_adapter_zr200_zero_${BENCHMARK}_8gpu_s${SEED}_${DATE}${RUN_TAG}"
ADAPTER_PATH="/mnt/songzijun/outputs/qwen35-27b-rl-ddp-200step-zero-recall-32gpu_20261023_r6/checkpoints/rl/step_000200/adapter"
DESC="Harness-1 Qwen3.5-27B + LoRA adapter (zero-rl 200step wmimn step_000200, r=8) zero ${BENCHMARK} seed${SEED}(EVAL_SEED=${EVAL_SEED}). 8gpu 64-worker upstream_api, vLLM enable-lora served as Qwen3.5-27B. 3-seed mean vs base 3-seed. no-think reasoning-parser strip kept, eval lock tf4.57.6."

DESC_LEN=$(printf '%s' "${DESC}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description is ${DESC_LEN} chars (>500)." >&2
  exit 1
fi
echo "[submit] BENCHMARK=${BENCHMARK} SEED=${SEED} JOB_NAME=${JOB_NAME} (desc=${DESC_LEN})"

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
  --env EVAL_MODE=zero --env BENCHMARK="$BENCHMARK" --env EVAL_SEED="$EVAL_SEED" --env RUN_ID="$RUN_ID" \
  --env ADAPTER_PATH="$ADAPTER_PATH" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
