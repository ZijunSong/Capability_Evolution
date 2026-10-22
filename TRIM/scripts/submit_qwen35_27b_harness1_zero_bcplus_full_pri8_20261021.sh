#!/usr/bin/env bash
# Submit TRIM Harness-1 / Qwen3.5-27B (bare base, no adapter) baseline eval — ZERO|ALL × 4bench × 3seed.
# 训练前 base model 完整基线：1 model × {zero, all} × {bcplus_full,longsealqa,frames,hotpotqa} × 3 seed = 24 任务。
# priority=5（用户要求），queue 2334，gm402-80g 8 卡单节点，MemQA-General-V5。
#
# 不复用 9 月旧结果（zero fcanl 16-worker / all v2_64k 64-worker 口径不一致且只 1 trial）。
# zero/all 都用 64 worker 统一口径（entry 内 zero TP=64 / all ALL_EVAL_TP=64），使 3-seed 均值可靠、
# zero vs all 可直接对比。3 seed = 3 次独立重跑（不同 RUN_ID，vLLM temp=1.0 采样无显式 RNG seed，
# 每次重跑即独立样本，同 vqqxp step200 3trial 模式，复现性好 std<1.6pp）。
#
# 用法: submit_qwen35_27b_harness1_baseline_eval_8gpu_20261015.sh <EVAL_MODE> <BENCHMARK> <SEED>
#   EVAL_MODE  ∈ {zero, all}         # zero=瘦harness, all=完整harness(带verifier)
#   BENCHMARK  ∈ {bcplus_full, longsealqa, frames, hotpotqa}
#   SEED       ∈ {1, 2, 3}            # 3 次独立重跑的编号，映射到 EVAL_SEED=42/123/2024
# 每次 EVAL_SEED 不同（42/123/2024）经 lib run_eval_cell 的 --seed 钩子驱动 run_eval.py 采样，
# 使 3 次重跑产生真正独立的 rollout 轨迹（非 byte-identical）。每次独立 RUN_ID 避开脏目录坑。
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "用法: $0 <EVAL_MODE> <BENCHMARK> <SEED>" >&2
  echo "  EVAL_MODE ∈ {zero, all}; BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}; SEED ∈ {1,2,3}" >&2
  exit 1
fi
EVAL_MODE="$1"; BENCHMARK="$2"; SEED="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y%m%d 2>/dev/null || echo 20261021)

case "$EVAL_MODE" in zero|all) ;; *) echo "FATAL: EVAL_MODE=$EVAL_MODE (want zero|all)" >&2; exit 1 ;; esac
case "$BENCHMARK" in bcplus_full|longsealqa|frames|hotpotqa) ;; *) echo "FATAL: BENCHMARK=$BENCHMARK" >&2; exit 1 ;; esac
# SEED 编号 1/2/3 映射到实际采样 seed 值（与 h3eb8 3seed 一致：42/123/2024）。
case "$SEED" in
  1) EVAL_SEED=42 ;;
  2) EVAL_SEED=123 ;;
  3) EVAL_SEED=2024 ;;
  *) echo "FATAL: SEED=$SEED (want 1|2|3)" >&2; exit 1 ;;
esac

ENTRY="${SCRIPT_DIR}/entry_qwen35_27b_harness_zero_all_bcplus_transfer_eval_8gpu.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness1-qwen35-27b-harness-${EVAL_MODE}-${BENCHMARK}-eval-8gpu-s${SEED}-${DATE}"
RUN_ID="qwen35_27b_harness_${EVAL_MODE}_${BENCHMARK}_8gpu_s${SEED}_${DATE}_memqa_v5"
DESC="Harness-1 Qwen3.5-27B base (no adapter,no-think) ${EVAL_MODE} ${BENCHMARK} seed${SEED}(EVAL_SEED=${EVAL_SEED}). 8gpu 64-worker. 1 of 3 seeds (3-seed mean). base baseline for vqqxp/iowty training-curve comparison."

# Description 必须 ≤500 字符（CloudML 拒绝超长）。
DESC_LEN=$(printf '%s' "${DESC}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description is ${DESC_LEN} chars (>500). Trim it." >&2; exit 1
fi
echo "[submit] EVAL_MODE=${EVAL_MODE} BENCHMARK=${BENCHMARK} SEED=${SEED} JOB_NAME=${JOB_NAME} RUN_ID=${RUN_ID} (desc=${DESC_LEN} chars)"

cml custom_train submit \
  --job_name "$JOB_NAME" --description "$DESC" \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* \
  --env CHROMA_API_KEY=local-offline --env CHROMA_DATABASE=local-offline --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env EVAL_MODE="$EVAL_MODE" --env BENCHMARK="$BENCHMARK" --env EVAL_SEED="$EVAL_SEED" --env RUN_ID="$RUN_ID" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
