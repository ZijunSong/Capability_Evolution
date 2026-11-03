#!/usr/bin/env bash
# Submit TRIM Harness-1 / Qwen3.5-9B (bare base, no adapter) COMPONENT ablation eval —
# 4 component 联盟 × 4bench × 3seed = 48 任务。priority=5（用户要求），queue 2334，
# gm402-80g 8 卡单节点，MemQA-General-V5。没卡就排队。
#
# 组件联盟（canonical 小写逗号列表，entry 内 COMPONENT_KEY 映射；用户 + 分隔形式等价）：
#   esp   Evidence Selection & Prioritization        = subtractive_curation,importance_tagging,adaptive_rerank_instruction
#   ccso  Context Consolidation & State Organization = sentence_compress,content_dedup,evidence_graph
#   rieve Retrieval Initiation & Expansion           = auto_populate_first_search,chunk_neighbors
#   msc   Metacognitive Search Control               = verify_tool,token_budget_marker
#
# 口径与 2026-10-15 27B base zero/all 24 任务基线完全一致：64 eval worker、GPU_UTIL=0.90、
# MAX_MODEL_LEN=65536、MAX_NUM_SEQS=6、no-think、temp=1.0、--seed 42/123/2024（EVAL_SEED 钩子），
# 使 component 3-seed 均值可与基线 3-seed 均值（zero 36.64/76.33/89.94/95.50% 等）直接对比。
# msc 含 verify_tool → 7 actor + harness-1 verifier 布局（entry/wrapper 自动分支）。
#
# 用法: submit_qwen35_9b_harness_component_eval_8gpu_20261026.sh <COMPONENT_KEY> <BENCHMARK> <SEED>
#   COMPONENT_KEY ∈ {esp, ccso, rieve, msc}
#   BENCHMARK     ∈ {bcplus_full, longsealqa, frames, hotpotqa}
#   SEED          ∈ {1, 2, 3}   # 映射 EVAL_SEED=42/123/2024（与 27B base 3seed 基线一致）
# 每次独立 RUN_ID 避开脏目录坑（assert_fresh_eval_dir 拒脏目录）。
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "用法: $0 <COMPONENT_KEY> <BENCHMARK> <SEED>" >&2
  echo "  COMPONENT_KEY ∈ {esp,ccso,rieve,msc}; BENCHMARK ∈ {bcplus_full,longsealqa,frames,hotpotqa}; SEED ∈ {1,2,3}" >&2
  exit 1
fi
COMPONENT_KEY="$1"; BENCHMARK="$2"; SEED="$3"
if [ "${COMPONENT_KEY}" = "trieve" ]; then COMPONENT_KEY="rieve"; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=${SUBMIT_DATE:-$(date +%Y%m%d 2>/dev/null || echo 20261026)}

case "$COMPONENT_KEY" in esp|ccso|rieve|trieve|msc) ;; *) echo "FATAL: COMPONENT_KEY=$COMPONENT_KEY (want esp|ccso|rieve|msc)" >&2; exit 1 ;; esac
case "$BENCHMARK" in bcplus_full|longsealqa|frames|hotpotqa) ;; *) echo "FATAL: BENCHMARK=$BENCHMARK (want bcplus_full|longsealqa|frames|hotpotqa)" >&2; exit 1 ;; esac
# SEED 编号 1/2/3 映射到实际采样 seed 值（与 27B base 3seed 基线一致：42/123/2024）。
case "$SEED" in
  1) EVAL_SEED=42 ;;
  2) EVAL_SEED=123 ;;
  3) EVAL_SEED=2024 ;;
  *) echo "FATAL: SEED=$SEED (want 1|2|3)" >&2; exit 1 ;;
esac

ENTRY="${SCRIPT_DIR}/entry_qwen35_9b_harness_component_ablation_eval_8gpu.sh"
test -f "${ENTRY}"
# imageCommand 撞 CloudML SQL DataException 铁律：entry <22KB。
ENTRY_BYTES=$(wc -c < "${ENTRY}")
if [ "${ENTRY_BYTES}" -gt 22000 ]; then
  echo "FATAL: entry is ${ENTRY_BYTES} bytes (>22000); compress comments first" >&2; exit 1
fi

JOB_NAME="trim-harness1-qwen35-9b-${COMPONENT_KEY}-${BENCHMARK}-eval-8gpu-s${SEED}-${DATE}"
RUN_ID="qwen35_9b_harness_${COMPONENT_KEY}_${BENCHMARK}_8gpu_s${SEED}_${DATE}_memqa_v5"
DESC="Harness-1 Qwen3.5-9B base (no adapter,no-think) component=${COMPONENT_KEY} ${BENCHMARK} seed${SEED}(EVAL_SEED=${EVAL_SEED}). 8gpu 64-worker, same caliber as 27B/9B verified kit (64-worker). 48-job component ablation (esp/ccso/rieve/msc x 4bench x 3seed). msc=7actor+verifier."

# Description 必须 ≤500 字符（CloudML 拒绝超长）。
DESC_LEN=$(printf '%s' "${DESC}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description is ${DESC_LEN} chars (>500). Trim it." >&2; exit 1
fi
echo "[submit] COMPONENT_KEY=${COMPONENT_KEY} BENCHMARK=${BENCHMARK} SEED=${SEED} JOB_NAME=${JOB_NAME} RUN_ID=${RUN_ID} (desc=${DESC_LEN} chars, entry=${ENTRY_BYTES}B)"

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
  --env COMPONENT_KEY="$COMPONENT_KEY" --env BENCHMARK="$BENCHMARK" --env EVAL_SEED="$EVAL_SEED" --env RUN_ID="$RUN_ID" \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
