#!/usr/bin/env bash
# Submit TRIM OPD-teacher 联盟消融：Qwen3.5-9B 2node16gpu 50step，priority 5（用户要求）。
# student=zero（--component all 派生），teacher=四联盟之一（env TRIM_TEACHER_COMPONENTS
# 精确 mask）。REWARD=final-answer-recall（显式 --rl-reward）。排队正常。
#
# 四组（TEACHER_KEY）：
#   esp   Evidence Selection & Prioritization  = subtractive_curation,importance_tagging,adaptive_rerank_instruction
#   ccso  Context Consolidation & State Org.   = sentence_compress,content_dedup,evidence_graph
#   rieve Retrieval Initiation & Expansion     = auto_populate_first_search,chunk_neighbors
#   msc   Metacognitive Search Control         = verify_tool,token_budget_marker
#
# 用法: submit_qwen35_9b_trim_teacher_ablation_16gpu_2node_20261027.sh <TEACHER_KEY>
#   TEACHER_KEY ∈ {esp, ccso, rieve, msc}
# DRY=1 干跑（只 echo cml 命令不提交）。
# 与 all-trim（entry_qwen35_27b_trim_all_recall_only_32gpu_4node_20261022.sh）唯一差异：
# teacher 精确联盟 + 50step；mb2/gpu-mem0.90/max-turns40/max-model-len8192 全同。
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "用法: $0 <TEACHER_KEY>   # esp|ccso|trieve|msc" >&2
  exit 1
fi
TEACHER_KEY="$1"
# 归一化检索联盟 key 拼写（rieve/trieve 同义；规范形=rieve）
if [ "${TEACHER_KEY}" = "trieve" ]; then TEACHER_KEY="rieve"; fi
case "${TEACHER_KEY}" in esp|ccso|rieve|msc) ;; *) echo "FATAL: TEACHER_KEY=${TEACHER_KEY}" >&2; exit 1 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen35_9b_trim_teacher_ablation_recall_16gpu_2node_20261027.sh"
test -f "${ENTRY}"
ENTRY_BYTES=$(wc -c < "${ENTRY}")
[ "${ENTRY_BYTES}" -le 22000 ] || { echo "FATAL: entry ${ENTRY_BYTES}B >22000（imageCommand SQL 铁律）" >&2; exit 1; }

JOB_NAME="trim-harness1-qwen35-9b-trim-teacher-${TEACHER_KEY}-50step-16gpu-2node-20261027"
OUT="/mnt/songzijun/outputs/qwen35-9b-trim-teacher-${TEACHER_KEY}-50step-16gpu_20261027"
PRIORITY="${PRIORITY:-5}"
DESCRIPTION="TRIM OPD-teacher ablation ${TEACHER_KEY}: Qwen3.5-9B 2node16gpu torch_ddp_lora 50step. student=zero; teacher=${TEACHER_KEY} exact coalition (TRIM_TEACHER_COMPONENTS env-gated patch, TEACHER_MASK_OK asserted). REWARD=final-answer-recall explicit. mb4 gpu-mem0.90 max-turns40 max-model-len8192. curate-clipping+4 DDP patches. world_size=16 audit."

DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description is ${DESC_LEN} chars (>500)." >&2
  exit 1
fi
echo "[submit] TEACHER_KEY=${TEACHER_KEY} JOB_NAME=${JOB_NAME} OUT=${OUT} (desc=${DESC_LEN} chars, entry=${ENTRY_BYTES}B, priority=${PRIORITY})"

CMD=(
  cml custom_train submit
  --job_name "${JOB_NAME}"
  --description "${DESCRIPTION}"
  --access_type PRIVATE
  --tag MemQA-General-V5
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
  --framework pytorch
  --queue_id 2334
  --priority "${PRIORITY}"
  --preemptible=false
  --enable_tensorboard
  --rdma
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'
  --resource_configs '[{"nodeRole":"worker","nodeNumber":2,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]'
  --env PYTHONUNBUFFERED=1
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  --env TRIM_GPU_KEEPALIVE=0
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
  --env OPENAI_API_KEY=**-****-***********-***-*-****-***
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
  --env SCAPE_FORCE_LOCAL_HARMONY=1
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
  --env OUT="${OUT}"
  --env TEACHER_KEY="${TEACHER_KEY}"
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}'
  --diag_options '[{"name":"HangDetection","enable":false}]'
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}'
  --image_command "$(cat "${ENTRY}")"
)

if [ "${DRY:-0}" = "1" ]; then
  echo "[dry-run] would execute:"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

exec "${CMD[@]}"
