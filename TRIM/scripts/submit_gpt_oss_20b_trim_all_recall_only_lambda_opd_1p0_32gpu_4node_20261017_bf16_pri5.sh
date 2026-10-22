#!/usr/bin/env bash
# === GPT-OSS-20B trim/all + recall-only + lambda_opd=1.0 (2026-10-17, priority 5) ===
# Fork of submit_gpt_oss_20b_trim_all_recall_only_32gpu_4node_20261002_bf16_pri2.sh
# (pncda). ONLY CHANGE vs pncda: lambda_opd 0.01 -> 1.0 (entry passes explicit
# --lambda-opd 1.0). Everything else byte-identical: same trim/all 200step 32gpu
# + recall-only reward + tensorboard ON + mb1 + bf16-dequant + NO
# SCAPE_FORCE_LOCAL_HARMONY + gpu-mem0.90 + 4 DDP patches.
#
# Differences vs pncda submit:
#   - ENTRY  -> entry_gpt_oss_20b_trim_all_recall_only_lambda_opd_1p0_32gpu_4node_20261017_bf16.sh
#   - JOB_NAME / OUT get distinct lambda_opd_1p0 / _lop10 suffix (do NOT collide
#     with pncda's _20261002 OUT — fresh dir per CLAUDE.md).
#   - --priority 2 -> --priority 5 (user request: 2 groups at priority 5).
#
# lambda_opd mechanism (verified, NOT a guess):
#   launch.py:760-768  --lambda-opd  type=float  default=None  dest=lambda_opd
#   launch.py:1307-1308  if args.lambda_opd is None:
#                          args.lambda_opd = SCAPE_RL_LAMBDA_OPD if seed_scale else 0.1
#   rl_opd_types.py:59  SCAPE_RL_LAMBDA_OPD = 0.01   # trim (seed_scale) default 0.01
#   pncda (no flag, --train_method trim) -> 0.01; THIS job -> 1.0. No code change.
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. image_command
# read verbatim from the entry script via "$(cat ...)".
#
# Usage:
#   bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_1p0_32gpu_4node_20261017_bf16_pri5.sh
# Dry-run (echo cml command, no submit):
#   DRY=1 bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_1p0_32gpu_4node_20261017_bf16_pri5.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_all_recall_only_lambda_opd_1p0_32gpu_4node_20261017_bf16.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-1p0-32gpu-4node-20261017"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-1p0-32gpu_20261017"
# Description MUST be <=500 chars (CloudML rejects >500).
DESCRIPTION="TRIM BC+ trim GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora 200step. REWARD=final-answer-recall ONLY. TENSORBOARD ON. lambda_opd=1.0(vs pncda default 0.01,launch.py --lambda-opd override,no code change). mb1(gpt-oss MoE no grad-ckpt,k6v0p) gpu-mem0.90 max-turns40 max-model-len8192. bf16-dequant避mxfp4 Marlin PTX坑. NO SCAPE_FORCE_LOCAL_HARMONY(gpt-oss to_json崩). curate-clipping patch. 4 DDP patches. world_size=32 audit."

# Defensive: refuse to submit if description exceeds 500 chars.
DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description is ${DESC_LEN} chars (>500). CloudML will reject it. Trim it." >&2
  exit 1
fi
echo "DESCRIPTION_LEN ${DESC_LEN} (<=500 ok)"

CMD=(
  cml custom_train submit
  --job_name "${JOB_NAME}"
  --description "${DESCRIPTION}"
  --access_type PRIVATE
  --tag MemQA-General-V5
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
  --framework pytorch
  --queue_id 2334
  --priority 5
  --preemptible=false
  --enable_tensorboard
  --rdma
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'
  --resource_configs '[{"nodeRole":"worker","nodeNumber":4,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]'
  --env PYTHONUNBUFFERED=1
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  --env TRIM_GPU_KEEPALIVE=0
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
  --env OPENAI_API_KEY=**-****-***********-***-*-****-***
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
  --env OUT="${OUT}"
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
