#!/usr/bin/env bash
# === GPT-OSS-20B trim/all + recall-only + lambda_opd=0.1 RESUME (2026-10-19, priority 5) ===
# Resume yueep (0.1 r2) stopped by user 2026-10-19 ~16:35 at step25.
# latest->step_000025 (STEP_COMPLETE+optimizer.pt+adapter all present), resume
# from step26. --resume reads $OUT/checkpoints/scape_seed/latest automatically.
#
# OUT is REUSED (not new) — resume MUST reuse the original OUT to find the
# latest ckpt (CLAUDE.md resume exception, distinct from failed-attempt
# recovery which needs a fresh OUT). Entry carries: barrier (inherited from
# r2) + resume preflight (assert latest+STEP_COMPLETE+optimizer.pt+
# STEP_MANIFEST.json) + rdzv-id ...-r2-resume (avoid collision with the
# stopped yueep rendezvous) + --resume at exec end.
#
# priority 5 per user request. Everything else byte-identical to the r2 submit.
#
# Usage:
#   bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261019_resume_pri5.sh
# Dry-run:
#   DRY=1 bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261019_resume_pri5.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261019_resume.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu-4node-20261019-resume"
# OUT REUSED from yueep r2 (resume needs the latest ckpt in this OUT)
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu_20261018_r2"
DESCRIPTION="TRIM BC+ trim GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora 200step. REWARD=final-answer-recall ONLY. TENSORBOARD ON. lambda_opd=0.1. RESUME yueep kill@step25(latest step_000025 ckpt完整)→从step26续跑. OUT复用原r2目录读latest. entry加barrier+resume preflight+rdzv-id r2-resume+--resume. mb1 gpu-mem0.90 max-turns40 max-model-len8192. bf16-dequant避mxfp4. NO SCAPE_FORCE_LOCAL_HARMONY. curate-clipping. world_size=32 audit."

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
