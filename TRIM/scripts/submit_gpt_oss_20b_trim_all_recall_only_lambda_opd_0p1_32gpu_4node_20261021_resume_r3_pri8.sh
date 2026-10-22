#!/usr/bin/env bash
# === GPT-OSS-20B trim/all + recall-only + lambda_opd=0.1 RESUME r3 (2026-10-21, priority 8) ===
# Resume after obkzm (r2 resume pri8) FAILED at step27 backward on 2026-10-20 23:36
# with "non-finite logprobs in CISPO" (fsdp2_actor.py:599 — GPT-OSS bf16 numeric
# flake, NOT a lambda_opd issue; crashed in rl logprob forward, not OPD loss).
# step_000026 STEP_COMPLETE+optimizer.pt+STEP_MANIFEST+rng+sampler+adapter all
# intact (verified). latest->step_000026, resume from step27.
#
# This re-run is the bet that the non-finite logprob was an incidental per-chunk
# flake: sampler/rng state restored (same query sharding resumes) but vLLM
# generation under temperature may diverge enough to avoid the same NaN. If it
# crashes again at the same point, fall back to skipping non-finite chunks
# (code change at fsdp2_actor.py:598-602: raise -> continue).
#
# OUT REUSED (not new) — resume MUST reuse the original OUT to find latest ckpt.
# Entry (r3) carries: barrier + resume preflight (assert latest step_000026 +
# STEP_COMPLETE+optimizer.pt+STEP_MANIFEST) + rdzv-id ...-r3-resume (avoid
# collision with obkzm's ...-r2-resume rendezvous) + --resume + --lambda-opd 0.1.
#
# priority 8 per user request. Everything else byte-identical to the r2 resume.
#
# Usage:
#   bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261021_resume_r3_pri8.sh
# Dry-run:
#   DRY=1 bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261021_resume_r3_pri8.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261021_resume_r3.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu-4node-20261021-resume-r3-pri8"
# OUT REUSED from yueep r2 (resume needs the latest ckpt in this OUT)
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu_20261018_r2"
DESCRIPTION="TRIM BC+ trim GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora 200step. REWARD=final-answer-recall ONLY. TENSORBOARD ON. lambda_opd=0.1. RESUME r3:obkzm FAILED@step27(non-finite logprobs in CISPO,GPT-OSS bf16数值flake非lambda_opd)→从step27续跑. step_000026 ckpt完整. OUT复用. entry加barrier+resume preflight+rdzv-id r3-resume+--resume. priority8. mb1 gpu-mem0.90. curate-clipping. world_size=32 audit."

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
  --priority 8
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
