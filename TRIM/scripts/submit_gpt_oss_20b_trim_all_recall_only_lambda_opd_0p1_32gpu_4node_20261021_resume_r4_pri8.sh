#!/usr/bin/env bash
# === GPT-OSS-20B trim/all + recall-only + lambda_opd=0.1 RESUME r4 (2026-10-21, priority 8) ===
# Resume after lhsyc (r3 resume pri8) FAILED at startup on 2026-10-21 01:50 with
# TCPStore 900s timeout — a BARRIER STALE-SIGNAL BUG, NOT a training issue:
# lhsyc reused the same OUT as obkzm; obkzm's leftover .rendezvous_barrier/
# master_ready_0 (same fixed filename) was read by lhsyc's worker BEFORE
# lhsyc's master reached the barrier (master was still cp'ing the 39GB model),
# so the worker launched torchrun early and hit TCPStore timeout. lhsyc never
# reached step27 backward, so the obkzm non-finite-CISPO flake is still untested.
#
# r4 FIX: barrier signal filename now carries the rdzv-id (master_ready_<rdzv-id>),
# so each run's signal is uniquely named — a stale signal from a prior run can
# never be read by this run's workers. r4 uses a fresh rdzv-id (...-r4-resume)
# distinct from r2/r3. Master also clears ITS OWN stale signal right after cd
# (before pip/preflight/cp model), so even a prior r4 attempt's signal won't
# linger while master is still in pip/cp.
#
# resume target: latest->step_000026 (STEP_COMPLETE+optimizer.pt+STEP_MANIFEST+
# rng+sampler+adapter all intact, verified). resume from step27. OUT REUSED.
# priority 8 per user request. --lambda-opd 0.1 + --resume + barrier(r4 fix).
#
# Usage:
#   bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261021_resume_r4_pri8.sh
# Dry-run:
#   DRY=1 bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261021_resume_r4_pri8.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261021_resume_r4.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu-4node-20261021-resume-r4-pri8"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu_20261018_r2"
DESCRIPTION="TRIM BC+ trim GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora 200step. REWARD=final-answer-recall ONLY. TENSORBOARD ON. lambda_opd=0.1. RESUME r4:lhsyc r3 FAILED@启动期barrier stale信号bug(非训练问题,未到step27)→修barrier信号文件名加rdzv-id+master早期清理,从step27续跑. step_000026 ckpt完整. OUT复用. priority8. mb1 gpu-mem0.90. curate-clipping. world_size=32 audit."

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
