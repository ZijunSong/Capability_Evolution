#!/usr/bin/env bash
# === GPT-OSS-20B trim/all + recall-only + lambda_opd=0.1 RETRY r2 (2026-10-18, priority 5) ===
# Resubmit of the 0.1 group after qvnwn FAILED at 2026-10-17 23:58.
#
# qvnwn failure root cause (verified from history_logs):
#   master (NODE_RANK=0, slave95) pip install took ~42min due to pip
#   dependency-resolver backtracking on latex2sympy2/pyserini version combos;
#   workers (slave100 etc.) finished pip+preflight+cp model in ~17min and
#   reached run_train_multinode's internal torchrun rendezvous first. Master
#   was still in pip/cp-model when workers tried to connect to its TCPStore
#   port 20672 (not open yet); workers hit "client socket timed out after
#   900000ms" x2 (30min) and crashed: torch.distributed.DistNetworkError.
#   --lambda-opd 0.1 was correctly passed (visible in worker torchrun cmdline
#   at 23:28:22); the job died BEFORE step0 — pure rendezvous timing race,
#   NOT a lambda_opd/training issue.
#
# FIX: entry now has a worker-wait-master barrier — workers poll a shared
# OUT-dir signal file that ONLY master writes, right before exec (after
# pip+preflight+cp model done). Workers never start torchrun before master,
# so the 900s TCPStore timeout can no longer fire. Entry also bumps rdzv-id
# to ...-lop01-32-r2 to avoid colliding with qvnwn's stale rendezvous endpoint.
#
# This submit uses a FRESH OUT dir (_20261018_r2, not qvnwn's _20261017) per
# CLAUDE.md "each recovery uses a new job name + explicit --out independent
# dir"; qvnwn's failed attempt (no step0, no STEP_COMPLETE) is preserved as-is.
#
# Everything else byte-identical to the 0p1 submit: same ENTRY (entry_*_0p1_
# *_20261017_bf16.sh, which now carries the barrier + --lambda-opd 0.1 +
# rdzv-id ...-lop01-32-r2), priority 5, 4node32gpu, recall-only reward,
# tensorboard ON, mb1, bf16-dequant, NO SCAPE_FORCE_LOCAL_HARMONY, gpu-mem0.90.
#
# Usage:
#   bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261018_r2_pri5.sh
# Dry-run:
#   DRY=1 bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261018_r2_pri5.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261017_bf16.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu-4node-20261018-r2"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu_20261018_r2"
# Description MUST be <=500 chars (CloudML rejects >500).
DESCRIPTION="TRIM BC+ trim GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora 200step. REWARD=final-answer-recall ONLY. TENSORBOARD ON. lambda_opd=0.1(launch.py --lambda-opd override,trim默认0.01). RETRY r2 of qvnwn(FAILED master pip慢致rendezvous超时,与lambda_opd无关). entry加worker-wait-master barrier防rendezvous竞争. mb1 gpu-mem0.90 max-turns40 max-model-len8192. bf16-dequant避mxfp4. NO SCAPE_FORCE_LOCAL_HARMONY. curate-clipping. world_size=32 audit."

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
