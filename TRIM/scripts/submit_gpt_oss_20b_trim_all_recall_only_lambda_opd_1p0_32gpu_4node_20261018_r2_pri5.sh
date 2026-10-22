#!/usr/bin/env bash
# === GPT-OSS-20B trim/all + recall-only + lambda_opd=1.0 RETRY r2 (2026-10-18, priority 5) ===
# Resubmit of the 1.0 group. njxzi (the original 1.0 job, 2026-10-17 08:06)
# was still enqueued after ~24h (priority 5, gm402 available stuck at ~19-30,
# never enough for a 32-GPU gang) — never started a pod. Kill njxzi and
# resubmit with the same entry (now carrying the worker-wait-master barrier
# + rdzv-id ...-lop10-32-r2) and a fresh OUT dir.
#
# Same barrier rationale as the 0p1 r2 submit (see there): qvnwn (0.1 group)
# FAILED because master pip (~42min, backtracking) was slower than workers
# (~17min) and workers hit TCPStore 900s timeout before master started
# torchrun. The barrier makes workers wait for a master-ready signal file in
# the shared OUT dir before launching run_train_multinode. lambda_opd was
# never the issue.
#
# FRESH OUT dir (_20261018_r2). njxzi never started a pod so its OUT had no
# step0/STEP_COMPLETE; this r2 dir is clean anyway.
#
# Everything else byte-identical: same ENTRY (entry_*_1p0_*_20261017_bf16.sh,
# now with barrier + --lambda-opd 1.0 + rdzv-id ...-lop10-32-r2), priority 5,
# 4node32gpu, recall-only reward, tensorboard ON, mb1, bf16-dequant, NO
# SCAPE_FORCE_LOCAL_HARMONY, gpu-mem0.90.
#
# Usage:
#   bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_1p0_32gpu_4node_20261018_r2_pri5.sh
# Dry-run:
#   DRY=1 bash submit_gpt_oss_20b_trim_all_recall_only_lambda_opd_1p0_32gpu_4node_20261018_r2_pri5.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_all_recall_only_lambda_opd_1p0_32gpu_4node_20261017_bf16.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-1p0-32gpu-4node-20261018-r2"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-1p0-32gpu_20261018_r2"
# Description MUST be <=500 chars (CloudML rejects >500).
DESCRIPTION="TRIM BC+ trim GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora 200step. REWARD=final-answer-recall ONLY. TENSORBOARD ON. lambda_opd=1.0(launch.py --lambda-opd override,trim默认0.01). RETRY r2 of njxzi(enqueued 24h未起,重提). entry加worker-wait-master barrier防rendezvous竞争. mb1 gpu-mem0.90 max-turns40 max-model-len8192. bf16-dequant避mxfp4. NO SCAPE_FORCE_LOCAL_HARMONY. curate-clipping. world_size=32 audit."

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
