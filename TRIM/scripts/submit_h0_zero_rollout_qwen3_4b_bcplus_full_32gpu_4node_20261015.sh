#!/usr/bin/env bash
# Submit H0 zero rollout (fixed-base measurement, NO training, NO adapter) on bcplus_full
# (830 queries) for Qwen3-4B-Instruct-2507 (dense bf16, bare base) via the
# run_h0_zero_rollout.py entry on FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards = 32 GPUs).
#
# H0 = the fixed-base score BEFORE any harness search or TRIM update. component=zero
# (student-only 0/10 mask, no harness search). legacy_local graph-free eval runtime.
# 32 vLLM engines (8 per node, TP=1) each score a round-robin shard of the 830-pool with
# BC+ local_bm25 retrieval. node 0 merges traces → ZERO_ROLLOUT.json (evidence_recall).
#
# layout: 4x8 matches multi-node training. run_h0_zero_rollout.py does NOT use torchrun/
# Gloo/rdzv — it coordinates purely via the shared JuiceFS --out dir (node 0 polls
# out/nodes/node{0..3}/NODE_DONE.json). priority=5 (eval, below the pri-8 training jobs).
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml (the schema field is
# `juiceFsCluster`, NOT `endpoint`; "bj9-cloudml" IS the cluster/endpoint id).
# image_command read verbatim from the entry script via "$(cat ...)" to avoid CLI
# nested-quote escaping bugs.
#
# === GPU contention — check at submit time ===
# Queue 2334 (记忆对话组-训练队列, kscn-hd-cloudml-prod 青海) cloudml.ng1gm402-8-8.20-224.
# A 32-GPU gang (4 nodes x 8 cards) may queue if the pool is busy. If queue FREE < 32,
# expect FailedQueueScheduling "insufficient resource"; it dequeues once a gang releases.
# Submit is fine; the job waits in queue. Verify with: cml custom_train describe <jobId>.
#
# Usage:
#   bash /mnt/songzijun/Capability_Evolution/TRIM/scripts/submit_h0_zero_rollout_qwen3_4b_bcplus_full_32gpu_4node_20261015.sh
# Dry-run (echo the cml command without submitting): DRY=1 bash ...sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_h0_zero_rollout_qwen3_4b_bcplus_full_32gpu_4node_20261015.sh"
test -f "${ENTRY}"

JOB_NAME="trim-h0-zero-rollout-bcplus-full-qwen3-4b-32gpu-4node-20261015"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/h0_zero_bcplus_full_qwen3_4b_32gpu_4node_20261015"
# Description MUST be <=500 chars (CloudML rejects >500).
DESCRIPTION="H0 zero rollout (fixed-base, NO training, NO adapter) bcplus_full 830q Qwen3-4B-Instruct-2507 dense bf16 bare base. 4node32gpu. component=zero (0/10 mask, no harness search) legacy_local graph-free eval. 32 vLLM engines (8/node TP=1) score round-robin shards w/ BC+ local_bm25. node0 merges→ZERO_ROLLOUT.json evidence_recall. run_h0_zero_rollout.py (no torchrun, file-based 4x8 coord via shared JuiceFS out). priority5 eval below pri8 training. H0=pre-TRIM baseline."

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
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'
  --resource_configs '[{"nodeRole":"worker","nodeNumber":4,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]'
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
