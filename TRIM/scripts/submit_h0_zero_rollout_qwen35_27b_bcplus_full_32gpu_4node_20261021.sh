#!/usr/bin/env bash
# Submit H0 zero rollout (fixed-base, NO training, NO adapter) on bcplus_full (830q)
# for Qwen3.5-27B (Qwen3_5ForConditionalGeneration dense bf16 27B, ~52GiB/card, no-think)
# via run_h0_zero_rollout.py on FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards = 32 GPUs).
#
# Qwen3.5-27B IS the verified reference for vLLM 0.19.1 + Qwen3_5 support (9 月
# fcanl/iytmg + bg3xs smoke). legacy_local eval runtime, component=zero (0/10 mask,
# student-only no harness search). 32 vLLM engines (8/node TP=1) score round-robin shards
# w/ BC+ local_bm25. node 0 merges → ZERO_ROLLOUT.json evidence_recall. priority=8.
#
# Pure CLI flags; JuiceFS via --juicefs_mount_configs (juiceFsCluster=bj9-cloudml).
# image_command read verbatim via "$(cat ...)".
#
# Usage:
#   bash /mnt/songzijun/Capability_Evolution/TRIM/scripts/submit_h0_zero_rollout_qwen35_27b_bcplus_full_32gpu_4node_20261021.sh
# Dry-run: DRY=1 bash ...sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_h0_zero_rollout_qwen35_27b_bcplus_full_32gpu_4node_20261021.sh"
test -f "${ENTRY}"

JOB_NAME="trim-h0-zero-rollout-bcplus-full-qwen35-27b-32gpu-4node-20261021-cp2"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/h0_zero_bcplus_full_qwen35_27b_32gpu_4node_20261021_cp2"
DESCRIPTION="H0 zero rollout (fixed-base, NO training, NO adapter) bcplus_full 830q Qwen3.5-27B (Qwen3_5ForConditionalGeneration dense bf16 27B, ~52GiB/card, no-think). 4node32gpu. component=zero (0/10 mask, no harness search) legacy_local graph-free eval. 32 vLLM engines (8/node TP=1) score round-robin shards w/ BC+ local_bm25. vLLM 0.19.1 supports Qwen3_5 (27B verified). node0 merges→ZERO_ROLLOUT.json. run_h0_zero_rollout.py (no torchrun, file-based 4x8). priority8. H0 baseline vs Qwen3-4B 6.26%."

DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then
  echo "FATAL: description is ${DESC_LEN} chars (>500). Trim it." >&2
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
