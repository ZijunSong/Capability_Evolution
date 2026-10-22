#!/usr/bin/env bash
# Submit Qwen3.5-27B harness self-evolution (run_harness_evolve.py, upstream_api, frozen H0).
# Starts 7 actor serves (9B, tool-call-parser qwen3_coder) + 1 verifier (harness-1),
# then evolves harness banks against the 9B H0 baseline (fa_recall 17.03%, 830q).
# priority=8, single-node 8gpu, queue 2334, MemQA-General-V5.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_harness_evolve_qwen35_27b_bcplus_full_8gpu_20261022.sh"
test -f "${ENTRY}"

JOB_NAME="trim-harness-evolve-qwen35-27b-bcplus-full-8gpu-20261022-v8"
DESC="Qwen3.5-27B harness self-evolution (run_harness_evolve.py). Frozen H0 baseline (upstream_api fa36.74% 830q). 7 actor serves (27B no-think, tool-call-parser qwen3_coder) + 1 verifier (harness-1). upstream_api path (fixes legacy_local 0%). Evolve banks, score component=all, stop before TRIM. pri8 8gpu. node0 merges."
DESC_LEN=$(printf '%s' "${DESC}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then echo "FATAL: desc ${DESC_LEN} chars (>500)" >&2; exit 1; fi
echo "DESCRIPTION_LEN ${DESC_LEN} (<=500 ok)"

cml custom_train submit \
  --job_name "${JOB_NAME}" --description "${DESC}" \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env TRIM_GPU_KEEPALIVE=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env OPENAI_API_KEY=**-*****-******* \
  --env CHROMA_API_KEY=local-offline --env CHROMA_DATABASE=local-offline --env HARNESS1_FORBID_CHROMA=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"
