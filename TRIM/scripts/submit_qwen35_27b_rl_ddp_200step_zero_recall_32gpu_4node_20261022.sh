#!/usr/bin/env bash
# Submit Harness-1 BC+ full RL (CISPO, H_zero baseline, --train_method rl
# --component zero) Qwen3.5-27B (dense bf16 27B, Qwen3_5ForConditionalGeneration)
# via torch_ddp_lora backend on FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards =
# 32 GPUs, 200 steps) with --rl-reward final-answer-recall (EXPLICIT path,
# score_explicit_reward, pure fa-recall metric, no env/no logprobs).
#
# Re-submit of the rvt6r attempt (t-20261019105015-rvt6r, 2026-10-19): that job
# NEVER RAN (runTime=0 — 5h of FailedQueueScheduling insufficient-resource,
# stopped 15:41 before ever being scheduled; OUT dir never created). Same entry
# (entry_qwen35_27b_rl_ddp_200step_zero_recall_32gpu_4node_20261019.sh, content
# unchanged — it was never executed), fresh jobName + fresh OUT per the
# 全新 job/output prefix rule. Paired with the all-trim 27B job
# (submit_qwen35_27b_trim_all_recall_only_32gpu_4node_20261022.sh) so both
# settings train Qwen3.5-27B with recall-only reward.
#
# Pure CLI flags (no --filename YAML); JuiceFS mount passed explicitly via
# --juicefs_mount_configs JSON with juiceFsCluster=bj9-cloudml. NOTE: the
# CloudML JuiceFS schema field is `juiceFsCluster` (there is no literal
# `endpoint` key) — its value "bj9-cloudml" IS the cluster/endpoint identifier.
# image_command read verbatim from the entry script via "$(cat ...)" to avoid
# CLI nested-quote escaping bugs.
#
# === GPU contention (2026-10-22) — check at submit time ===
# Queue 2334 cloudml.ng1gm402-8-8.20-224 RESOURCEFREE=2 at submit time. A
# 32-GPU gang (4 nodes x 8 cards) WILL queue. Expect FailedQueueScheduling
# "insufficient resource ... left X, request 32"; it dequeues once a gang
# releases. Submit is fine; the job waits in queue. Verify with:
# cml custom_train describe <jobId> (state + events Scheduled/Unschedulable/
# FailedQueueScheduling).
#
# world_size=32 is REQUIRED. Post-submit provenance audit (CLAUDE.md): verify
# world_size is truly 32 by the three-tuple — (1) stdout has torchrun --nnodes 4
# --nproc_per_node 8; (2) checkpoints/<cell>/step_000001/ has 32
# optimizer.rank0000..0031.pt; (3) vllm_sessions/ has 32 _r0.._r31 sessions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_qwen35_27b_rl_ddp_200step_zero_recall_32gpu_4node_20261019.sh"
test -f "${ENTRY}"

JOB_NAME="${JOB_NAME:-trim-harness1-bcplus-full-qwen35-27b-rl-ddp-200step-zero-recall-32gpu-4node-20261022}"
OUT="${OUT:-/mnt/songzijun/outputs/qwen35-27b-rl-ddp-200step-zero-recall-32gpu_20261022}"
PRIORITY="${PRIORITY:-8}"
# Description MUST be <=500 chars (CloudML rejects >500).
DESCRIPTION="TRIM BC+ rl/zero (H_zero RL) Qwen3.5-27B dense bf16 27B 4node32gpu torch_ddp_lora 200step. REWARD=final-answer-recall EXPLICIT (score_explicit_reward, pure metric, no env/logprobs, fed to CISPO backward). mb2 gpu-mem0.90 max-turns40 max-model-len8192 (27B dense grad-ckpt, peak~76GB). SCAPE_FORCE_LOCAL_HARMONY=1. rl/zero no curate projection so 3 DDP patches. world_size=32 audit. Re-submit of rvt6r (never ran, queue-starved)."

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
  --priority "${PRIORITY}"
  --preemptible=false
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
