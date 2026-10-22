#!/usr/bin/env bash
# === GPT-OSS-20B trim/all + recall-only + lambda_opd=0.1 RESUME r5 (2026-10-21, priority 8) ===
# Continue 0.1 group after qv0qw (r4) was stopped by user at step27 (step27 backward
# SUCCEEDED — the non-finite-CISPO flake from obkzm was avoided by the re-run, confirmed
# by STEP_MANIFEST step_000027 n_optimizer_steps=1). latest->step_000027 (STEP_COMPLETE+
# optimizer.pt+STEP_MANIFEST+rng+sampler+adapter all intact). resume from step28.
# OUT REUSED. r5 uses a fresh rdzv-id (...-r5-resume) distinct from r4 to keep the
# barrier signal uniquely named (barrier fix from r4 carried over).
# priority 8 per user request. --lambda-opd 0.1 + --resume + barrier(r5 rdzv-id).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261021_resume_r5.sh"
test -f "${ENTRY}"
JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu-4node-20261021-resume-r5-pri8"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu_20261018_r2"
DESCRIPTION="TRIM BC+ trim GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora 200step. REWARD=final-answer-recall ONLY. TENSORBOARD ON. lambda_opd=0.1. RESUME r5:qv0qw r4 stop@step27(已成功避开non-finite CISPO偶发flake)→从step28续跑. step_000027 ckpt完整. OUT复用. barrier修复(信号名带rdzv-id r5-resume). priority8. mb1 gpu-mem0.90. curate-clipping. world_size=32 audit."
DESC_LEN=$(printf '%s' "${DESCRIPTION}" | wc -c)
if [ "${DESC_LEN}" -gt 500 ]; then echo "FATAL: description is ${DESC_LEN} chars (>500)." >&2; exit 1; fi
echo "DESCRIPTION_LEN ${DESC_LEN} (<=500 ok)"
CMD=(
  cml custom_train submit --job_name "${JOB_NAME}" --description "${DESCRIPTION}"
  --access_type PRIVATE --tag MemQA-General-V5
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false --enable_tensorboard --rdma
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'
  --resource_configs '[{"nodeRole":"worker","nodeNumber":4,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]'
  --env PYTHONUNBUFFERED=1 --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --env TRIM_GPU_KEEPALIVE=0
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
  --env OPENAI_API_KEY=**-****-***********-***-*-****-*** --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
  --env OUT="${OUT}"
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}'
  --diag_options '[{"name":"HangDetection","enable":false}]'
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}'
  --image_command "$(cat "${ENTRY}")"
)
if [ "${DRY:-0}" = "1" ]; then echo "[dry-run] would execute:"; printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
