#!/usr/bin/env bash
# === GPT-OSS-20B trim/all + recall-only + lambda_opd=0.1 RESUME r6 (2026-10-21, priority 8) ===
# Resume 0.1 group after ff1vu (r5) FAILED at step28 backward with
# "non-finite logprobs in CISPO" (fsdp2_actor.py:599) — the SAME flake that killed
# obkzm at step27. r4 qv0qw dodged it at step27 (re-run succeeded), but r5 ff1vu hit
# it again at step28 → this is a RECURRING GPT-OSS bf16 numeric flake in this data
# region, not a one-off. "Bet on a flake" is unreliable here.
#
# r6 ROOT FIX (code-tree change): fsdp2_actor.py:598-602 changed the two
#   raise ValueError("non-finite logprobs in CISPO" / "non-finite CISPO ratio")
# to `n_skip_nonfinite += 1; continue` — SKIP the row's CISPO contribution instead
# of aborting the run. Non-finite RL gradients are untrustworthy anyway (NaN); skipping
# lets the step finish (OPD distillation still runs on valid chunks; valid RL chunks
# still contribute). If ALL rows are non-finite, `terms` is empty → numer becomes a
# zero-grad tensor (safe, no crash). A [CISPO_NONFINITE_SKIP] warn is printed so skips
# surface in logs/audit. This is training-side numeric robustness, NOT a scorer change.
#
# resume target: latest->step_000027 (step28 crashed in backward, no step_000028). resume
# from step28. OUT REUSED. r6 fresh rdzv-id (...-r6-resume) keeps barrier signal unique.
# priority 8 per user request. --lambda-opd 0.1 + --resume + barrier(r6 rdzv-id).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_gpt_oss_20b_trim_all_recall_only_lambda_opd_0p1_32gpu_4node_20261021_resume_r6.sh"
test -f "${ENTRY}"
JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu-4node-20261021-resume-r6-pri8"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/gpt-oss-20b-trim-all-recall-only-bf16-lambda-opd-0p1-32gpu_20261018_r2"
DESCRIPTION="TRIM BC+ trim GPT-OSS-20B bf16-dequant 4node 32gpu torch_ddp_lora 200step. REWARD=final-answer-recall ONLY. TENSORBOARD ON. lambda_opd=0.1. RESUME r6:ff1vu r5 FAILED@step28 non-finite CISPO(obkzm step27同类recurring flake非偶发)→改代码fsdp2_actor.py:598-602 raise改continue跳过non-finite chunk不abort. 从step28续跑. step_000027 ckpt完整. OUT复用. barrier r6-resume. priority8. mb1 gpu-mem0.90. curate-clipping. world_size=32 audit."
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
