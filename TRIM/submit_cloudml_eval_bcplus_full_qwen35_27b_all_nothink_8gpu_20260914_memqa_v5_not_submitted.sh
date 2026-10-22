#!/usr/bin/env bash
# Submit script for the Qwen3.5-27B "all" (no-thinking) 8-GPU bcplus_full eval.
# 2026-09-14. NOT submitted by this script on its own — run it to submit,
# or copy the cml command at the bottom.
#
# Entrypoint: scripts/run_qwen35_27b_all_nothink_eval_cloudml_8gpu.sh, which
# installs the locked four-package runtime, runs JVM/tiktoken/import/ARCH
# preflight, then execs scripts/run_bcplus_w_harness1_qwen35_27b_all_gpu8_eval_nothinking.sh
# (6 Qwen3.5-27B actors TP=1 on GPU 0-5 + harness-1 verifier on GPU 6,
#  run_eval.py --harness Harness-1 --evaluation-path upstream_api --component all --tp 64,
#  local_bm25 no Chroma). Thinking is off via vLLM serve
#  --chat-template-kwargs {"enable_thinking":false} injected by the wrapper.
#
# JuiceFS is passed via the direct CLI --juicefs_mount_configs argument (NOT via the
# YAML --filename path) so endpoint=bj9-cloudml is explicit in the API request.
# JSON fields: volume, juiceFsCluster, subPath, mountPath, ReadOnly (+ endpoint double-
# written to bj9-cloudml for safety, though the Go client keys off juiceFsCluster).
#
# Submit to: queue 2334 (记忆对话, kscn-hd-cloudml-prod / 青海), gm402-80g, 8 GPU.
# Priority 8, tag MemQA-General-V5. JuiceFS volume songzijun mounted at /mnt/songzijun.
set -euo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/cloudml_custom_train_harness1_bcplus_full_qwen35_27b_all_nothink_8gpu_20260914_memqa_v5_not_submitted.yaml"

JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

set -x
cml custom_train submit \
  --filename "${YAML}" \
  --juicefs_mount_configs "${JUICEFS_JSON}" \
  --tag MemQA-General-V5
