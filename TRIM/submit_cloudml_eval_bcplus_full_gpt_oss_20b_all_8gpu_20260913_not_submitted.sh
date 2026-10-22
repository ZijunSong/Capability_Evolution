#!/usr/bin/env bash
# Submit script for the GPT-OSS-20B "all" 8-GPU bcplus_full eval.
# 2026-09-13 fresh attempt. NOT submitted by this script on its own — run it to submit,
# or copy the cml command at the bottom.
#
# Entrypoint: scripts/run_bcplus_w_harness1_gptoss_all_gpu8_eval.sh (single-purpose
# gpt-oss-20b "all" script; needs no MODELS/SKIP_ZERO env).
#
# JuiceFS is passed via the direct CLI --juicefs_mount_configs argument (NOT via the
# YAML --filename path) so endpoint=bj9-cloudml is explicit in the API request.
# JSON fields: volume, juiceFsCluster, subPath, mountPath, ReadOnly (+ endpoint double-
# written to bj9-cloudml for safety, though the Go client keys off juiceFsCluster).
#
# Submit to: queue 2334 (记忆对话, kscn-hd-cloudml-prod / 青海), gm402-80g, 8 GPU.
# Priority 8, tag MemQA-General-V5. JuiceFS volume songzijun mounted at /mnt/songzijun.
set -euo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/cloudml_eval_bcplus_full_gpt_oss_20b_all_8gpu_20260913_not_submitted.yaml"

JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","ReadOnly":false}]'

set -x
cml custom_train submit \
  --filename "${YAML}" \
  --juicefs_mount_configs "${JUICEFS_JSON}" \
  --tag MemQA-General-V5
