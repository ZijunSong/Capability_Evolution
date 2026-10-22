#!/usr/bin/env bash
# Submit script for the GPT-OSS-20B "all" 8-GPU bcplus_full eval (native gpu8 wrapper).
# NOT submitted by this script on its own — it just prints and runs the cml submit.
#
# JuiceFS is passed via the direct CLI --juicefs_mount_configs argument (NOT via the
# YAML --filename path) so endpoint=bj9-cloudml is explicit in the API request.
# JSON fields: volume, juiceFsCluster, subPath, mountPath, ReadOnly (+ endpoint double-
# written to bj9-cloudml for safety, though the Go client keys off juiceFsCluster).
set -euo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/cloudml_eval_bcplus_full_gpt_oss_20b_all_8gpu_20260911b_not_submitted.yaml"

JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","ReadOnly":false}]'

set -x
cml custom_train submit \
  --filename "${YAML}" \
  --juicefs_mount_configs "${JUICEFS_JSON}" \
  --tag MemQA-General-V5
