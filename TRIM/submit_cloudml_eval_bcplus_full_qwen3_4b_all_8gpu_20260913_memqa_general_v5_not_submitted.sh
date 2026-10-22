#!/usr/bin/env bash
# Submit script for the Qwen3-4B-Instruct "all" 8-GPU bcplus_full eval — 20260913.
# Driven by scripts/run_bcplus_w_harness1_qwen_all_gpu8_eval.sh (env-parameterized lib).
# NOT submitted automatically by this script; it prints and runs the cml submit.
#
# JuiceFS is passed via the direct CLI --juicefs_mount_configs argument (NOT via the
# YAML --filename path) so endpoint=bj9-cloudml is explicit in the API request, per
# user requirement. JSON fields: volume, juiceFsCluster, endpoint, subPath, mountPath,
# ReadOnly. The YAML intentionally omits the mount.
#
# Usage:
#   bash submit_cloudml_eval_bcplus_full_qwen3_4b_all_8gpu_20260913_memqa_general_v5_not_submitted.sh
#
# To abort after submit: cml custom_train kill <JOB_ID>
set -euo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/cloudml_eval_bcplus_full_qwen3_4b_all_8gpu_20260913_memqa_general_v5_not_submitted.yaml"

# volume=songzijun, cluster=bj9-cloudml, endpoint=bj9-cloudml (explicit per requirement),
# subPath=/ -> mounts the whole /mnt/songzijun JuiceFS at /mnt/songzijun in the container.
JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

cml custom_train submit \
  --filename "${YAML}" \
  --juicefs_mount_configs "${JUICEFS_JSON}" \
  --tag MemQA-General-V5
