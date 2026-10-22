#!/usr/bin/env bash
# Submit TRIM Harness-1 BC+ full Qwen3-4B "zero" eval (8x gm402-80g) to CloudML.
# NOT YET EXECUTED. Run this script to submit (consumes real resources).
#
# JuiceFS mount is passed via direct CLI --juicefs_mount_configs JSON (NOT via the
# YAML --filename path), so endpoint=bj9-cloudml is explicit in the API request.
# Per cloudml-trim-train-job-config experience: the Go backend ignores the `endpoint`
# field and uses juiceFsCluster; double-writing both to bj9-cloudml is safest.
set -euo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/cloudml_eval_bcplus_full_qwen3_4b_zero_8gpu_20260911_not_submitted.yaml"

# JuiceFS: volume=songzijun, cluster=bj9-cloudml, mount at /mnt/songzijun, read-write.
JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

cml custom_train submit \
  --filename "${YAML}" \
  --juicefs_mount_configs "${JUICEFS_JSON}" \
  --tag "MemQA-General-V5"
