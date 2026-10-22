#!/usr/bin/env bash
# Submit gpt-oss-20b "zero" eval (8x gm402-80g) to CloudML.
# Prepared 2026-09-11; NOT YET SUBMITTED — review YAML first, then run this script.
#
# JuiceFS mount is passed via direct CLI --juicefs_mount_configs (NOT YAML --filename path),
# so endpoint=bj9-cloudml is explicit in the API request. endpoint is currently ignored by
# the Go layer (unknown field); the effective field is juiceFsCluster=bj9-cloudml. Both are
# set to bj9-cloudml for belt-and-suspenders safety (see cloudml-trim-train-job-config memory).
#
# Queue 2334 (记忆对话, kscn-hd-cloudml-prod / 青海), gm402-80g, GUARANTEED_PUBLIC,
# priority 8, tag MemQA-General-V5, image micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto.
set -euo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/cloudml_eval_bcplus_full_gpt_oss_20b_zero_8gpu_20260911_not_submitted.yaml"
JUICEFS_JSON='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

cml custom_train submit \
  --filename "${YAML}" \
  --juicefs_mount_configs "${JUICEFS_JSON}" \
  --tag MemQA-General-V5
