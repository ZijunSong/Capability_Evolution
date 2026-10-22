#!/usr/bin/env bash
# Submit the Qwen3-4B TRIM recovery4 training job. NOT YET EXECUTED.
# JuiceFS mount is passed directly via CLI --juicefs_mount_configs (NOT via the YAML
# --filename path) so endpoint=bj9-cloudml is explicit in the API request, per
# CLAUDE.md / cloudml-trim-train-job-config memory. Both endpoint and juiceFsCluster
# are set to bj9-cloudml (endpoint is ignored by the Go layer as an unknown field,
# juiceFsCluster is the one actually consumed; double-writing both is the safe form).
#
# Workspace 10063, context songzijun-cto. Queue 2334 (记忆对话, gm402-80g,
# cluster kscn-hd-cloudml-prod/青海), priority 8, tag MemQA-General-V5.
set -euxo pipefail

source /mnt/songzijun/mify_api.env   # sets CML API creds / GITHUB token etc.
export CML_WORKSPACE_ID=10063
export CML_CONTEXT=songzijun-cto

cml custom_train submit \
  --filename /mnt/songzijun/Capability_Evolution/TRIM/cloudml_custom_train_harness1_bcplus_full_qwen3_4b_trim_8gpu_20260911_memqa_general_v5_recovery4_not_submitted.yaml \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --tag MemQA-General-V5
