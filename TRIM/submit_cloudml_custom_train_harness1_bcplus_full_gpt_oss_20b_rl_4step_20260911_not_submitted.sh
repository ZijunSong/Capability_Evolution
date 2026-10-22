#!/usr/bin/env bash
# Submit TRIM Harness-1 BC+ full GPT-OSS-20B RL (4-step) training to CloudML.
# NOT YET SUBMITTED — review the job spec, then run this script to submit.
#
# JuiceFS mount is passed via the direct CLI flag --juicefs_mount_configs (NOT via the YAML
# --filename), so endpoint/cluster=bj9-cloudml is explicit in the API request, per requirement.
# The endpoint field is double-written alongside juiceFsCluster=bj9-cloudml for stability
# (the Go client currently honors juiceFsCluster; endpoint is kept for explicit intent).
#
# Target: queue 2334 (记忆对话, kscn-hd-cloudml-prod), 8x gm402-80g, GUARANTEED_PUBLIC, priority 8,
# tag MemQA-General-V5, not preemptible. Image: songzijun-cto:3050749-songzijun-cto.
set -euo pipefail

YAML="/mnt/songzijun/Capability_Evolution/TRIM/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_4step_20260911_not_submitted.yaml"

JUICEFS_MOUNTS='[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

echo "=== Submitting TRIM GPT-OSS-20B RL 4-step training ==="
echo "YAML:        $YAML"
echo "Queue:       2334 (记忆对话, gm402-80g, 8x GPU, GUARANTEED_PUBLIC)"
echo "Priority:   8 | Tag: MemQA-General-V5 | preemptible: false"
echo "JuiceFS:    volume=songzijun, endpoint/cluster=bj9-cloudml, mountPath=/mnt/songzijun"
echo "Image:      micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto"
echo "---"

cml custom_train submit \
  --filename "$YAML" \
  --juicefs_mount_configs "$JUICEFS_MOUNTS" \
  --tag "MemQA-General-V5"
