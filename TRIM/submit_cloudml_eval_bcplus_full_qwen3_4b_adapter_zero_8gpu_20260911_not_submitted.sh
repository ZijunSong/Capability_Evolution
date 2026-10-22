#!/usr/bin/env bash
# Submit the Qwen3-4B adapter zero-component eval job. NOT YET EXECUTED.
# JuiceFS mount passed directly via CLI --juicefs_mount_configs (NOT via the
# YAML --filename path) so endpoint=bj9-cloudml is explicit in the API request.
# Both endpoint and juiceFsCluster are set to bj9-cloudml (the safe double-write
# form; endpoint is ignored by the Go layer, juiceFsCluster is consumed).
set -euxo pipefail

source /mnt/songzijun/mify_api.env
export CML_WORKSPACE_ID=10063
export CML_CONTEXT=songzijun-cto

cml custom_train submit \
  --filename /mnt/songzijun/Capability_Evolution/TRIM/cloudml_eval_bcplus_full_qwen3_4b_adapter_zero_8gpu_20260911_not_submitted.yaml \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","endpoint":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --tag MemQA-General-V5
