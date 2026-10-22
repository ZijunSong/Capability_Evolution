#!/usr/bin/env bash
# Submit minimal vLLM EngineArgs diagnostic (does LLM()/EngineArgs accept tool_call_parser?).
# Determines whether legacy_local's 0% tool-call-parse bug can be fixed by passing these to
# vllm_rollout_worker.py's LLM(**llm_kwargs). No model load — pure EngineArgs dataclass inspect.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/entry_diag_vllm_engineargs_20261021.sh"
test -f "${ENTRY}"

cml custom_train submit \
  --job_name trim-diag-vllm-engineargs-20261021 \
  --description "Minimal vLLM 0.19.1 EngineArgs diagnostic: does LLM() accept tool_call_parser/enable_auto_tool_choice/reasoning_parser? Pure dataclass inspect, no model load. Determines if legacy_local tool-call-parse bug is fixable." \
  --access_type PRIVATE --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --framework pytorch --queue_id 2334 --priority 8 --preemptible=false \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}' \
  --image_command "$(cat "${ENTRY}")"