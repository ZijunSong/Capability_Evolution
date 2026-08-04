#!/usr/bin/env bash
# GPU6: Qwen3-1.7B AgentCore → FullHarness shard2 + injected rollback
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup
GPU=6 PORT=9306 SHARD=shard2
MODEL="/data/ppnm/models/Qwen3-1.7B"
AC="${REPO_ROOT}/harness/configs/agent_core.yaml"
FH="${REPO_ROOT}/harness/configs/agent_core_full_harness.yaml"
scope8_log "GPU6 queue start"
scope8_run_agent_core "${GPU}" "${SHARD}" "${MODEL}" "${PORT}" "${AC}" qwen3_1.7b_agent_core
scope8_run_agent_core "${GPU}" "${SHARD}" "${MODEL}" "${PORT}" "${FH}" qwen3_1.7b_full_harness
scope8_collect_rollback "${GPU}" "${SHARD}" "${MODEL}" "${PORT}" injected
scope8_log "GPU6 queue complete"
