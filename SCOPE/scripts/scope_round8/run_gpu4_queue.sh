#!/usr/bin/env bash
# GPU4: Qwen2.5 AgentCore 100q shard0 + natural rollback collection
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup
GPU=4 PORT=9304 SHARD=shard0
AC="${REPO_ROOT}/harness/configs/agent_core.yaml"
scope8_log "GPU4 queue start"
scope8_run_agent_core "${GPU}" "${SHARD}" "${BASE_MODEL}" "${PORT}" "${AC}" agent_core
scope8_collect_rollback "${GPU}" "${SHARD}" "${BASE_MODEL}" "${PORT}" natural
scope8_log "GPU4 queue complete"
