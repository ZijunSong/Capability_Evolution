#!/usr/bin/env bash
# GPU5: Qwen2.5 AgentCore+FullHarness shard1 + natural rollback
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup
GPU=5 PORT=9305 SHARD=shard1
FH="${REPO_ROOT}/harness/configs/agent_core_full_harness.yaml"
scope8_log "GPU5 queue start"
scope8_run_agent_core "${GPU}" "${SHARD}" "${BASE_MODEL}" "${PORT}" "${FH}" full_harness
scope8_collect_rollback "${GPU}" "${SHARD}" "${BASE_MODEL}" "${PORT}" natural
scope8_log "GPU5 queue complete"
