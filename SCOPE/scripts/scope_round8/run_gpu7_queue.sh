#!/usr/bin/env bash
# GPU7: Qwen3-30B AgentCore → FullHarness shard3 + injected rollback
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup
GPU=7 PORT=9307 SHARD=shard3
MODEL="/data/ppnm/models/Qwen3-30B-A3B-Thinking-2507"
AC="${REPO_ROOT}/harness/configs/agent_core.yaml"
FH="${REPO_ROOT}/harness/configs/agent_core_full_harness.yaml"
scope8_log "GPU7 queue start"
scope8_run_agent_core "${GPU}" "${SHARD}" "${MODEL}" "${PORT}" "${AC}" qwen3_30b_agent_core
scope8_run_agent_core "${GPU}" "${SHARD}" "${MODEL}" "${PORT}" "${FH}" qwen3_30b_full_harness
scope8_collect_rollback "${GPU}" "${SHARD}" "${MODEL}" "${PORT}" injected
scope8_log "GPU7 queue complete"
