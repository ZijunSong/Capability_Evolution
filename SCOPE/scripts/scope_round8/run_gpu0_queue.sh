#!/usr/bin/env bash
# GPU0: shard0 Dup Base → O7 seeds
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup
GPU=0 PORT=9300 SHARD=shard0
scope8_log "GPU0 queue start"
scope8_run_dup_retention "${GPU}" "${SHARD}" "${BASE_MODEL}" "${PORT}" 0 base
scope8_run_dup_retention "${GPU}" "${SHARD}" "${R5}/merged/o7_r64_seed42" "${PORT}" 42 seed42
scope8_run_dup_retention "${GPU}" "${SHARD}" "${R5}/merged/o7_r64_seed43" "${PORT}" 43 seed43
scope8_run_dup_retention "${GPU}" "${SHARD}" "${R5}/merged/o7_r64_seed44" "${PORT}" 44 seed44
scope8_log "GPU0 queue complete"
