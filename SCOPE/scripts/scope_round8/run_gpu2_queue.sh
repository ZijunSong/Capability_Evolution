#!/usr/bin/env bash
# GPU2: shard2 Dup Base → O7 seeds
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup
GPU=2 PORT=9302 SHARD=shard2
scope8_log "GPU2 queue start"
scope8_run_dup_retention "${GPU}" "${SHARD}" "${BASE_MODEL}" "${PORT}" 0 base
scope8_run_dup_retention "${GPU}" "${SHARD}" "${R5}/merged/o7_r64_seed42" "${PORT}" 42 seed42
scope8_run_dup_retention "${GPU}" "${SHARD}" "${R5}/merged/o7_r64_seed43" "${PORT}" 43 seed43
scope8_run_dup_retention "${GPU}" "${SHARD}" "${R5}/merged/o7_r64_seed44" "${PORT}" 44 seed44
scope8_log "GPU2 queue complete"
