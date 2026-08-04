#!/usr/bin/env bash
# GPU3: shard3 Dup Base → O7 seeds
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup
GPU=3 PORT=9303 SHARD=shard3
scope8_log "GPU3 queue start"
scope8_run_dup_retention "${GPU}" "${SHARD}" "${BASE_MODEL}" "${PORT}" 0 base
scope8_run_dup_retention "${GPU}" "${SHARD}" "${R5}/merged/o7_r64_seed42" "${PORT}" 42 seed42
scope8_run_dup_retention "${GPU}" "${SHARD}" "${R5}/merged/o7_r64_seed43" "${PORT}" 43 seed43
scope8_run_dup_retention "${GPU}" "${SHARD}" "${R5}/merged/o7_r64_seed44" "${PORT}" 44 seed44
scope8_log "GPU3 queue complete"
