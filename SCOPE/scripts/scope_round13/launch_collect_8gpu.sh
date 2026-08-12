#!/usr/bin/env bash
# Barrier1: TRAIN200 on GPU0-5 (6 shards), VALID100 on GPU6-7 (2 shards).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

r13_log "Launch Barrier1 on-policy collect 8gpu"

for gpu in 0 1 2 3 4 5; do
  shard="shard${gpu}"
  nohup bash "$(dirname "$0")/run_collect_gpu.sh" "${gpu}" train "${shard}" 6 \
    >> "${LOG_DIR}/collect_train_${shard}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/collect_train_gpu${gpu}.pid"
  r13_log "started train ${shard} on GPU${gpu} pid=$!"
  sleep 25
done

for gpu in 6 7; do
  shard_idx=$((gpu - 6))
  shard="shard${shard_idx}"
  nohup bash "$(dirname "$0")/run_collect_gpu.sh" "${gpu}" valid "${shard}" 2 \
    >> "${LOG_DIR}/collect_valid_${shard}_supervisor.log" 2>&1 &
  echo $! > "${PID_DIR}/collect_valid_gpu${gpu}.pid"
  r13_log "started valid ${shard} on GPU${gpu} pid=$!"
  sleep 25
done

r13_log "All 8 collect jobs launched"
