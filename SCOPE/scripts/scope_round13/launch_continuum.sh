#!/usr/bin/env bash
# Full Round13 continuum: Barrier0 → collect 8gpu → monitor advances rest.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

r13_log "======== Round13 continuum start ========"

# Barrier 0
r13_log "Barrier 0: manifests + retirement note + env snapshot"
python training/scope_round13/create_r13_manifests.py \
  >> "${LOG_DIR}/create_manifests.log" 2>&1
python training/scope_round13/write_barrier0.py \
  >> "${LOG_DIR}/barrier0.log" 2>&1

# Barrier 1
bash "$(dirname "$0")/launch_collect_8gpu.sh"

# Monitor (stale restart + phase advance)
nohup bash "$(dirname "$0")/monitor_loop.sh" auto \
  >> "${LOG_DIR}/monitor_loop.log" 2>&1 &
echo $! > "${PID_DIR}/monitor_loop.pid"
r13_log "monitor_loop pid=$!"

r13_log "======== continuum launcher returned ========"
