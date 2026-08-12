#!/usr/bin/env bash
# GPU1: stop_decision
set -euo pipefail
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_capability_queue.sh"
r14_setup

GPU="${1:-1}"
r14_capability_queue "${GPU}" stop_decision \
  "${OUT}/gpu1_stop" \
  "${DATA_DIR}/stop_decision" \
  "${OUT}/gpu1_stop/HEARTBEAT"
