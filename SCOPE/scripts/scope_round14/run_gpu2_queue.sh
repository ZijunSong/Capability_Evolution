#!/usr/bin/env bash
# GPU2: verification_routing
set -euo pipefail
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_capability_queue.sh"
r14_setup

GPU="${1:-2}"
r14_capability_queue "${GPU}" verification_routing \
  "${OUT}/gpu2_verify_routing" \
  "${DATA_DIR}/verification_routing" \
  "${OUT}/gpu2_verify_routing/HEARTBEAT"
