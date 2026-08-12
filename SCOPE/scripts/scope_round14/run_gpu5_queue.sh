#!/usr/bin/env bash
# GPU5: external_verification_routing
set -euo pipefail
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_capability_queue.sh"
r14_setup

GPU="${1:-5}"
r14_capability_queue "${GPU}" external_verification_routing \
  "${OUT}/gpu5_external_verify" \
  "${DATA_DIR}/external_verification_routing" \
  "${OUT}/gpu5_external_verify/HEARTBEAT"
