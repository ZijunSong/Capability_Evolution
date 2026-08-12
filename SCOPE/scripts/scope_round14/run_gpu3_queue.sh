#!/usr/bin/env bash
# GPU3: evidence_admission
set -euo pipefail
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_capability_queue.sh"
r14_setup

GPU="${1:-3}"
r14_capability_queue "${GPU}" evidence_admission \
  "${OUT}/gpu3_evidence_admission" \
  "${DATA_DIR}/evidence_admission" \
  "${OUT}/gpu3_evidence_admission/HEARTBEAT"
