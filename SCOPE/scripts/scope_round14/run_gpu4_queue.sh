#!/usr/bin/env bash
# GPU4: context_budget_routing
set -euo pipefail
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/_capability_queue.sh"
r14_setup

GPU="${1:-4}"
r14_capability_queue "${GPU}" context_budget_routing \
  "${OUT}/gpu4_context_budget" \
  "${DATA_DIR}/context_budget_routing" \
  "${OUT}/gpu4_context_budget/HEARTBEAT"
