#!/usr/bin/env bash
# Round 5 B5 — Top-2 closed-loop 50q (wave launch, nohup per shard)
# Placeholder: runs after B4 gate; actual rollout reuses round3/4 closed-loop infra.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round5/_common.sh"
scope5_setup

if [[ ! -f "${OUT}/B4_TOP2" ]]; then
  echo "B5 skip: B4_TOP2 not found"
  exit 0
fi

logfile="${LOG_DIR}/b5/closed_loop.log"
mkdir -p "${LOG_DIR}/b5"

# TODO: wire round3 closed-loop runner with Top-2 from B4_TOP2
# For now mark as pending implementation once B4 gate completes.
echo "B5 closed-loop pending — waiting for B4_TOP2 models and rollout script wiring" | tee "${logfile}"
echo "Top-2 candidates:" | tee -a "${logfile}"
cat "${OUT}/B4_TOP2" | tee -a "${logfile}"
