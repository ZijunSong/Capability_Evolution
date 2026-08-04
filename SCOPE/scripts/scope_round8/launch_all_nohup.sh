#!/usr/bin/env bash
# Nohup launcher for Round 8 full pipeline (Phase 1 immediate; Phase 2/3 after gates)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${REPO_ROOT}/outputs/scope_round8"
mkdir -p "${OUT}/logs" "${OUT}/pids"

nohup bash "${REPO_ROOT}/scripts/scope_round8/launch_phase1_8gpu.sh" \
  > "${OUT}/logs/launch_phase1.nohup.log" 2>&1 &
echo $! > "${OUT}/pids/launch_phase1.pid"
echo "Phase 1 launched pid=$(cat "${OUT}/pids/launch_phase1.pid")"
echo "Monitor: bash scripts/scope_round8/status.sh"
echo "Phase 2/3 run after Gate 1A/1B/1C and Offline Gate — see 0802-todo1.md"
