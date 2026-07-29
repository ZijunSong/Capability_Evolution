#!/usr/bin/env bash
# Re-run only FAILED tasks (status != DONE)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)/outputs/scope_round3/logs"
for g in $(seq 0 7); do
  for d in "${ROOT}/gpu${g}"/*/; do
    [[ -d "$d" ]] || continue
    if [[ -f "${d}/status" && "$(cat "${d}/status")" == "FAILED" ]]; then
      task=$(basename "$d")
      echo "Rerun GPU${g} ${task}"
      rm -f "${d}/status"
    fi
  done
done
PHASE="${1:-all}" bash "$(dirname "$0")/run_all_8gpu.sh"
