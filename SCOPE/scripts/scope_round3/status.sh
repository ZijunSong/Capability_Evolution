#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/../.." && pwd)/outputs/scope_round3/logs"
for g in $(seq 0 7); do
  echo "=== GPU${g} ==="
  find "${ROOT}/gpu${g}" -name status 2>/dev/null | while read -r f; do
    task=$(basename "$(dirname "$f")")
    echo "  ${task}: $(cat "$f")"
  done
done
