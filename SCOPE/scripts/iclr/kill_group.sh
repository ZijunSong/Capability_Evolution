#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
GROUP_DIR="${1:?usage: kill_group.sh <pid_dir>}"
if [[ ! -d "$GROUP_DIR" ]]; then
  echo "missing $GROUP_DIR"
  exit 1
fi
for pidf in "$GROUP_DIR"/*.pid; do
  [[ -f "$pidf" ]] || continue
  pid=$(cat "$pidf")
  if kill -0 "$pid" 2>/dev/null; then
    echo "killing $pid from $pidf"
    kill "$pid" || true
  else
    echo "not running: $pid"
  fi
done
