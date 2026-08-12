#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
for v in a1_same_state_on_policy a1_trajectory_teacher a1_cross_state_matched a1_static_offline; do
  python -m experiments.common.launcher --experiment-id "$v" --smoke-query-limit 16 --resume "$@"
done
