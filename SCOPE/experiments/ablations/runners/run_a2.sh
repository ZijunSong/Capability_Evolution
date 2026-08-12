#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
for v in a2_current_student_on_policy a2_base_model_states a2_full_harness_states a2_stale_checkpoint_states a2_mixed_replay_states; do
  python -m experiments.common.launcher --experiment-id "$v" --smoke-query-limit 16 --resume "$@"
done
