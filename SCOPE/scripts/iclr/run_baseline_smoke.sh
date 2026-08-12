#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
for eid in b0_base_hmin b1_base_full_harness b2_prompt_hint b3_trajectory_sft b4_direct_operation_sft b5_dagger; do
  python -m experiments.common.launcher --experiment-id "$eid" --smoke-query-limit 2 --resume
done
# External: dry-run only unless env ready
for eid in b_seed_dryrun b_opid_dryrun; do
  python -m experiments.common.launcher --experiment-id "$eid" --dry-run --smoke-query-limit 2 --resume
done
echo "baseline smoke complete"
