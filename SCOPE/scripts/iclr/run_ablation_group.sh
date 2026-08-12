#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
GROUP="${1:?usage: run_ablation_group.sh <group> [--dry-run]}"
DRY="${2:-}"
python - <<PY
from experiments.common.registry import ExperimentRegistry
from experiments.common.launcher import launch
from experiments.ablations.runners.dispatch import dispatch_ablation
reg = ExperimentRegistry()
dry = "$DRY" == "--dry-run"
for eid in reg.by_group("$GROUP"):
    spec = reg.resolve(eid, dry_run=dry, smoke_query_limit=4, resume=True)
    print("launch", eid)
    launch(spec, dispatch_ablation)
print("done group", "$GROUP")
PY
