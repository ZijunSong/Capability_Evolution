#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python - <<'PY'
from experiments.common.registry import ExperimentRegistry
from experiments.common.spec import ExperimentSpec
reg = ExperimentRegistry()
errs = reg.validate()
for eid in reg.ids():
    spec = reg.resolve(eid)
    assert isinstance(spec, ExperimentSpec)
print(f"validated {len(reg.ids())} experiments; errors={len(errs)}")
if errs:
    for e in errs:
        print(e)
    raise SystemExit(1)
PY
