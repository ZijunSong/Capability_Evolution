#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python - <<'PY'
from pathlib import Path
from experiments.common.validation import validate_run_dir
roots = [Path("outputs/iclr_ablations"), Path("outputs/iclr_baselines")]
n = 0
bad = 0
for root in roots:
    if not root.exists():
        continue
    for summary in root.rglob("summary.json"):
        n += 1
        errs = validate_run_dir(summary.parent)
        if errs:
            bad += 1
            print(summary.parent, errs)
print(f"checked={n} bad={bad}")
raise SystemExit(1 if bad else 0)
PY
