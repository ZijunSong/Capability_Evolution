#!/usr/bin/env bash
# Stage M (Multi-component annealing) — only after Gate S pass.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/stage_m}"
mkdir -p "$OUT_ROOT"

python3 - "$OUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
queue = {
  "0": {"job": "A->B sequential", "seed": 42},
  "1": {"job": "A->B sequential", "seed": 43},
  "2": {"job": "B->A sequential", "seed": 42},
  "3": {"job": "joint A+B dropout", "seed": 42},
  "4": {"job": "joint A+B dropout", "seed": 43},
  "5": {"job": "random dropout control", "seed": 42},
  "6": {"job": "prestage-guided annealing", "seed": 42},
  "7": {"job": "runtime mask sweep + Pareto", "seed": 42},
}
(out / "GPU_QUEUE.json").write_text(json.dumps(queue, indent=2) + "\n")
for gpu, spec in queue.items():
    gdir = out / f"gpu{gpu}"
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "RUN_MANIFEST.json").write_text(json.dumps({
        "schema_version": "scape_run_manifest_v1",
        "status": "queued",
        "gpu": int(gpu),
        **spec,
    }, indent=2) + "\n")
print(out / "GPU_QUEUE.json")
PY
