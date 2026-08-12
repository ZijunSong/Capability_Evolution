#!/usr/bin/env bash
# Stage S (Single-component migration) 8-GPU queue scaffolding.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/stage_s}"
mkdir -p "$OUT_ROOT"

python3 - "$OUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
queue = {
  "0": {"job": "SCAPE-A distill-only", "seed": 42},
  "1": {"job": "SCAPE-A distill-only", "seed": 43},
  "2": {"job": "SCAPE-A distill-only", "seed": 44},
  "3": {"job": "SCAPE-B or A+confidence", "seed": 42},
  "4": {"job": "SCAPE-B or A name-only", "seed": 43},
  "5": {"job": "SCAPE-B or A args ablation", "seed": 44},
  "6": {"job": "SCAPE-A + RL", "seed": 42, "requires": "distill_positive_compensation"},
  "7": {"job": "SCAPE-A + RL", "seed": 43, "requires": "distill_positive_compensation"},
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
