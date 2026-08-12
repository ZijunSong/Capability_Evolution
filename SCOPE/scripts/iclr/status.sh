#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
ROOT_OUT="${1:-outputs/iclr_ablations}"
python - <<PY
import json
from pathlib import Path
root = Path("$ROOT_OUT")
rows = []
for done in root.rglob("DONE"):
    d = done.parent
    summary = {}
    sp = d / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text())
    manifest = {}
    mp = d / "run_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())
    pid = manifest.get("pid")
    rows.append({
        "dir": str(d),
        "experiment_id": summary.get("experiment_id"),
        "status": summary.get("status"),
        "n_queries": summary.get("n_queries"),
        "pid": pid,
        "done": True,
        "errors": summary.get("errors"),
    })
for sp in root.rglob("summary.json"):
    if (sp.parent / "DONE").exists():
        continue
    s = json.loads(sp.read_text())
    rows.append({
        "dir": str(sp.parent),
        "experiment_id": s.get("experiment_id"),
        "status": s.get("status"),
        "n_queries": s.get("n_queries"),
        "done": False,
        "errors": s.get("errors"),
    })
print(f"runs={len(rows)}")
for r in rows:
    print(f"{r.get('experiment_id')}\tdone={r.get('done')}\tstatus={r.get('status')}\tnq={r.get('n_queries')}\t{r.get('dir')}")
PY
