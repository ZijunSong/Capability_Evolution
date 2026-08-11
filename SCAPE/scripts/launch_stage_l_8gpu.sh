#!/usr/bin/env bash
# Stage L (Micro-Learnability) 8-GPU queue — scaffolding launcher.
# Does not start heavy training until candidates + env preflight pass.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/stage_l}"
mkdir -p "$OUT_ROOT"

CAND_A="${CANDIDATE_A:-}"
CAND_B="${CANDIDATE_B:-}"
if [[ -z "$CAND_A" && -f "$ROOT/outputs/scape_prestage/CANDIDATE_SELECTION.json" ]]; then
  CAND_A=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("A",{}).get("component_id",""))' \
    "$ROOT/outputs/scape_prestage/CANDIDATE_SELECTION.json")
  CAND_B=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("B",{}).get("component_id",""))' \
    "$ROOT/outputs/scape_prestage/CANDIDATE_SELECTION.json")
fi

echo "[Stage L] Candidate A=${CAND_A:-UNSET} B=${CAND_B:-UNSET}"
echo "[Stage L] Writing queue manifests under $OUT_ROOT (dry scaffolding)."

python3 - "$OUT_ROOT" "$CAND_A" "$CAND_B" <<'PY'
import json, sys
from pathlib import Path
out, a, b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
queue = {
  "0": {"job": "A-L512/2K/8K", "component": a, "seed": 42, "kind": "learnability"},
  "1": {"job": "A-L512/2K/8K", "component": a, "seed": 43, "kind": "learnability"},
  "2": {"job": "B-L512/2K/8K", "component": b, "seed": 42, "kind": "learnability"},
  "3": {"job": "B-L512/2K/8K", "component": b, "seed": 43, "kind": "learnability"},
  "4": {"job": "same_state_action_ce", "component": a, "seed": 42, "kind": "baseline"},
  "5": {"job": "full_response_opd", "component": a, "seed": 42, "kind": "baseline"},
  "6": {"job": "offpolicy_harness_trace", "component": a, "seed": 42, "kind": "baseline"},
  "7": {"job": "oneshot_full_to_slim", "component": a, "seed": 42, "kind": "baseline"},
}
out.mkdir(parents=True, exist_ok=True)
(out / "GPU_QUEUE.json").write_text(json.dumps(queue, indent=2) + "\n")
for gpu, spec in queue.items():
    gdir = out / f"gpu{gpu}"
    gdir.mkdir(exist_ok=True)
    (gdir / "RUN_MANIFEST.json").write_text(json.dumps({
        "schema_version": "scape_run_manifest_v1",
        "status": "queued",
        "gpu": int(gpu),
        **spec,
    }, indent=2) + "\n")
print(f"wrote {out/'GPU_QUEUE.json'}")
PY

echo "NOTE: Training entrypoints are intentionally not auto-started."
echo "Pass SCAPE_START_TRAINING=1 after Gate inputs are ready."
