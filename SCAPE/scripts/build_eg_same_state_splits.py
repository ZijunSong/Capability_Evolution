#!/usr/bin/env python3
"""Build query-disjoint EG_TRAIN_8K / EG_VALID_1K / EG_TEST_1K splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.same_state import audit_same_state, build_query_disjoint_splits


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument(
    "--out-dir",
    type=Path,
    default=REPO / "outputs/true_scape_evidence_graph/data",
  )
  ap.add_argument("--component-id", default="evidence_graph")
  ap.add_argument("--seed", type=int, default=42)
  args = ap.parse_args()

  meta = build_query_disjoint_splits(
    component_id=args.component_id,
    out_dir=args.out_dir,
    seed=args.seed,
  )

  lines = [
    "# DATA_AUDIT — evidence_graph same-state splits",
    "",
    f"- component: `{args.component_id}`",
    f"- query_disjoint: true",
    f"- legacy_scope_path_used: false",
    "",
  ]
  for name, info in meta["splits"].items():
    audit = info["audit"]
    lines.append(f"## {name}")
    lines.append(f"- path: `{info['path']}`")
    lines.append(f"- n_states: {info['n_states']}")
    lines.append(f"- query_prefix: `{info['query_prefix']}`")
    lines.append(f"- audit_pass: {audit.get('pass')}")
    lines.append("")

  out_root = args.out_dir.parent
  out_root.mkdir(parents=True, exist_ok=True)
  (out_root / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(json.dumps(meta, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
