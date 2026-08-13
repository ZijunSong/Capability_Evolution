#!/usr/bin/env python3
"""Build query-disjoint same-state splits for Candidate-B micro-learnability tournament."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.same_state import audit_same_state, collect_same_state_dataset

COMPONENTS = [
    "subtractive_curation",
    "importance_tagging",
    "verify_tool",
]


def build_component_splits(
    component_id: str,
    out_dir: Path,
    *,
    train_n: int = 8000,
    valid_n: int = 512,
    test_n: int = 512,
    seed: int = 42,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        f"{component_id}_TRAIN_8K": ("train_q", train_n, seed),
        f"{component_id}_VALID_512": ("valid_q", valid_n, seed + 1),
        f"{component_id}_TEST_512": ("test_q", test_n, seed + 2),
    }
    meta: dict = {
        "component_id": component_id,
        "query_disjoint": True,
        "legacy_scope_path_used": False,
        "verify_natural_only_for_gate_l": component_id == "verify_tool",
        "splits": {},
    }
    for name, (prefix, n, split_seed) in splits.items():
        path = out_dir / f"{name}.jsonl"
        rows = collect_same_state_dataset(
            n_states=n,
            component_id=component_id,
            seed=split_seed,
            out_path=path,
            query_prefix=prefix,
        )
        audit = audit_same_state(rows)
        meta["splits"][name] = {
            "path": str(path),
            "n_states": len(rows),
            "query_prefix": prefix,
            "seed": split_seed,
            "audit": audit,
        }
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs/true_scape_candidate_b_tournament/data",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_meta: dict = {"components": {}, "legacy_scope_path_used": False}
    lines = [
        "# DATA_AUDIT — Candidate-B micro-learnability tournament",
        "",
        "- query_disjoint: true",
        "- legacy_scope_path_used: false",
        "- trainer: harness-1 same-state tool-token OPD",
        "",
    ]
    for comp in COMPONENTS:
        comp_seed = args.seed + sum(ord(c) for c in comp) % 97
        meta = build_component_splits(comp, args.out_dir, seed=comp_seed)
        all_meta["components"][comp] = meta
        lines.append(f"## {comp}")
        for name, info in meta["splits"].items():
            audit = info["audit"]
            lines.append(f"- `{name}`: n={info['n_states']} audit_pass={audit.get('pass')}")
        lines.append("")

    audit_path = args.out_dir.parent / "DATA_AUDIT.md"
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_dir / "DATA_AUDIT.json").write_text(
        json.dumps(all_meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(all_meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
