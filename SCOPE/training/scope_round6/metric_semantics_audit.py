#!/usr/bin/env python3
"""DCR vs direct admission metrics semantics audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round6.aggregate_round6 import aggregate_from_jsonl
from training.scope_round6.common import B6_ROOT, OUT, SEEDS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT / "phase_b")
    args = p.parse_args()

    lines = [
        "# Metric Semantics Audit",
        "",
        "DCR (duplicate_curate_rate) measures downstream `actually_curated` on duplicates.",
        "Direct behavior metrics measure pred KEEP/SKIP vs shadow label.",
        "",
        "## Base vs O7 comparison",
        "",
    ]

    for variant, label in [("base", "Base"), *[(f"best_o7_{s}", f"O7-seed{s}") for s in SEEDS]]:
        root = B6_ROOT / variant
        shard_dirs = sorted(root.glob("shard*"))
        if not shard_dirs:
            continue
        # aggregate all shards
        from training.scope_round6.common import load_jsonl
        all_eps = []
        all_ev = []
        for sd in shard_dirs:
            all_eps.extend(load_jsonl(sd / "episodes.jsonl"))
            all_ev.extend(load_jsonl(sd / "dup_admission_events.jsonl"))
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        (tmp / "episodes.jsonl").write_text(
            "\n".join(__import__("json").dumps(e) for e in all_eps) + "\n", encoding="utf-8"
        )
        (tmp / "dup_admission_events.jsonl").write_text(
            "\n".join(__import__("json").dumps(e) for e in all_ev) + "\n", encoding="utf-8"
        )
        rep = aggregate_from_jsonl(tmp / "episodes.jsonl", tmp / "dup_admission_events.jsonl")
        d = rep.get("direct_behavior", {})
        lines.append(f"### {label}")
        lines.append(f"- DCR={rep.get('DCR', 0):.4f} FSR(tel)={rep.get('FSR', 0):.4f}")
        lines.append(
            f"- DupRejectRecall={d.get('DupRejectRecall', 0):.4f} "
            f"FalseSkipRate(direct)={d.get('FalseSkipRate', 0):.4f} "
            f"BalancedAcc={d.get('BalancedAcc', 0):.4f}"
        )
        lines.append(f"- predicted_SKIP_prior={d.get('predicted_SKIP_prior', 0):.4f}")
        lines.append("")

    out = args.output_dir / "metric_semantics_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
