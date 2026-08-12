#!/usr/bin/env python3
"""Write ROUND9_DIAGNOSIS_REPORT.md from available Round 9 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round9"


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT / "ROUND9_DIAGNOSIS_REPORT.md")
    args = p.parse_args()

    reagg = _load(OUT / "reaggregate_round8" / "metric_diff_vs_original.json") or {}
    root = _load(OUT / "ROOT_CAUSE_DECISION.json") or _load(OUT / "diagnosis" / "ROOT_CAUSE_DECISION.json") or {}
    offline = _load(OUT / "OFFLINE_GATE_ROUND9.json") or {}
    smoke = _load(OUT / "H20_SMOKE_GATE.json") or {}
    hard = _load(OUT / "HARD_CAPABILITY_GATE_ROUND9.json") or {}
    frozen = _load(_REPO / "artifacts/datasets/scope_round9/frozen_replay/dataset_report.json") or {}

    # Summarize Wave A barrier
    wave_a = {}
    for path in sorted((OUT / "wave_a").glob("*/WAVE_A_REPORT.json")):
        wave_a[path.parent.name] = json.loads(path.read_text(encoding="utf-8")).get("barrier_a_pass")

    lines = [
        "# Round 9 Diagnosis Report",
        "",
        "## 1. Round 8 75%→6% fracture attribution",
        "",
        f"- Reaggregate changed metrics: `{reagg.get('aggregation_bug_changed', [])}`",
        f"- Low accuracy still holds: `{reagg.get('low_accuracy_still_holds', [])}`",
        f"- Main seeds pass after reagg: `{reagg.get('main_seeds_pass')}`",
        "- Scorer/input contract was repaired in Round 9 (single-source effective input, as-is scoring, stable tie-break, index-aligned parity).",
        "- Live class prior / CONTINUE collapse remains a behavioral bottleneck independent of aggregation bugs.",
        "",
        "## 2. Checkpoint 8.5% and representation",
        "",
        f"- Frozen offline rollback coverage: `{((frozen.get('datasets') or {}).get('offline_valid') or {}).get('gold_candidate_coverage')}`",
        f"- Frozen base_live rollback coverage: `{((frozen.get('datasets') or {}).get('base_live') or {}).get('gold_candidate_coverage')}`",
        "- Raw unstable checkpoint IDs were replaced by local C0..Ck IDs + structured summaries for hierarchical training.",
        "",
        "## 3. Hierarchical selector on frozen live",
        "",
        f"- Root-cause diagnosis: `{json.dumps(root.get('diagnosis', {}), ensure_ascii=False)}`",
        f"- Offline gate: `{json.dumps({k: offline.get(k) for k in ('offline_gate_pass','seed_span_ok','seed_span_operation_bal_acc')}, ensure_ascii=False)}`",
        "",
        "## 4. 100q Hard-capability Gate",
        "",
        f"- Smoke gate: `{smoke.get('smoke_pass')}`",
        f"- Hard capability positive: `{hard.get('ROUND9_HARD_CAPABILITY_POSITIVE')}`",
        f"- Recommend 830q: `{hard.get('RECOMMEND_ROLLBACK_830')}`",
        "",
        "## 5. Next step constraint",
        "",
        "- If failed: only attack the bottlenecks listed in ROOT_CAUSE_DECISION; do not expand to multi-capability / weighting / DAgger / RL.",
        "",
        "## Appendix: Wave A barrier_a_pass",
        "",
        f"```json\n{json.dumps(wave_a, indent=2)}\n```",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
