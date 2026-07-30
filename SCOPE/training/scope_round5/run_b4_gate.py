#!/usr/bin/env python3
"""Evaluate B4 gate for closed-loop eligibility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs/scope_round5/b4_full/offline"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def gate(metrics: dict) -> bool:
    keep_rec = metrics.get("KEEP_EVIDENCE", {}).get("recall", 0)
    skip_rec = metrics.get("SKIP_DUPLICATE", {}).get("recall", 0)
    bal_acc = metrics.get("balanced_accuracy", 0)
    macro_f1 = metrics.get("macro_f1", 0)
    pred_dist = metrics.get("prediction_distribution", {})
    n_classes = sum(1 for k, v in pred_dist.items() if v > 0 and k != "PARSE_FAIL")
    return (
        keep_rec > 0 and skip_rec > 0
        and bal_acc > 0.5
        and macro_f1 > 0.448  # all-KEEP baseline from Round4
        and n_classes >= 2
    )


def main() -> None:
    results = {}
    top = []
    for p in sorted(OUT.glob("*.json")):
        m = json.loads(p.read_text())
        ok = gate(m)
        results[p.stem] = {
            "balanced_accuracy": m.get("balanced_accuracy"),
            "macro_f1": m.get("macro_f1"),
            "keep_recall": m.get("KEEP_EVIDENCE", {}).get("recall"),
            "skip_recall": m.get("SKIP_DUPLICATE", {}).get("recall"),
            "gate_pass": ok,
            "margins": m.get("margins"),
        }
        if ok:
            top.append((m.get("macro_f1", 0), p.stem))

    top.sort(reverse=True)
    top2 = [t[1] for t in top[:2]]
    b4_pass = len(top) > 0

    report = {"variants": results, "top2": top2, "B4_PASS": b4_pass}
    out_path = _REPO / "outputs/scope_round5/b4_full/B4_GATE.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    (_REPO / "outputs/scope_round5/B4_PASS").write_text(str(b4_pass) + "\n")
    if top2:
        (_REPO / "outputs/scope_round5/B4_TOP2").write_text("\n".join(top2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
