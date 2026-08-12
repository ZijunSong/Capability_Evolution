#!/usr/bin/env python3
"""Aggregate Phase A A0–A4 metrics and choose Stage1 view for Phase B."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.aggregate_frozen_replay import load_jsonl, operation_metrics
from training.scope_round9.aggregate_phase3_gate import _balanced_accuracy
from training.scope_round11.stage1_views import VIEW_NAMES

OUT = _REPO / "outputs/scope_round11/phase_a_state_factorization"


def _recall(matrix: dict, op: str) -> float:
    support = sum(matrix.get(op, {}).values()) if isinstance(matrix.get(op), dict) else 0
    if support <= 0:
        return float("nan")
    return matrix[op][op] / support


def _margins(rows: list[dict]) -> dict:
    margins = []
    err_margins = []
    for r in rows:
        logits = r.get("vllm_logits") or r.get("canonical_logits") or {}
        s_c = float(logits.get("CONTINUE", -1e9))
        s_r = float(logits.get("ROLLBACK_TO", -1e9))
        m = s_c - s_r
        margins.append(m)
        if r.get("pred_operation") != r.get("gold_operation"):
            err_margins.append(m)
    if not margins:
        return {}
    margins_sorted = sorted(margins)

    def q(p: float) -> float:
        if not margins_sorted:
            return float("nan")
        idx = min(len(margins_sorted) - 1, max(0, int(round(p * (len(margins_sorted) - 1)))))
        return margins_sorted[idx]

    mean = sum(margins) / len(margins)
    var = sum((x - mean) ** 2 for x in margins) / max(len(margins), 1)
    return {
        "margin_mean": mean,
        "margin_std": math.sqrt(var),
        "margin_q10": q(0.10),
        "margin_q50": q(0.50),
        "margin_q90": q(0.90),
        "error_margin_mean": (sum(err_margins) / len(err_margins)) if err_margins else None,
        "n_errors": len(err_margins),
    }


def metrics_for(rows: list[dict]) -> dict:
    opm = operation_metrics(rows)
    matrix = opm["confusion_matrix"]
    prior_counts = Counter(r.get("pred_operation") for r in rows)
    n = max(len(rows), 1)
    prior = {k: prior_counts.get(k, 0) / n for k in ("CONTINUE", "ROLLBACK_TO", "REPLAN")}
    bal = _balanced_accuracy(matrix) if callable(_balanced_accuracy) else opm.get("balanced_accuracy")
    # _balanced_accuracy may expect different signature — fall back.
    if bal is None or not isinstance(bal, (int, float)):
        crs = []
        for op in ("CONTINUE", "ROLLBACK_TO"):
            r = _recall(matrix, op)
            if not math.isnan(r):
                crs.append(r)
        bal = sum(crs) / max(len(crs), 1)
    return {
        "n": len(rows),
        "balanced_accuracy": float(bal),
        "ContinueRecall": _recall(matrix, "CONTINUE"),
        "RollbackRecall": _recall(matrix, "ROLLBACK_TO"),
        "predicted_CONTINUE_prior": prior["CONTINUE"],
        "predicted_ROLLBACK_prior": prior["ROLLBACK_TO"],
        "confusion_matrix": matrix,
        **_margins(rows),
    }


def gate_ok(m: dict) -> bool:
    return (
        float(m.get("ContinueRecall") or 0) >= 0.65
        and float(m.get("RollbackRecall") or 0) >= 0.65
        and float(m.get("balanced_accuracy") or 0) >= 0.65
    )


def pareto_score(m: dict) -> float:
    """Higher is better: geometric mean of CR/RR with bal as tie-break."""
    cr = max(float(m.get("ContinueRecall") or 0), 1e-6)
    rr = max(float(m.get("RollbackRecall") or 0), 1e-6)
    bal = float(m.get("balanced_accuracy") or 0)
    return (cr * rr) ** 0.5 + 0.01 * bal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase-a-root", type=Path, default=OUT)
    args = p.parse_args()
    root = args.phase_a_root
    per_view: dict = {}
    paired_path = root / "paired_events.jsonl"
    a0_live = {}
    if (root / "base_live" / "A0" / "canonical_vllm_replay.jsonl").exists():
        for r in load_jsonl(root / "base_live" / "A0" / "canonical_vllm_replay.jsonl"):
            a0_live[r.get("event_id")] = r

    with paired_path.open("w", encoding="utf-8") as pf:
        for view in VIEW_NAMES:
            per_view[view] = {}
            for split in ("offline_valid", "base_live"):
                path = root / split / view / "canonical_vllm_replay.jsonl"
                if not path.exists():
                    per_view[view][split] = {"missing": True, "path": str(path)}
                    continue
                rows = load_jsonl(path)
                per_view[view][split] = metrics_for(rows)
                if split == "base_live" and view != "A0":
                    for r in rows:
                        a0 = a0_live.get(r.get("event_id"))
                        if not a0:
                            continue
                        pf.write(
                            json.dumps(
                                {
                                    "event_id": r.get("event_id"),
                                    "view": view,
                                    "gold_operation": r.get("gold_operation"),
                                    "a0_pred": a0.get("pred_operation"),
                                    "view_pred": r.get("pred_operation"),
                                    "changed": a0.get("pred_operation") != r.get("pred_operation"),
                                    "a0_margin": float((a0.get("vllm_logits") or {}).get("CONTINUE", 0))
                                    - float((a0.get("vllm_logits") or {}).get("ROLLBACK_TO", 0)),
                                    "view_margin": float((r.get("vllm_logits") or {}).get("CONTINUE", 0))
                                    - float((r.get("vllm_logits") or {}).get("ROLLBACK_TO", 0)),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

    # Choose smallest view that passes on base_live; else Pareto-best.
    order = ["A1", "A2", "A3", "A4", "A0"]  # smallest → richest
    chosen = None
    for view in order:
        m = (per_view.get(view) or {}).get("base_live") or {}
        if m and not m.get("missing") and gate_ok(m):
            chosen = view
            break
    if chosen is None:
        scored = []
        for view in VIEW_NAMES:
            m = (per_view.get(view) or {}).get("base_live") or {}
            if m and not m.get("missing"):
                scored.append((pareto_score(m), view))
        scored.sort(reverse=True)
        chosen = scored[0][1] if scored else "A3"
        allow_closed_loop = False
    else:
        allow_closed_loop = True

    decision = {
        "selected_stage1_view": chosen,
        "gate_threshold": {"ContinueRecall": 0.65, "RollbackRecall": 0.65, "balanced_accuracy": 0.65},
        "any_view_passed_gate": allow_closed_loop,
        "allow_closed_loop_after_phase_b": allow_closed_loop,
        "note": (
            "Selected smallest view meeting base_live gate."
            if allow_closed_loop
            else "No view met gate; selected Pareto-best; Phase B representation training only."
        ),
    }
    (root / "per_view_metrics.json").write_text(json.dumps(per_view, indent=2) + "\n", encoding="utf-8")
    (root / "PHASE_A_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# FEATURE_FACTOR_REPORT",
        "",
        f"Selected Stage1 view: **{chosen}**",
        f"Any view passed Continue/Rollback/bal >= 0.65 on base_live: **{allow_closed_loop}**",
        "",
        "## base_live metrics",
        "",
        "| view | bal | ContinueRecall | RollbackRecall | CONTINUE prior | ROLLBACK prior | margin_mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for view in VIEW_NAMES:
        m = (per_view.get(view) or {}).get("base_live") or {}
        if m.get("missing"):
            lines.append(f"| {view} | missing | | | | | |")
            continue
        lines.append(
            f"| {view} | {m.get('balanced_accuracy', float('nan')):.4f} | "
            f"{m.get('ContinueRecall', float('nan')):.4f} | {m.get('RollbackRecall', float('nan')):.4f} | "
            f"{m.get('predicted_CONTINUE_prior', float('nan')):.4f} | "
            f"{m.get('predicted_ROLLBACK_prior', float('nan')):.4f} | "
            f"{m.get('margin_mean', float('nan')):.4f} |"
        )
    lines += ["", "## offline_valid metrics", ""]
    lines.append("| view | bal | ContinueRecall | RollbackRecall |")
    lines.append("|---|---:|---:|---:|")
    for view in VIEW_NAMES:
        m = (per_view.get(view) or {}).get("offline_valid") or {}
        if m.get("missing"):
            lines.append(f"| {view} | missing | | |")
            continue
        lines.append(
            f"| {view} | {m.get('balanced_accuracy', float('nan')):.4f} | "
            f"{m.get('ContinueRecall', float('nan')):.4f} | {m.get('RollbackRecall', float('nan')):.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- A0 = current full Stage1 (candidate semantics + IDs).",
        "- A1 = pure state-only (no candidate content).",
        "- A2 = state + rollback feasibility scalars.",
        "- A3 = A2 + failure/progress scalars.",
        "- A4 = A0 shape with candidate text masked (shape/length control).",
        "",
        decision["note"],
        "",
    ]
    report = "\n".join(lines)
    (root / "FEATURE_FACTOR_REPORT.md").write_text(report, encoding="utf-8")
    (_REPO / "outputs/scope_round11/FEATURE_FACTOR_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(report)


if __name__ == "__main__":
    main()
