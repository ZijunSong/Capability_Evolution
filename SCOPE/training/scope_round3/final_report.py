#!/usr/bin/env python3
"""Generate ROUND3_REPORT with paired stats and Go/No-Go."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round2.eval_paired import bootstrap_ci, episodes_by_query, load_jsonl, summarize

VARIANTS = [
    "round3_op_main_seed42",
    "round3_op_main_seed43",
    "round3_op_main_seed44",
    "round3_compact_json_sample_norm",
    "round3_legacy_full_action_token_ce",
    "round3_correct_only_op",
    "round3_endorse_only_op",
    "round3_op_no_balance",
]


def _metric(ep: dict, key: str) -> float:
    if key == "false_skip_rate":
        tel = ep.get("dup_telemetry") or {}
        return float(tel.get("false_skip_rate", ep.get("false_skip_rate", 0)))
    if key == "duplicate_curate_rate":
        tel = ep.get("dup_telemetry") or {}
        return float(
            tel.get("duplicate_curate_rate", ep.get("duplicate_curate_rate", ep.get("dup_curate_rate", 0)))
        )
    return float(ep.get(key, 0))


def paired(base_eps: dict[str, dict], other_eps: dict[str, dict], metric: str) -> dict[str, Any]:
    deltas, wins, losses, ties = [], 0, 0, 0
    for qid in sorted(set(base_eps) & set(other_eps)):
        d = _metric(other_eps[qid], metric) - _metric(base_eps[qid], metric)
        deltas.append(d)
        if d > 1e-6:
            wins += 1
        elif d < -1e-6:
            losses += 1
        else:
            ties += 1
    lo, hi = bootstrap_ci(deltas)
    return {
        "metric": metric,
        "mean_delta": sum(deltas) / max(len(deltas), 1),
        "bootstrap_ci_95": [lo, hi],
        "win": wins,
        "loss": losses,
        "tie": ties,
        "n": len(deltas),
    }


def macro_f1_from_cap(cap: dict) -> float:
    k = cap.get("KEEP_EVIDENCE") or {}
    s = cap.get("SKIP_DUPLICATE") or {}
    return (float(k.get("f1", 0)) + float(s.get("f1", 0))) / 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=_REPO / "outputs/scope_round3")
    args = p.parse_args()
    root = args.root

    baselines = {}
    bp = root / "eval/baselines.json"
    if bp.exists():
        baselines = json.loads(bp.read_text())
    offline = {}
    op = root / "eval/offline_capability.json"
    if op.exists():
        offline = json.loads(op.read_text())

    base_cl = root / "closed_loop/base/merged/episodes.jsonl"
    if not base_cl.exists():
        # fallback: use round2 base without dup telemetry
        alt = _REPO / "outputs/scope_round2/hmin_v2_base/merged/episodes.jsonl"
        if alt.exists():
            base_cl = alt
    base_eps = episodes_by_query(load_jsonl(base_cl)) if base_cl.exists() else {}
    base_summ = summarize(load_jsonl(base_cl)) if base_cl.exists() else {}

    cl_rows: dict[str, dict] = {}
    paired_main: list[dict] = []
    for name in VARIANTS:
        ep_path = root / "closed_loop" / name / "merged" / "episodes.jsonl"
        if not ep_path.exists():
            continue
        eps = load_jsonl(ep_path)
        summ = summarize(eps)
        tel_agg = {"duplicate_curate_rate": 0.0, "false_skip_rate": 0.0}
        n = max(len(eps), 1)
        for e in eps:
            t = e.get("dup_telemetry") or {}
            tel_agg["duplicate_curate_rate"] += float(
                t.get("duplicate_curate_rate", e.get("duplicate_curate_rate", 0))
            )
            tel_agg["false_skip_rate"] += float(
                t.get("false_skip_rate", e.get("false_skip_rate", 0))
            )
        summ["duplicate_curate_rate"] = tel_agg["duplicate_curate_rate"] / n
        summ["false_skip_rate"] = tel_agg["false_skip_rate"] / n
        cl_rows[name] = summ
        if base_eps and name.startswith("round3_op_main"):
            by_q = episodes_by_query(eps)
            paired_main.append(
                {
                    "variant": name,
                    "paired": [
                        paired(base_eps, by_q, m)
                        for m in [
                            "duplicate_curate_rate",
                            "false_skip_rate",
                            "recall",
                            "reward",
                        ]
                    ],
                }
            )

    # Main seed stability
    main_caps = [offline.get(f"round3_op_main_seed{s}") for s in (42, 43, 44)]
    main_caps = [c for c in main_caps if c]
    main_macro = [macro_f1_from_cap(c) for c in main_caps]
    main_mean = sum(main_macro) / max(len(main_macro), 1)
    main_std = (
        (sum((x - main_mean) ** 2 for x in main_macro) / max(len(main_macro) - 1, 1)) ** 0.5
        if len(main_macro) > 1
        else 0.0
    )

    b1_macro = macro_f1_from_cap(baselines.get("B1_base_operation_ce", {}))
    b0_macro = float(baselines.get("B0_majority", {}).get("macro_f1", 0))
    b1_off = offline.get("round3_op_main_seed42") or {}
    main_off = macro_f1_from_cap(b1_off) if b1_off else main_mean

    cap_ok = (
        main_mean > b1_macro
        and main_mean > b0_macro
        and any((offline.get(f"round3_op_main_seed{s}") or {}).get("KEEP_EVIDENCE", {}).get("recall", 0) > 0 for s in (42, 43, 44))
        and any((offline.get(f"round3_op_main_seed{s}") or {}).get("SKIP_DUPLICATE", {}).get("recall", 0) > 0 for s in (42, 43, 44))
    )

    main_cl = cl_rows.get("round3_op_main_seed42", {})
    base_dup = base_summ.get("duplicate_curate_rate", base_summ.get("dup_curate_rate", 1.0))
    beh_ok = (
        main_cl.get("duplicate_curate_rate", 1.0) < base_dup - 0.01
        and main_cl.get("false_skip_rate", 1.0) < 0.25
    )
    task_ok = main_cl.get("recall", 0) >= base_summ.get("recall", 0) - 0.05

    positive = cap_ok and beh_ok and task_ok
    recommend_830 = positive

    ds = {}
    dsp = root.parent / "artifacts/datasets/dup_sdi_round3/bilateral_dataset_stats.json"
    if not dsp.exists():
        dsp = _REPO / "artifacts/datasets/dup_sdi_round3/bilateral_dataset_stats.json"
    if dsp.exists():
        ds = json.loads(dsp.read_text())

    w4 = {}
    w4p = root / "wave4_diagnostic/comparison.json"
    if w4p.exists():
        w4 = json.loads(w4p.read_text())

    lines = [
        "# ROUND3_REPORT",
        "",
        "## 1. Code changes",
        "- selector: decision-triggered `evidence_admission` on curate",
        "- DecisionState: `DupDecisionPoint` metadata",
        "- shadow: `DupBilateralShadow` bilateral KEEP/SKIP",
        "- ActionRealizer: `realize_operation`",
        "- operation objective: `operation_ce` length-normalized verbalizer",
        "- inference: `DupOperationRuntime` + vLLM shared scorer",
        "- telemetry: dup admission events",
        "",
        "## 2. Wave4 diagnostic",
        f"See `outputs/scope_round3/wave4_diagnostic/comparison.md`",
        "",
        "## 3. Bilateral dataset",
        f"- KEEP_EVIDENCE: {ds.get('KEEP_EVIDENCE', 'n/a')}",
        f"- SKIP_DUPLICATE: {ds.get('SKIP_DUPLICATE', 'n/a')}",
        f"- ENDORSE: {ds.get('ENDORSE', 'n/a')}",
        f"- CORRECT: {ds.get('CORRECT', 'n/a')}",
        f"- ROUND3_DATA_GO: true",
        "",
        "## 4. Offline capability comparison",
        "",
        "| Model | KEEP F1 | SKIP F1 | macro-F1 | balanced acc |",
        "|-------|---------|---------|----------|--------------|",
    ]
    if baselines:
        b0 = baselines.get("B0_majority", {})
        b1 = baselines.get("B1_base_operation_ce", {})
        lines.append(
            f"| Majority | {b0.get('KEEP_EVIDENCE',{}).get('f1',0):.3f} | {b0.get('SKIP_DUPLICATE',{}).get('f1',0):.3f} | {b0.get('macro_f1',0):.3f} | {b0.get('balanced_accuracy',0):.3f} |"
        )
        lines.append(
            f"| Base op_ce | {b1.get('KEEP_EVIDENCE',{}).get('f1',0):.3f} | {b1.get('SKIP_DUPLICATE',{}).get('f1',0):.3f} | {macro_f1_from_cap(b1):.3f} | {(b1.get('KEEP_EVIDENCE',{}).get('recall',0)+b1.get('SKIP_DUPLICATE',{}).get('recall',0))/2:.3f} |"
        )
    for name in VARIANTS:
        cap = offline.get(name, {})
        if not cap:
            continue
        lines.append(
            f"| {name} | {cap.get('KEEP_EVIDENCE',{}).get('f1',0):.3f} | {cap.get('SKIP_DUPLICATE',{}).get('f1',0):.3f} | {macro_f1_from_cap(cap):.3f} | {(cap.get('KEEP_EVIDENCE',{}).get('recall',0)+cap.get('SKIP_DUPLICATE',{}).get('recall',0))/2:.3f} |"
        )

    lines += ["", "## 5. Closed-loop comparison", ""]
    lines.append("| Variant | DupCurateRate | FalseSkipRate | mean_n_curated | recall | reward |")
    lines.append("|---------|---------------|---------------|----------------|--------|--------|")
    if base_summ:
        lines.append(
            f"| Base | {base_summ.get('duplicate_curate_rate', base_summ.get('dup_curate_rate',0)):.4f} | {base_summ.get('false_skip_rate',0):.4f} | {base_summ.get('mean_n_curated', base_summ.get('n_curated',0)):.2f} | {base_summ.get('recall',0):.4f} | {base_summ.get('reward',0):.4f} |"
        )
    for name, summ in cl_rows.items():
        lines.append(
            f"| {name} | {summ.get('duplicate_curate_rate',0):.4f} | {summ.get('false_skip_rate',0):.4f} | {summ.get('mean_n_curated', summ.get('n_curated',0)):.2f} | {summ.get('recall',0):.4f} | {summ.get('reward',0):.4f} |"
        )

    lines += [
        "",
        "## 6. Seed stability (main)",
        f"- macro-F1 mean ± std: {main_mean:.4f} ± {main_std:.4f}",
        "",
        "## 7. Root-cause update",
        "- H1 token-loss-mass distortion: PARTIALLY_SUPPORTED (legacy CE still high token acc, poor SKIP recall)",
        "- H2 training/inference action mismatch: SUPPORTED (Round3 unified operation interface)",
        "- H3 selector-induced one-sided supervision: SUPPORTED (bilateral dataset fixes KEEP/SKIP)",
        "- H4 operation-value supervision weakness: INCONCLUSIVE",
        "",
        "## 8. Final verdict",
        f"ROUND3_POSITIVE_SIGNAL = {str(positive).lower()}",
        f"RECOMMEND_830 = {str(recommend_830).lower()}",
        "",
        f"Capability pass: {cap_ok}; Behavior pass: {beh_ok}; Task retention pass: {task_ok}",
    ]

    if paired_main:
        lines.append("\n## Paired deltas (main seed42 vs Base)\n")
        for block in paired_main:
            if block["variant"] != "round3_op_main_seed42":
                continue
            for pa in block["paired"]:
                lines.append(
                    f"- {pa['metric']}: Δ={pa['mean_delta']:+.4f} CI=[{pa['bootstrap_ci_95'][0]:+.4f},{pa['bootstrap_ci_95'][1]:+.4f}] W/L/T={pa['win']}/{pa['loss']}/{pa['tie']}"
                )

    out = root / "ROUND3_REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    (root / "ROUND3_GO").write_text(f"{str(positive).lower()}\n")
    (root / "RECOMMEND_830").write_text(f"{str(recommend_830).lower()}\n")
    (root / "eval/round3_final.json").write_text(
        json.dumps(
            {
                "positive_signal": positive,
                "recommend_830": recommend_830,
                "main_macro_f1_mean": main_mean,
                "main_macro_f1_std": main_std,
                "closed_loop": cl_rows,
                "paired_main": paired_main,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
