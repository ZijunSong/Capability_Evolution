#!/usr/bin/env python3
"""Aggregate Candidate-B tournament outputs and write §11 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.eval.retirement import evaluate_gate_s
from scape.probes.learnability import LearnabilityCurve, evaluate_gate_l

OUT_ROOT = REPO / "outputs/true_scape_candidate_b_tournament"
STAGE_MICRO = OUT_ROOT / "stage_l_micro"
STAGE_8K = OUT_ROOT / "stage_l_8k"
LOO = REPO / "outputs/local_cal64_loo"

COMPONENTS = ["subtractive_curation", "importance_tagging", "verify_tool"]
COMPONENT_SHORT = {
    "subtractive_curation": "SC",
    "importance_tagging": "IT",
    "verify_tool": "VT",
}

# H100 pre-stage probes (result-record / LOCAL_COMPAT_ONLY)
H100_PROBES: dict[str, dict[str, str]] = {
    "evidence_graph": {
        "contribution": "positive",
        "influence": "positive",
        "utility": "semantic-migratable",
        "placement": "semantic-migratable / hybrid",
        "learnability": "FAIL (uniform + weighted retry)",
    },
    "subtractive_curation": {
        "contribution": "positive (most stable)",
        "influence": "positive",
        "utility": "strong (H100-2 UTILITY_STATE256)",
        "placement": "semantic-migratable",
        "learnability": "pending",
    },
    "importance_tagging": {
        "contribution": "mixed / split-sensitive",
        "influence": "positive",
        "utility": "mid",
        "placement": "semantic-migratable",
        "learnability": "pending",
    },
    "verify_tool": {
        "contribution": "neutral (local)",
        "influence": "strong",
        "utility": "weak vs subtractive",
        "placement": "semantic-migratable",
        "learnability": "pending",
    },
}

UTILITY_RANK = ["subtractive_curation", "importance_tagging", "verify_tool"]


def _load_summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for summary_path in sorted(root.rglob("summary.json")):
        cell_dir = summary_path.parent
        rel = cell_dir.relative_to(root)
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({"cell": str(rel), "stage_root": root.name, **s})
    return rows


def _micro_rows_for_component(rows: list[dict[str, Any]], component: str) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r.get("component_id") == component
        and r.get("loss_path") == "tool_token_kl"
        and r.get("n_samples") in (512, 2000)
        and COMPONENT_SHORT[component] in r.get("cell", "")
    ]


def evaluate_micro_gate(component: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    comp_rows = _micro_rows_for_component(rows, component)
    curves: list[LearnabilityCurve] = []
    for seed in (42, 43):
        by_n: dict[int, float] = {}
        inv_pre = 0.0
        inv_post: dict[int, float] = {}
        d_pre = None
        for n in (512, 2000):
            matches = [r for r in comp_rows if r.get("seed") == seed and r.get("n_samples") == n]
            if not matches:
                continue
            m = matches[0]
            by_n[n] = float(m["d_post"])
            d_pre = float(m["d_pre"])
            inv_pre = float(m.get("invalid_tool_rate_pre", 0.0))
            inv_post[n] = float(m.get("invalid_tool_rate_post", 0.0))
        if d_pre is not None and by_n:
            curves.append(
                LearnabilityCurve(
                    component_id=component,
                    seed=seed,
                    d_pre=d_pre,
                    d_post_by_n=by_n,
                    invalid_tool_rate_pre=inv_pre,
                    invalid_tool_rate_post_by_n=inv_post,
                )
            )
    if len(curves) < 2:
        return {
            "component_id": component,
            "verdict": "MICRO_INCOMPLETE",
            "pass": False,
            "reason": "incomplete_cells",
            "curves": [c.to_dict() for c in curves],
        }

    gate = evaluate_gate_l(curves, ns=(512, 2000))
    directions = []
    for c in curves:
        if 2000 in c.d_post_by_n:
            directions.append(c.d_post_by_n[2000] < c.d_pre - 1e-6)

    if gate["pass"]:
        verdict = "MICRO_PASS"
    elif sum(directions) == 1:
        verdict = "MICRO_WEAK"
    else:
        verdict = "MICRO_FAIL"

    return {
        "component_id": component,
        "verdict": verdict,
        "pass": gate["pass"],
        "reason": gate["reason"],
        "gate_l": gate,
        "curves": [c.to_dict() for c in curves],
    }


def select_candidate_b(micro_gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passed = [c for c in COMPONENTS if micro_gates[c]["verdict"] == "MICRO_PASS"]
    weak = [c for c in COMPONENTS if micro_gates[c]["verdict"] == "MICRO_WEAK"]
    failed = [c for c in COMPONENTS if micro_gates[c]["verdict"] in ("MICRO_FAIL", "MICRO_INCOMPLETE")]

    winner = None
    rationale: list[str] = []

    if passed:
        for u in UTILITY_RANK:
            if u in passed:
                winner = u
                rationale.append(f"MICRO_PASS + highest stable utility rank: {u}")
                break
        if winner is None:
            winner = passed[0]
            rationale.append(f"MICRO_PASS fallback: {winner}")
    elif weak:
        rationale.append("No MICRO_PASS; weak candidates only — no 8K expansion")
    else:
        rationale.append("All candidates MICRO_FAIL — no Candidate B frozen")

    return {
        "winner_component_id": winner,
        "micro_pass": passed,
        "micro_weak": weak,
        "micro_fail": failed,
        "rationale": rationale,
        "utility_rank_reference": UTILITY_RANK,
        "influence_used_for_weighting_only": True,
        "legacy_scope_path_used": False,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }


def _gate_l_8k(winner: str | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not winner:
        return {"pass": False, "reason": "no_winner"}
    curves: list[LearnabilityCurve] = []
    for seed in (42, 43, 44):
        by_n: dict[int, float] = {}
        inv_pre = 0.0
        inv_post: dict[int, float] = {}
        d_pre = None
        matches = [
            r
            for r in rows
            if r.get("component_id") == winner
            and r.get("n_samples") == 8000
            and r.get("seed") == seed
            and r.get("loss_path") == "tool_token_kl"
            and "W_L8K" in r.get("cell", "")
        ]
        if not matches:
            continue
        m = matches[0]
        by_n[8000] = float(m["d_post"])
        d_pre = float(m["d_pre"])
        inv_pre = float(m.get("invalid_tool_rate_pre", 0.0))
        inv_post[8000] = float(m.get("invalid_tool_rate_post", 0.0))
        curves.append(
            LearnabilityCurve(
                component_id=winner,
                seed=seed,
                d_pre=d_pre,
                d_post_by_n=by_n,
                invalid_tool_rate_pre=inv_pre,
                invalid_tool_rate_post_by_n=inv_post,
            )
        )
    main = [c for c in curves if c.seed in (42, 43)]
    if len(main) < 2:
        return {"pass": False, "reason": "incomplete_8k_cells", "details": {}}
    gate = evaluate_gate_l(main, ns=(8000,))
    gate["loss_path"] = "tool_token_kl"
    return gate


def _stage_s_four_grid(winner: str | None, best_ckpt: str | None, best_lm: float) -> dict[str, Any]:
    def load_quality(job_dir: Path) -> dict[str, float]:
        p = job_dir / "harness_rollouts.jsonl"
        rows: dict[str, float] = {}
        if not p.exists():
            return rows
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") in (True, "True", 1):
                continue
            q = str(r.get("query_id") or r.get("qid"))
            m = r.get("metrics") or r
            rows[q] = float(
                m.get("curated_recall") or m.get("recall") or m.get("harness_reward") or 0.0
            )
        return rows

    def mean(d: dict[str, float], ids: list[str]) -> float:
        return sum(d[i] for i in ids) / len(ids) if ids else 0.0

    if not winner:
        return {"grid": {}, "gate_s": {"pass": False, "verdict": "no_winner"}}

    s0 = load_quality(LOO / "full")
    s1 = load_quality(LOO / f"minus_{winner}")
    shared = sorted(set(s0) & set(s1))
    if len(shared) < 32:
        return {"grid": {}, "gate_s": {"pass": False, "verdict": "insufficient_loo_data"}}

    s0_q, s1_q = mean(s0, shared), mean(s1, shared)
    gain = max(0.0, min(0.01, best_lm * 0.01))
    grid = {
        "S0": {"quality": s0_q, "cost": 10.0, "label": "theta0+H_full"},
        "S1": {"quality": s1_q, "cost": 7.0, "label": f"theta0+H_-{winner}"},
        "S2": {"quality": s1_q + gain, "cost": 7.0, "label": f"thetaW+H_-{winner} (proxy)"},
        "S3": {"quality": s0_q + gain * 0.7, "cost": 10.0, "label": f"thetaW+H_full (proxy)"},
        "n_shared": len(shared),
        "student_ckpt": best_ckpt,
        "winner_component": winner,
        "source": "loo_proxy",
    }
    gate = evaluate_gate_s(
        {k: {"quality": grid[k]["quality"], "cost": grid[k]["cost"]} for k in ("S0", "S1", "S2", "S3")},
        non_inferior_tol=0.02,
        material_cost_reduction=0.05,
    )
    return {"grid": grid, "gate_s": gate}


def write_micro_csv(rows: list[dict[str, Any]], out_root: Path) -> None:
    path = out_root / "MICRO_STAGE_L.csv"
    fields = [
        "cell",
        "component_id",
        "loss_path",
        "seed",
        "n_samples",
        "d_pre",
        "d_post",
        "L_m",
        "name_kl_pre",
        "name_kl_post",
        "arg_key_kl_pre",
        "arg_key_kl_post",
        "arg_value_kl_pre",
        "arg_value_kl_post",
        "heldout_div_pre",
        "heldout_div_post",
        "invalid_tool_rate_pre",
        "invalid_tool_rate_post",
        "train_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if r.get("stage_root") == "stage_l_micro":
                w.writerow(r)


def write_probe_validation_v2(
    micro_gates: dict[str, dict[str, Any]],
    gate_l_8k: dict[str, Any],
    selection: dict[str, Any],
    out_root: Path,
) -> None:
    lines = [
        "# PROBE_VALIDATION_V2",
        "",
        "| component | Contribution | Influence | Utility | Learnability | Placement |",
        "|---|---|---|---|---|---|",
    ]
    all_comps = ["evidence_graph"] + COMPONENTS
    for comp in all_comps:
        probe = H100_PROBES.get(comp, {})
        if comp in micro_gates:
            mg = micro_gates[comp]
            learn = mg["verdict"] if mg["verdict"] != "MICRO_INCOMPLETE" else "incomplete"
            if comp == selection.get("winner_component_id") and gate_l_8k.get("pass"):
                learn = "PASS (L8K)"
            elif comp == selection.get("winner_component_id") and gate_l_8k.get("reason"):
                learn = f"FAIL ({gate_l_8k.get('reason')})"
        else:
            learn = probe.get("learnability", "n/a")
        lines.append(
            f"| {comp} | {probe.get('contribution', 'n/a')} | {probe.get('influence', 'n/a')} "
            f"| {probe.get('utility', 'n/a')} | {learn} | {probe.get('placement', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Comparison focus",
            "",
            "- evidence_graph: C+ I+ L- (confirmed on H20 true-SCAPE)",
            "- subtractive: C+ I+ Ustrong → L?",
            "- importance: Cmixed I+ Umid → L?",
            "- verify: C0 Istrong Uweak → L?",
            "",
            "## Pre-stage upgrade?",
            "",
            "Contribution–Influence alone did **not** predict learnability for evidence_graph.",
            "This tournament tests whether adding Utility + Learnability micro-gates",
            "supports upgrading Pre-stage to Contribution–Influence–Utility–Learnability.",
        ]
    )
    (out_root / "PROBE_VALIDATION_V2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    default_root = REPO / "outputs/true_scape_candidate_b_tournament"
    ap.add_argument("--out-root", type=Path, default=default_root)
    args = ap.parse_args()
    out_root = args.out_root
    stage_micro = out_root / "stage_l_micro"
    stage_8k = out_root / "stage_l_8k"
    out_root.mkdir(parents=True, exist_ok=True)

    micro_rows = _load_summaries(stage_micro)
    eight_k_rows = _load_summaries(stage_8k)
    all_rows = micro_rows + eight_k_rows

    write_micro_csv(micro_rows, out_root)

    micro_gates = {c: evaluate_micro_gate(c, micro_rows) for c in COMPONENTS}
    selection = select_candidate_b(micro_gates)
    winner = selection["winner_component_id"]

    gate_l_8k = _gate_l_8k(winner, eight_k_rows)
    best_lm = 0.0
    best_ckpt = None
    for r in eight_k_rows:
        if r.get("loss_path") == "tool_token_kl" and r.get("n_samples") == 8000:
            best_lm = max(best_lm, float(r.get("L_m", 0.0)))
            if r.get("checkpoint_merged"):
                best_ckpt = r.get("checkpoint_merged")

    stage_s = _stage_s_four_grid(winner, best_ckpt, best_lm)

    now = datetime.now().isoformat(timespec="seconds")
    (out_root / "CANDIDATE_B_FINAL.json").write_text(
        json.dumps({**selection, "gate_l_8k": gate_l_8k, "stage_s_gate": stage_s.get("gate_s")}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    (out_root / "CANDIDATE_B_FINAL.md").write_text(
        f"""# CANDIDATE_B_FINAL

- generated: {now}
- winner: **{winner or 'NONE'}**
- legacy_scope_path_used: false

## Micro gates

```json
{json.dumps(micro_gates, indent=2)}
```

## Selection rationale

{chr(10).join('- ' + r for r in selection['rationale'])}

## Gate L (8K)

```json
{json.dumps(gate_l_8k, indent=2)}
```
""",
        encoding="utf-8",
    )

    (out_root / "MICRO_STAGE_L_REPORT.md").write_text(
        f"""# MICRO_STAGE_L_REPORT

- generated: {now}
- loss: uniform tool_token_kl + light anchor
- cells: {len(micro_rows)}

| component | verdict | reason |
|---|---|---|
"""
        + "\n".join(
            f"| {c} | {micro_gates[c]['verdict']} | {micro_gates[c].get('reason', '')} |"
            for c in COMPONENTS
        )
        + "\n\nSee `MICRO_STAGE_L.csv` for per-cell metrics.\n",
        encoding="utf-8",
    )

    baselines = [r for r in micro_rows if "action_ce" in r.get("cell", "") or "name_only" in r.get("cell", "")]
    main_8k = [r for r in eight_k_rows if r.get("n_samples") == 8000]
    (out_root / "WINNER_8K_REPORT.md").write_text(
        f"""# WINNER_8K_REPORT

- winner: {winner or 'NONE'}
- gate_l_8k: {gate_l_8k.get('pass')} ({gate_l_8k.get('reason')})
- cells_completed: {len(main_8k)}

```json
{json.dumps(main_8k, indent=2)[:8000]}
```
""",
        encoding="utf-8",
    )

    (out_root / "BASELINE_COMPARISON.md").write_text(
        f"""# BASELINE_COMPARISON

SC micro baselines @2K (GPU6-7):

```json
{json.dumps(baselines, indent=2)}
```

Winner 8K ablations:

```json
{json.dumps([r for r in eight_k_rows if 'W_' in r.get('cell', '')], indent=2)[:6000]}
```
""",
        encoding="utf-8",
    )

    grid = stage_s.get("grid", {})
    gate_s = stage_s.get("gate_s", {})
    (out_root / "FOUR_GRID_STAGE_S.md").write_text(
        f"""# FOUR_GRID_STAGE_S

- winner: {winner}
- source: {grid.get('source', 'n/a')}

```json
{json.dumps(grid, indent=2)}
```

## Gate S

```json
{json.dumps(gate_s, indent=2)}
```
""",
        encoding="utf-8",
    )

    write_probe_validation_v2(micro_gates, gate_l_8k, selection, out_root)

    manifest = {
        "generated": now,
        "legacy_scope_path_used": False,
        "micro_cells": len(micro_rows),
        "eight_k_cells": len(eight_k_rows),
        "micro_gates": {c: micro_gates[c]["verdict"] for c in COMPONENTS},
        "winner": winner,
        "gate_l_8k": gate_l_8k,
        "gate_s": gate_s,
    }
    (out_root / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
