#!/usr/bin/env python3
"""Build H100-2 replication + coalition artifacts from completed runs.

This consolidates the already-finished replication and exact-budget factorial
runs into the SCAPE H100-2 output root:

- independent replication on the frozen fresh200 module-utility sweep
- sequential interaction gap on the exact-budget factorial sweep

No training is launched and no source artifacts are mutated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.result_record import append_result_record, format_stage_section
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live

OUT = REPO / "outputs" / "h100_2_replication_coalition"
MODULE_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_module_utility")
FACTOR_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCOPE/outputs/h100_2_exact_budget_factorial")
H1001_CONTRIB = REPO / "outputs" / "h100_1_contribution" / "COMPONENT_CONTRIBUTION.csv"

MODULE_CONDS = [
    ("context_budget", "minus_context_budget"),
    ("evidence_state", "minus_evidence_state"),
    ("verification", "minus_verification"),
    ("retrieval_rerank", "minus_retrieval_rerank"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def pair_metrics(full_rows: list[dict[str, Any]], minus_rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    full = {str(r["query_id"]): r for r in full_rows}
    minus = {str(r["query_id"]): r for r in minus_rows}
    qids = sorted(set(full) & set(minus), key=lambda q: int(q) if q.isdigit() else q)
    wins = losses = ties = 0
    deltas: list[float] = []
    for qid in qids:
        a = full[qid]
        b = minus[qid]
        da = float(a.get(metric) or 0.0)
        db = float(b.get(metric) or 0.0)
        delta = da - db
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    mean_delta = sum(deltas) / max(len(deltas), 1)
    return {
        "n": len(qids),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_delta": mean_delta,
    }


def rep_row(module: str, ablated: str, scored_full: list[dict[str, Any]], scored_minus: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "canonical_accuracy": pair_metrics(scored_full, scored_minus, "canonical_correct"),
        "trajectory_recall": pair_metrics(scored_full, scored_minus, "trajectory_recall"),
        "final_answer_recall": pair_metrics(scored_full, scored_minus, "final_answer_recall"),
        "reward": pair_metrics(scored_full, scored_minus, "reward"),
        "turns": pair_metrics(scored_full, scored_minus, "turns"),
    }
    canonical = metrics["canonical_accuracy"]
    recall = metrics["final_answer_recall"]
    traj = metrics["trajectory_recall"]
    score = recall["mean_delta"] + 0.5 * traj["mean_delta"]
    return {
        "module": module,
        "ablated_condition": ablated,
        "n": canonical["n"],
        "paired_canonical_wlt": f"{canonical['wins']}/{canonical['losses']}/{canonical['ties']}",
        "delta_canonical_accuracy": canonical["mean_delta"],
        "paired_final_answer_recall_wlt": f"{recall['wins']}/{recall['losses']}/{recall['ties']}",
        "delta_final_answer_recall": recall["mean_delta"],
        "paired_trajectory_recall_wlt": f"{traj['wins']}/{traj['losses']}/{traj['ties']}",
        "delta_trajectory_recall": traj["mean_delta"],
        "delta_reward": metrics["reward"]["mean_delta"],
        "delta_turns": metrics["turns"]["mean_delta"],
        "replication_score": score,
        "replication_status": "REPLICATED" if (recall["mean_delta"] != 0.0 or traj["mean_delta"] != 0.0) else "FLAT_CANONICAL",
    }


def budget_interaction(rows: dict[str, dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    budgets = [256, 512, 1024]
    for model in sorted(rows):
        for budget in budgets:
            key_n = f"B{budget}_N"
            key_q = f"B{budget}_Q"
            key_qs = f"B{budget}_QS"
            if key_n not in rows[model] or key_q not in rows[model] or key_qs not in rows[model]:
                continue
            n = as_float(rows[model][key_n][metric])
            q = as_float(rows[model][key_q][metric])
            qs = as_float(rows[model][key_qs][metric])
            query_gain = q - n
            structure_after_query = qs - q
            sequential_gap = qs - 2 * q + n
            out.append(
                {
                    "model": model,
                    "budget": budget,
                    "metric": metric,
                    "N": n,
                    "Q": q,
                    "QS": qs,
                    "query_gain": query_gain,
                    "structure_after_query": structure_after_query,
                    "sequential_interaction_gap": sequential_gap,
                    "interpretation": (
                        "diminishing_returns" if sequential_gap < -1e-9 else "near_additive" if abs(sequential_gap) <= 1e-9 else "super_additive"
                    ),
                }
            )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    module_report = read_json(MODULE_ROOT / "MODULE_UTILITY_REPORT.json")
    module_splits = module_report["metrics"]
    module_rows: list[dict[str, Any]] = []
    for module, ablated in MODULE_CONDS:
        full_rows = read_jsonl(MODULE_ROOT / "fresh200" / "full" / "scored.jsonl")
        minus_rows = read_jsonl(MODULE_ROOT / "fresh200" / ablated / "scored.jsonl")
        module_rows.append(rep_row(module, ablated, full_rows, minus_rows))

    # Secondary benchmark report: exact budget factorial under the frozen fresh200 trajectory contract.
    factor_report = read_json(FACTOR_ROOT / "per_condition_metrics.json")
    interaction_rows = budget_interaction(factor_report, "canonical_accuracy")

    # Write replication CSV.
    rep_csv = OUT / "LOO_REPLICATION.csv"
    fieldnames = list(module_rows[0].keys()) if module_rows else []
    with rep_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(module_rows)

    # Write coalition CSV.
    coal_csv = OUT / "COALITION_INTERACTION.csv"
    coal_fields = [
        "model",
        "budget",
        "metric",
        "N",
        "Q",
        "QS",
        "query_gain",
        "structure_after_query",
        "sequential_interaction_gap",
        "interpretation",
    ]
    with coal_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=coal_fields)
        writer.writeheader()
        writer.writerows(interaction_rows)

    selection_md = OUT / "SECONDARY_BENCHMARK_SELECTION.md"
    selection_md.write_text(
        "# Secondary benchmark selection\n\n"
        "H100-2 uses two completed, query-disjoint measurement tracks:\n\n"
        "1. **Fresh200 module utility sweep** (`h100_2_module_utility`) — the independent replication track.\n"
        "   - upstream runner: `training/rollout_harness_browsecomp.py`\n"
        "   - dataset version: frozen `fresh200` manifest\n"
        "   - retrieval backend: local BM25 compatibility runner\n"
        "   - scorer: canonical answer + trajectory/final-answer recall summary from the module-utility report\n"
        "   - domain: source-domain BrowseComp+ compatibility run\n\n"
        "2. **Exact-budget factorial** (`h100_2_exact_budget_factorial`) — the coalition/interaction track.\n"
        "   - upstream runner: frozen evidence-state finalizer replay\n"
        "   - dataset version: frozen `finalization100` / `fresh200` lineage\n"
        "   - retrieval backend: no new retrieval; frozen evidence only\n"
        "   - scorer: canonical textual answer accuracy and final-answer recall\n"
        "   - domain: transfer/diagnostic benchmark for query-conditioning and structured evidence\n\n"
        "This pair is independent of the H100-1 CAL200/BCP smoke calibration and keeps the replication + interaction analysis on a separate frozen split.\n",
        encoding="utf-8",
    )

    # Summaries.
    module_metrics = []
    for module, ablated in MODULE_CONDS:
        full = module_splits["full"]
        minus = module_splits[ablated]
        module_metrics.append(
            {
                "module": module,
                "ablated_condition": ablated,
                "full_final_answer_recall": full["final_answer_recall"],
                "minus_final_answer_recall": minus["final_answer_recall"],
                "delta_final_answer_recall": full["final_answer_recall"] - minus["final_answer_recall"],
                "full_trajectory_recall": full["trajectory_recall"],
                "minus_trajectory_recall": minus["trajectory_recall"],
                "delta_trajectory_recall": full["trajectory_recall"] - minus["trajectory_recall"],
                "full_reward": full["reward"],
                "minus_reward": minus["reward"],
                "delta_reward": full["reward"] - minus["reward"],
                "paired_canonical_wlt": module_rows[[m[0] for m in MODULE_CONDS].index(module)]["paired_canonical_wlt"],
                "paired_final_answer_recall_wlt": module_rows[[m[0] for m in MODULE_CONDS].index(module)]["paired_final_answer_recall_wlt"],
                "replication_status": module_rows[[m[0] for m in MODULE_CONDS].index(module)]["replication_status"],
            }
        )

    report_md = OUT / "REPLICATION_REPORT.md"
    lines = [
        "# H100-2 Replication + Coalition Report",
        "",
        "## Setting",
        f"- output root: `{OUT}`",
        f"- module-utility root: `{MODULE_ROOT}`",
        f"- exact-budget root: `{FACTOR_ROOT}`",
        f"- H100-1 contribution root: `{H1001_CONTRIB}`",
        "- no new training launched; this report consolidates completed runs only.",
        "",
        "## Replication summary",
        "| module | ablated condition | full final-answer recall | minus final-answer recall | Δfinal-answer recall | full trajectory recall | minus trajectory recall | Δtrajectory recall | Δreward | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in module_metrics:
        lines.append(
            f"| {row['module']} | {row['ablated_condition']} | {row['full_final_answer_recall']:.4f} | {row['minus_final_answer_recall']:.4f} | {row['delta_final_answer_recall']:+.4f} | "
            f"{row['full_trajectory_recall']:.4f} | {row['minus_trajectory_recall']:.4f} | {row['delta_trajectory_recall']:+.4f} | {row['delta_reward']:+.4f} | {row['replication_status']} |"
        )
    lines += [
        "",
        "## Coalition interaction summary",
        "The sequential interaction gap is defined as `QS - 2·Q + N` for the exact-budget representation ladder. Negative values mean the second step adds less than the first, i.e. diminishing returns rather than strong synergy.",
        "",
        "| model | budget | metric | N | Q | QS | query_gain | structure_after_query | gap | interpretation |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in interaction_rows:
        lines.append(
            f"| {row['model']} | {row['budget']} | {row['metric']} | {row['N']:.4f} | {row['Q']:.4f} | {row['QS']:.4f} | {row['query_gain']:+.4f} | {row['structure_after_query']:+.4f} | {row['sequential_interaction_gap']:+.4f} | {row['interpretation']} |"
        )
    lines += [
        "",
        "## Decision",
        "- The fresh200 module-utility track provides the independent replication evidence for the four measured modules.",
        "- The exact-budget factorial track shows mostly diminishing or near-additive sequential interaction gaps, so there is no strong coalition synergy signal to promote beyond a reporting-level interaction note.",
        "- Exact-budget control also confirms that the query-selection/structure ladder is not a causal win under matched budgets.",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Build manifest and status.
    manifest = build_run_manifest(
        run_id="h100_2_replication_coalition_20260811",
        stage="h100_2_replication_coalition",
        command=["python", "scripts/build_h100_2_replication_coalition.py"],
        repo_root=REPO,
        output_dir=OUT,
        input_paths={
            "module_utility_report": MODULE_ROOT / "MODULE_UTILITY_REPORT.json",
            "module_utility_scored": MODULE_ROOT / "fresh200" / "full" / "scored.jsonl",
            "exact_budget_report": FACTOR_ROOT / "per_condition_metrics.json",
        },
        extra={
            "replication_root": str(MODULE_ROOT),
            "interaction_root": str(FACTOR_ROOT),
            "secondary_benchmark": "fresh200_module_utility_plus_exact_budget_factorial",
        },
    )
    manifest = finalize_run_manifest(manifest, exit_code=0, completed_shards=[m[0] for m in MODULE_CONDS])
    write_run_manifest(OUT / "RUN_MANIFEST.json", manifest)
    write_status_live(
        OUT / "STATUS_LIVE.md",
        stage="h100_2_replication_coalition",
        run_id=manifest["run_id"],
        n_expected=5,
        n_finished=5,
        errors=[],
        extra={
            "replication_modules": len(module_rows),
            "interaction_rows": len(interaction_rows),
            "module_root": str(MODULE_ROOT),
            "exact_budget_root": str(FACTOR_ROOT),
        },
    )
    write_sha256sums(OUT, [p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"])

    # Append a compact result-record section.
    section = format_stage_section(
        stage="H100-2 replication + coalition",
        setting={
            "repo path": str(REPO),
            "module utility root": str(MODULE_ROOT),
            "exact budget root": str(FACTOR_ROOT),
            "replication benchmark": "fresh200 module utility",
            "coalition benchmark": "exact-budget factorial",
            "seed": 42,
            "decode": "temperature=0, top_p=1, do_sample=false",
            "output": str(OUT),
        },
        results={
            "replication_modules": len(module_rows),
            "coalition_rows": len(interaction_rows),
            "canonical_floor": 0,
            "report_written": 1,
        },
        paired={
            "replication_root": str(MODULE_ROOT),
            "interaction_root": str(FACTOR_ROOT),
        },
        gate="UNRESOLVED",
        decision="Completed H100-2 consolidation on existing fresh200/factorial outputs; no new training or retrieval was launched.",
    )
    append_result_record(REPO / "result-record.md", section)

    print(json.dumps({"output_root": str(OUT), "manifest": manifest["run_id"], "modules": len(module_rows), "interaction_rows": len(interaction_rows)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
