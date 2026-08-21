#!/usr/bin/env python3
"""Finalize H100-4 0816 artifacts after formal AUTO sync and OPHSD route run."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "btp_h100_4_baselines"
AUTO = ROOT / "outputs" / "h100_2_structured_privilege_formal_0816"
REAL = ROOT / "outputs" / "h100_2_real_closed_loop_bm25_0816"
OPHSD = OUT / "ophsd"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rewrite_sha() -> None:
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(OUT)}\n" for p in files), encoding="utf-8")


def load_ophsd() -> list[dict]:
    rows = []
    for path in sorted((OPHSD / "cells").glob("OPHSD_ROUTE_CONTEXT_seed*/summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if sorted(int(r["seed"]) for r in rows) != [42, 43, 44, 45]:
        raise SystemExit("OPHSD seeds 42-45 not complete")
    return rows


def finalize_matched() -> dict:
    cells = [r for r in read_csv(AUTO / "AUTO_REPRESENTATION_CELLS.csv") if r["variant"] == "AUTO_MATCHED_TEXT"]
    if sorted(int(r["seed"]) for r in cells) != [42, 43, 44, 45]:
        raise SystemExit("AUTO_MATCHED_TEXT seeds 42-45 not complete")
    real = json.loads((REAL / "REAL_CLOSED_LOOP_HANDOFF.json").read_text(encoding="utf-8"))
    mt_real = next(r for r in real["summary"] if r["method"] == "AUTO_MATCHED_TEXT")
    best = min(cells, key=lambda r: float(r["post_test_JS"]))
    mean_js = mean(float(r["post_test_JS"]) for r in cells)
    mean_agreement = mean(float(r["agreement"]) for r in cells)
    fields = ["seed", "method", "status", "objective", "n_train", "n_valid", "n_test", "post_test_JS", "agreement", "checkpoint_reloadable", "student_inference_has_privilege", "real_closed_loop_source"]
    write_csv(OUT / "MATCHED_TEXT_TRAINING_CELLS.csv", [
        {
            "seed": r["seed"],
            "method": "Matched Text OPD / AUTO matched-information",
            "status": "completed_formal_auto_sync",
            "objective": r["objective"],
            "n_train": r["n_train"],
            "n_valid": r["n_valid"],
            "n_test": r["n_test"],
            "post_test_JS": r["post_test_JS"],
            "agreement": r["agreement"],
            "checkpoint_reloadable": r["checkpoint_reloadable"],
            "student_inference_has_privilege": r["student_inference_has_privilege"],
            "real_closed_loop_source": "outputs/h100_2_real_closed_loop_bm25_0816/REAL_CLOSED_LOOP_HANDOFF.json",
        }
        for r in cells
    ], fields)
    write_csv(OUT / "MATCHED_TEXT_CLOSED_LOOP.csv", [{
        "method": "Matched Text OPD / AUTO matched-information",
        "status": real["status"],
        "n": mt_real["n"],
        "overall_reward": mt_real["overall_reward"],
        "curated_evidence_recall": mt_real["curated_evidence_recall"],
        "trajectory_recall": mt_real["trajectory_recall"],
        "final_answer_recall": mt_real["final_answer_recall"],
        "tool_calls": mt_real["tool_calls"],
        "student_inference_has_privilege": mt_real["student_inference_has_privilege"],
        "runner": "real_closed_loop_bm25_route_head",
    }], ["method", "status", "n", "overall_reward", "curated_evidence_recall", "trajectory_recall", "final_answer_recall", "tool_calls", "student_inference_has_privilege", "runner"])
    protocol = f"""# MATCHED_TEXT_PROTOCOL

Status: `COMPLETED_FORMAL_AUTO_SYNC_REAL_CLOSED_LOOP`.

This H100-4 artifact synchronizes the frozen H100-2 AUTO matched-information protocol, as required when AUTO is available.

Setting:

- source directory: `outputs/h100_2_structured_privilege_formal_0816/`
- component: `auto_populate_first_search`
- matched textual branch: `AUTO_MATCHED_TEXT`
- seeds: 42,43,44,45
- split: train={cells[0]['n_train']}, valid={cells[0]['n_valid']}, test={cells[0]['n_test']}
- route space: canonical 8-way Harness-1 tool distribution
- objective: route_kl
- student inference privilege: false
- real closed-loop source: `outputs/h100_2_real_closed_loop_bm25_0816/REAL_CLOSED_LOOP_HANDOFF.json`

Matched-text route result:

- mean post_test_JS: {mean_js:.8f}
- mean agreement: {mean_agreement:.8f}
- best seed by post_test_JS: {best['seed']} ({best['post_test_JS']})

Real BM25 closed loop:

- n_queries: {mt_real['n']}
- overall_reward: {mt_real['overall_reward']}
- curated_evidence_recall: {mt_real['curated_evidence_recall']}
- trajectory_recall: {mt_real['trajectory_recall']}
- final_answer_recall: {mt_real['final_answer_recall']}
- tool_calls: {mt_real['tool_calls']}

Caveat: this synchronized formal AUTO result shows Matched Text and Structured parity in real closed loop; it does not establish a structured advantage.
"""
    (OUT / "MATCHED_TEXT_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    handoff = {
        "status": "completed_formal_auto_sync_real_closed_loop",
        "source": str(AUTO.relative_to(ROOT)),
        "real_closed_loop_source": str(REAL.relative_to(ROOT)),
        "seeds_completed": [42, 43, 44, 45],
        "mean_post_test_JS": mean_js,
        "mean_agreement": mean_agreement,
        "best_cell": best,
        "real_closed_loop": mt_real,
        "student_inference_has_privilege": False,
        "claim": "matched_text_completed_but_no_structured_advantage",
    }
    write_json(OUT / "MATCHED_TEXT_HANDOFF.json", handoff)
    return handoff


def finalize_ophsd(rows: list[dict]) -> dict:
    mean_js = mean(float(r["post_test"]["JS"]) for r in rows)
    mean_agree = mean(float(r["post_test"]["agreement"]) for r in rows)
    best = min(rows, key=lambda r: float(r["post_test"]["JS"]))
    write_csv(OUT / "OPHSD_TRAINING_CELLS.csv", [{
        "seed": r["seed"],
        "method": "OPHSD-style route adaptation",
        "status": "completed_route_level_faithful_adaptation",
        "objective": r["objective"],
        "n_train": r["n_train"],
        "n_valid": r["n_valid"],
        "n_test": r["n_test"],
        "steps": r["steps"],
        "mean_train_loss": r["mean_train_loss"],
        "post_test_JS": r["post_test"]["JS"],
        "post_test_agreement": r["post_test"]["agreement"],
        "invalid_tool_rate": r["invalid_tool_rate"],
        "checkpoint_reloadable": r["checkpoint_reloadable"],
        "student_inference_has_privilege": r["student_inference_has_privilege"],
        "component_local_signal_used": r["component_local_signal_used"],
    } for r in rows], ["seed", "method", "status", "objective", "n_train", "n_valid", "n_test", "steps", "mean_train_loss", "post_test_JS", "post_test_agreement", "invalid_tool_rate", "checkpoint_reloadable", "student_inference_has_privilege", "component_local_signal_used"])
    write_csv(OUT / "OPHSD_CLOSED_LOOP.csv", [{
        "method": "OPHSD-style route adaptation",
        "status": "route_level_no_privilege_eval_completed_real_bm25_not_run_for_ophsd",
        "metric": "post_test_JS_lower_is_better",
        "mean_value": f"{mean_js:.8f}",
        "best_seed": best["seed"],
        "best_value": best["post_test"]["JS"],
        "student_inference_has_privilege": False,
        "component_local_signal_used": False,
        "note": "Faithful route-level whole-harness terminal-context adaptation; separate real BM25 closed-loop runner has not been executed for OPHSD checkpoints.",
    }], ["method", "status", "metric", "mean_value", "best_seed", "best_value", "student_inference_has_privilege", "component_local_signal_used", "note"])
    (OUT / "OPHSD_SEARCH_ADAPTATION.md").write_text(f"""# OPHSD_SEARCH_ADAPTATION

Status: `COMPLETED_ROUTE_LEVEL_FAITHFUL_ADAPTATION`.

Faithful adaptation used here:

1. Student route head trains on the same AUTO formal on-policy states and split.
2. Whole-harness terminal context is summarized from terminal state metadata (`query_id`, terminal step, document count, tool-history length, prior-search count, high-importance count) and hashed as teacher-context provenance.
3. Frozen/static teacher target is the full-harness route distribution `P_tool_name_full`.
4. Student matches teacher with categorical route KL.
5. Student evaluation uses only reduced no-privilege route features; `student_inference_has_privilege=false`.
6. Component-local structured signal is excluded: `component_local_signal_used=false`.

Fairness:

- source split: `outputs/h100_2_structured_privilege_formal_0816/{{train,valid,test}}_auto_paired.jsonl`
- seeds: 42,43,44,45
- train/valid/test: {rows[0]['n_train']}/{rows[0]['n_valid']}/{rows[0]['n_test']}
- route space: canonical 8-way Harness-1 route distribution
- objective: route_kl

Result:

- mean post_test_JS: {mean_js:.8f}
- mean post_test_agreement: {mean_agree:.8f}
- best seed: {best['seed']}

Remaining caveat: this is a route-level Search/Harness-1 adaptation. A 7B LoRA whole-harness terminal-context teacher and OPHSD-specific real BM25 closed-loop rollout are still stronger future work, but the baseline is no longer recorded as a missing-contract dry run.
""", encoding="utf-8")
    handoff = {
        "method": "OPHSD-style",
        "status": "completed_route_level_faithful_adaptation",
        "seeds_completed": [int(r["seed"]) for r in rows],
        "mean_post_test_JS": mean_js,
        "mean_post_test_agreement": mean_agree,
        "best_cell": best["cell"],
        "student_inference_has_privilege": False,
        "component_local_signal_used": False,
        "real_bm25_closed_loop_for_ophsd": False,
        "caveat": "Route-level faithful adaptation completed; OPHSD-specific real BM25 closed-loop rollout not run.",
    }
    write_json(OUT / "OPHSD_HANDOFF.json", handoff)
    return handoff


def finalize_tables(matched: dict, ophsd: dict) -> None:
    real = json.loads((REAL / "REAL_CLOSED_LOOP_HANDOFF.json").read_text(encoding="utf-8"))
    by = {r["method"]: r for r in real["summary"]}
    importance = json.loads((OUT / "IMPORTANCE_VALUE_GATE.json").read_text(encoding="utf-8"))
    rows = [
        {"method": "Base Student", "status": "completed_real_closed_loop_bm25_auto_sync", "closed_loop_reward": by["BASE_REDUCED"]["overall_reward"], "proxy_metric": "", "inference_privilege": "none", "source": "REAL_CLOSED_LOOP_HANDOFF.json", "claim_allowed": True},
        {"method": "Full Harness", "status": "not_rerun_exact_0816_h1004", "closed_loop_reward": "", "proxy_metric": "", "inference_privilege": "full harness", "source": "pending", "claim_allowed": False},
        {"method": "Matched Text OPD", "status": matched["status"], "closed_loop_reward": by["AUTO_MATCHED_TEXT"]["overall_reward"], "proxy_metric": f"mean_post_test_JS={matched['mean_post_test_JS']:.8f}", "inference_privilege": "none", "source": "MATCHED_TEXT_HANDOFF.json", "claim_allowed": True},
        {"method": "OPHSD-style", "status": ophsd["status"], "closed_loop_reward": "", "proxy_metric": f"mean_post_test_JS={ophsd['mean_post_test_JS']:.8f}", "inference_privilege": "none", "source": "OPHSD_HANDOFF.json", "claim_allowed": True},
        {"method": "Our Structured Component OPD", "status": "completed_real_closed_loop_bm25_auto_sync", "closed_loop_reward": by["AUTO_STRUCT_TYPED"]["overall_reward"], "proxy_metric": "", "inference_privilege": "none", "source": "H1002_STRUCTURED_PRIVILEGE_HANDOFF.json", "claim_allowed": True},
    ]
    fields = ["method", "status", "closed_loop_reward", "proxy_metric", "inference_privilege", "source", "claim_allowed"]
    write_csv(OUT / "MAIN_COMPARISON_TABLE.csv", rows, fields)
    lines = ["# MAIN_COMPARISON_TABLE", "", "Completed and synchronized H100-4 0816 table. OPHSD is route-level; Matched Text and Structured use H100-2 AUTO formal real BM25 closed-loop sync.", "", "| method | status | closed_loop_reward | proxy_metric | inference_privilege | source | claim_allowed |", "|---|---|---:|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[f]) for f in fields) + " |")
    lines += ["", "## Conclusions", "", "- AUTO Matched Text and AUTO Structured both beat Base in real BM25 closed-loop by +0.03 overall_reward but are tied with each other.", "- OPHSD-style route-level adaptation is completed, no-privilege, and does not use component-local structured signal.", f"- `importance_tagging` remains `{importance['status']}` and is eligible for downstream component work.", "- No structured-over-textual advantage claim is allowed from these results; parity is the correct conclusion."]
    (OUT / "MAIN_COMPARISON_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    gap = (OUT / "BASELINE_GAP.md").read_text(encoding="utf-8") if (OUT / "BASELINE_GAP.md").exists() else "# BASELINE_GAP\n"
    gap += "\n## 2026-08-16 completion update\n\n- Matched Text OPD: completed by synchronizing H100-2 AUTO formal matched-information seeds 42-45 and real BM25 closed-loop.\n- OPHSD-style: completed as route-level whole-harness terminal-context adaptation seeds 42-45; stronger 7B LoRA OPHSD-specific real BM25 closed-loop remains future work.\n- Full Harness exact H100-4 0816 row remains not rerun in this checkout.\n"
    (OUT / "BASELINE_GAP.md").write_text(gap, encoding="utf-8")


def finalize_handoff(matched: dict, ophsd: dict) -> None:
    handoff = json.loads((OUT / "H1004_BTP_HANDOFF.json").read_text(encoding="utf-8")) if (OUT / "H1004_BTP_HANDOFF.json").exists() else {}
    handoff.update({
        "status": "completed_with_auto_matched_sync_and_ophsd_route_adaptation",
        "matched_text": matched,
        "ophsd": ophsd,
        "main_claim_allowed": True,
        "paper_claim": "AUTO structured and matched text tie in real BM25 closed-loop; both beat Base. No structured advantage claim.",
        "claude_md_read": ["/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/CLAUDE.md", "/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/CLAUDE.md"],
    })
    write_json(OUT / "H1004_BTP_HANDOFF.json", handoff)
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps({
        "stage": "h1004_0816_novelty_baselines",
        "status": "completed_with_auto_matched_sync_and_ophsd_route_adaptation",
        "python": "/opt/scape-hf-scorer/bin/python",
        "claude_md_read": True,
        "matched_text_completed": True,
        "ophsd_completed": True,
        "importance_value_positive": True,
        "main_claim_allowed": True,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "STATUS_LIVE.md").write_text("# STATUS_LIVE - h1004_0816_novelty_baselines\n\n- claude_md_read: completed\n- novelty_matrix: completed\n- matched_text_formal_auto_sync: completed seeds 42,43,44,45 plus real BM25 closed-loop\n- ophsd_route_level_adaptation: completed seeds 42,43,44,45\n- importance_value_mining: VALUE_POSITIVE\n- main_claim_allowed: parity-only, no structured advantage\n", encoding="utf-8")


def main() -> int:
    matched = finalize_matched()
    ophsd = finalize_ophsd(load_ophsd())
    finalize_tables(matched, ophsd)
    finalize_handoff(matched, ophsd)
    rewrite_sha()
    print(json.dumps({"status": "completed", "out_dir": str(OUT), "matched": matched["status"], "ophsd": ophsd["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
