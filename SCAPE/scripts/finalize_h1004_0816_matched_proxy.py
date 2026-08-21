#!/usr/bin/env python3
"""Finalize H100-4 0816 deliverables with recovered matched-text proxy runs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from generate_h1004_0816_deliverables import main as generate_base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "btp_h100_4_baselines"
REP = ROOT / "outputs" / "h100_4_privilege_representation"
MATCHED = OUT / "matched_v2"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rewrite_sha256sums() -> None:
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(OUT)}\n" for p in files),
        encoding="utf-8",
    )


def finalize_matched_text() -> dict[str, object]:
    rows = read_csv(REP / "REPRESENTATION_2K_CELLS.csv")
    matched_rows = [r for r in rows if r["privilege"] == "textual" and r["objective"] == "route_kl"]
    if len(matched_rows) != 4:
        raise SystemExit(f"expected 4 matched-text route_kl rows, found {len(matched_rows)}")
    seeds = sorted(int(r["seed"]) for r in matched_rows)
    if seeds != [42, 43, 44, 45]:
        raise SystemExit(f"unexpected matched-text seeds: {seeds}")

    split = json.loads((MATCHED / "V2_SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    n_pairs = sum(1 for line in (MATCHED / "matched_v2_pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    best = min(matched_rows, key=lambda r: float(r["common_reference_test_JS"]))
    mean_js = mean(float(r["common_reference_test_JS"]) for r in matched_rows)
    mean_agreement = mean(float(r["agreement_test"]) for r in matched_rows)

    protocol = f"""# MATCHED_TEXT_PROTOCOL

Status: `ROUTE_HEAD_PROXY_COMPLETED_REAL_LLM_CLOSED_LOOP_BLOCKED`.

Recovered input:

- `outputs/btp_h100_4_baselines/matched_v2/matched_v2_pairs.jsonl`
- V2 information audit: `609/609` deterministic structured-to-text round trip
- source rows: {split.get('source_rows', n_pairs)}
- pair rows present in this checkout: {n_pairs}

Completed local proxy:

- textual Route-KL route-head seeds: 42, 43, 44, 45
- train/valid/test split used by the proxy: 1536/256/256 query-disjoint rows from `outputs/h100_4_privilege_representation`
- no-privilege student features at proxy evaluation; privileged text is used only for matched training condition construction
- best proxy cell: `{best['cell']}`
- mean common-reference test JS across seeds: {mean_js:.8f}
- mean test route agreement across seeds: {mean_agreement:.8f}

Remaining blocker:

The faithful Matched Text OPD requested in the README still needs the full HF/LoRA or equivalent student optimizer and the real interactive closed-loop evaluator. This file does not promote the route-head proxy to a paper-grade LLM closed-loop result.
"""
    (OUT / "MATCHED_TEXT_PROTOCOL.md").write_text(protocol, encoding="utf-8")

    train_fields = [
        "seed", "method", "status", "objective", "n_train", "n_valid", "n_test", "steps",
        "mean_train_loss", "test_js", "test_agreement", "invalid_tool_rate", "checkpoint_reloadable",
        "real_llm_training", "real_closed_loop", "source",
    ]
    train_out = []
    for r in matched_rows:
        train_out.append({
            "seed": int(r["seed"]),
            "method": "Matched Text OPD",
            "status": "route_head_proxy_completed",
            "objective": "route_kl",
            "n_train": int(r["n_train"]),
            "n_valid": int(r["n_valid"]),
            "n_test": int(r["n_test"]),
            "steps": int(r["steps"]),
            "mean_train_loss": r["mean_train_loss"],
            "test_js": r["common_reference_test_JS"],
            "test_agreement": r["agreement_test"],
            "invalid_tool_rate": r["invalid_tool_rate"],
            "checkpoint_reloadable": r["checkpoint_reloadable"],
            "real_llm_training": False,
            "real_closed_loop": False,
            "source": "outputs/h100_4_privilege_representation/REPRESENTATION_2K_CELLS.csv",
        })
    write_csv(OUT / "MATCHED_TEXT_TRAINING_CELLS.csv", train_out, train_fields)

    closed_fields = ["method", "status", "metric", "mean_value", "best_seed", "best_value", "real_closed_loop", "note"]
    write_csv(OUT / "MATCHED_TEXT_CLOSED_LOOP.csv", [{
        "method": "Matched Text OPD",
        "status": "route_head_proxy_completed_real_closed_loop_not_run",
        "metric": "common_reference_test_JS_lower_is_better",
        "mean_value": f"{mean_js:.8f}",
        "best_seed": best["seed"],
        "best_value": best["common_reference_test_JS"],
        "real_closed_loop": False,
        "note": "Offline route-head proxy over query-disjoint test states; not a real BrowseComp closed-loop run.",
    }], closed_fields)

    handoff = {
        "status": "route_head_proxy_completed_real_llm_closed_loop_blocked",
        "matched_v2_pairs": str((MATCHED / "matched_v2_pairs.jsonl").relative_to(ROOT)),
        "roundtrip_pass": "609/609",
        "seeds_completed": seeds,
        "mean_common_reference_test_JS": mean_js,
        "mean_test_agreement": mean_agreement,
        "best_cell": best,
        "real_llm_training": False,
        "real_closed_loop": False,
        "blocker": "No faithful full student HF/LoRA matched-text OPD launcher plus real interactive closed-loop evaluator is present in this checkout.",
    }
    write_json(OUT / "MATCHED_TEXT_HANDOFF.json", handoff)
    return handoff


def finalize_main_table(matched: dict[str, object]) -> None:
    importance = json.loads((OUT / "IMPORTANCE_VALUE_GATE.json").read_text(encoding="utf-8"))
    ophsd = json.loads((OUT / "OPHSD_HANDOFF.json").read_text(encoding="utf-8"))
    rows = [
        {"method": "Base Student", "status": "not_completed_for_0816_main_table", "closed_loop_reward": "", "proxy_metric": "", "inference_privilege": "none", "source": "not rerun under exact 0816 protocol", "claim_allowed": False},
        {"method": "Full Harness", "status": "not_completed_for_0816_main_table", "closed_loop_reward": "", "proxy_metric": "", "inference_privilege": "full harness", "source": "not rerun under exact 0816 protocol", "claim_allowed": False},
        {"method": "Matched Text OPD", "status": "route_head_proxy_completed_real_closed_loop_not_run", "closed_loop_reward": "", "proxy_metric": f"mean_test_JS={matched['mean_common_reference_test_JS']:.8f}", "inference_privilege": "none in proxy eval", "source": "MATCHED_TEXT_HANDOFF.json", "claim_allowed": False},
        {"method": "OPHSD-style", "status": ophsd.get("status", "blocked_no_faithful_contract"), "closed_loop_reward": "", "proxy_metric": "", "inference_privilege": "none target", "source": "OPHSD_HANDOFF.json", "claim_allowed": False},
        {"method": "Our Structured Component OPD", "status": "supporting_evidence_only_no_new_training", "closed_loop_reward": "", "proxy_metric": "", "inference_privilege": "none at target inference", "source": "H100-4 verify/influence/B-utility confirmations", "claim_allowed": False},
    ]
    fields = ["method", "status", "closed_loop_reward", "proxy_metric", "inference_privilege", "source", "claim_allowed"]
    write_csv(OUT / "MAIN_COMPARISON_TABLE.csv", rows, fields)
    lines = [
        "# MAIN_COMPARISON_TABLE", "",
        "Only completed evidence is filled. Proxy route-head numbers are marked as proxy and do not support paper-grade closed-loop claims.", "",
        "| method | status | closed_loop_reward | proxy_metric | inference_privilege | source | claim_allowed |",
        "|---|---|---:|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[f]) for f in fields) + " |")
    lines += [
        "", "## Supporting Evidence", "",
        "- Matched Text V2 round-trip: `609/609`; route-head seeds 42-45 completed; real closed-loop not run.",
        f"- `importance_tagging` gate: `{importance.get('status')}`, n_states={importance.get('n_states')}, mean_I={float(importance.get('mean_I_name_normalized', 0.0)):.6f}.",
        "- OPHSD-style remains blocked by missing whole-harness terminal-context renderer/training launcher/evaluator binding.",
    ]
    (OUT / "MAIN_COMPARISON_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_handoff(matched: dict[str, object]) -> None:
    handoff = json.loads((OUT / "H1004_BTP_HANDOFF.json").read_text(encoding="utf-8"))
    handoff.update({
        "status": "partial_completed_matched_proxy_completed_ophsd_blocked",
        "matched_text": matched,
        "main_claim_allowed": False,
        "reason": "Matched Text V2 route-head proxy completed, but faithful LLM OPD closed-loop and OPHSD-style baseline remain blocked.",
    })
    write_json(OUT / "H1004_BTP_HANDOFF.json", handoff)
    run_manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    run_manifest.update({
        "status": "partial_completed_matched_proxy_completed_ophsd_blocked",
        "training_launched": True,
        "training_scope": "matched_text_route_head_proxy_seeds_42_43_44_45",
        "real_closed_loop_launched": False,
    })
    write_json(OUT / "RUN_MANIFEST.json", run_manifest)
    (OUT / "STATUS_LIVE.md").write_text(
        "# STATUS_LIVE - h1004_0816_novelty_baselines\n\n"
        "- novelty_matrix: completed\n"
        "- matched_text_v2_data: recovered, 609/609 round-trip\n"
        "- matched_text_route_head_proxy: completed seeds 42,43,44,45\n"
        "- matched_text_real_llm_closed_loop: blocked_not_run\n"
        "- ophsd_training: blocked_no_faithful_contract\n"
        "- importance_value_mining: completed from H100-4 REAL_INF_CONFIRM128 source rows\n"
        "- main_claim_allowed: false\n",
        encoding="utf-8",
    )


def main() -> int:
    generate_base()
    matched = finalize_matched_text()
    finalize_main_table(matched)
    finalize_handoff(matched)
    rewrite_sha256sums()
    print(json.dumps({"out_dir": str(OUT), "status": "finalized", "matched": matched["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
