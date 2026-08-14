#!/usr/bin/env python3
"""Score Graph-Hybrid micro checkpoints with V2 metrics and write report."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.same_state import load_same_state_jsonl
from scape.eval.learnability_metrics_v2 import aggregate_rows_v2, v2_gate_pass
from scape.training.hf_tool_opd import ScapeHFToolOPD


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-valid", type=int, default=256)
    args = ap.parse_args()

    out_root = REPO / "outputs/0813_next_h20"
    micro = out_root / "graph_hybrid/micro"
    valid = out_root / "graph_hybrid/data/GH_VALID_1K.jsonl"
    teacher_path = "/data/ppnm/models/harness-1"
    rows_out: list[dict] = []

    valid_rows = load_same_state_jsonl(valid)[:args.max_valid]
    teacher = ScapeHFToolOPD(model_path=teacher_path, device_map="cuda:0", use_lora=False)
    pre = aggregate_rows_v2(teacher, teacher, valid_rows, loss_path="tool_name_only_kl")

    for summary_path in sorted(micro.glob("gpu*/*/summary.json")):
        data = json.loads(summary_path.read_text())
        ck = Path(data["checkpoint_merged"])
        tag = summary_path.parent.name
        gpu = summary_path.parent.parent.name
        loss = data.get("loss_path", "tool_token_kl")
        student = ScapeHFToolOPD(model_path=str(ck), device_map="cuda:0", use_lora=False)
        post = aggregate_rows_v2(
            teacher, student, valid_rows,
            loss_path=loss,  # type: ignore[arg-type]
        )
        ok, reason = v2_gate_pass(pre, post)
        rows_out.append({
            "gpu": gpu,
            "tag": tag,
            "loss_path": loss,
            "JS_name_pre": pre.JS_name,
            "JS_name_post": post.JS_name,
            "CE_pre": pre.CE_T_on_S,
            "CE_post": post.CE_T_on_S,
            "legacy_L_m": data.get("L_m"),
            "legacy_d_pre": data.get("d_pre"),
            "legacy_d_post": data.get("d_post"),
            "v2_gate_pass": ok,
            "v2_gate_reason": reason,
        })
        del student

    csv_path = out_root / "graph_hybrid/GRAPH_HYBRID_MICRO_V2.csv"
    if rows_out:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    passes = [r for r in rows_out if r["v2_gate_pass"]]
    name_only = [r for r in rows_out if "name_only" in r["tag"] and r["v2_gate_pass"]]
    two_seed = len(name_only) >= 2

    lines = [
        "# GRAPH_HYBRID_MICRO_REPORT",
        "",
        f"- cells: {len(rows_out)}",
        f"- v2_pass: {len(passes)}",
        f"- name_only_two_seed_pass: {two_seed}",
        "",
        "| gpu | tag | JS_name Δ | CE Δ | v2_pass | reason |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in rows_out:
        js_d = r["JS_name_post"] - r["JS_name_pre"]
        ce_d = r["CE_post"] - r["CE_pre"]
        lines.append(
            f"| {r['gpu']} | {r['tag']} | {js_d:.4f} | {ce_d:.4f} | {r['v2_gate_pass']} | {r['v2_gate_reason']} |"
        )

    gate_pass = two_seed and all(
        r["JS_name_post"] < r["JS_name_pre"] and r["CE_post"] < r["CE_pre"]
        for r in name_only
    )
    lines.append("")
    lines.append(f"**Graph-Hybrid micro gate**: {'PASS' if gate_pass else 'FAIL'}")

    report = out_root / "GRAPH_HYBRID_MICRO_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    next_dec = "GRAPH_HYBRID_8K" if gate_pass else "PLACEMENT_BOUNDARY_RESULT"
    if not gate_pass and len(passes) == 0:
        next_dec = "PLACEMENT_BOUNDARY_RESULT"

    (out_root / "NEXT_DECISION.json").write_text(
        json.dumps({"NEXT_DECISION": next_dec}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_cells": len(rows_out), "v2_pass": len(passes), "gate_pass": gate_pass}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
