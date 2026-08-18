#!/usr/bin/env python3
"""Score Clean Graph-Hybrid micro cells with Metric V2."""

from __future__ import annotations

import argparse
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


def _base_of(tag: str) -> str:
    if tag.startswith("full_"):
        return "CLEAN-SFT-FULL"
    if tag.startswith("tool_"):
        return "CLEAN-SFT-TOOL"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/0814_clean_mechanism")
    ap.add_argument("--max-valid", type=int, default=128)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    out = args.out_dir
    micro = out / "micro"
    valid = REPO / "outputs/0813_next_h20/graph_hybrid/data/GH_VALID_1K.jsonl"
    if not valid.is_file():
        print("missing GH valid")
        return 1
    valid_rows = load_same_state_jsonl(valid)[: args.max_valid]
    rows_out: list[dict] = []
    teacher_cache: dict[str, ScapeHFToolOPD] = {}

    for summary_path in sorted(micro.glob("gpu*/*/summary.json")):
        data = json.loads(summary_path.read_text())
        tag = summary_path.parent.name
        gpu = summary_path.parent.parent.name
        loss = data.get("loss_path", "tool_token_kl")
        ck = data.get("checkpoint_merged") or data.get("checkpoint_lora")
        if not ck or not Path(ck).exists():
            continue
        base_model = data.get("base_checkpoint") or "/data/ppnm/models/gpt-oss-20b"
        if base_model not in teacher_cache:
            teacher_cache[base_model] = ScapeHFToolOPD(
                model_path=str(base_model), device_map=f"cuda:{args.gpu}", use_lora=False
            )
        teacher = teacher_cache[base_model]
        pre = aggregate_rows_v2(teacher, teacher, valid_rows, loss_path="tool_name_only_kl")
        student = ScapeHFToolOPD(model_path=str(ck), device_map=f"cuda:{args.gpu}", use_lora=False)
        post = aggregate_rows_v2(teacher, student, valid_rows, loss_path=loss)  # type: ignore[arg-type]
        ok, reason = v2_gate_pass(pre, post)
        rows_out.append(
            {
                "gpu": gpu,
                "tag": tag,
                "base": _base_of(tag),
                "loss_path": loss,
                "seed": data.get("seed"),
                "n_samples": data.get("n_samples"),
                "JS_name_pre": pre.JS_name,
                "JS_name_post": post.JS_name,
                "CE_T_on_S_pre": pre.CE_T_on_S,
                "CE_T_on_S_post": post.CE_T_on_S,
                "invalid_tool_rate_pre": pre.invalid_tool_rate,
                "invalid_tool_rate_post": post.invalid_tool_rate,
                "v2_gate_pass": ok,
                "v2_gate_reason": reason,
                "legacy_L_m": data.get("L_m"),
            }
        )
        del student

    csv_path = out / "CLEAN_MICRO_V2.csv"
    if rows_out:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    lines = [
        "# CLEAN_MICRO_REPORT",
        "",
        f"- cells scored: {len(rows_out)}",
        f"- v2_pass: {sum(1 for r in rows_out if r['v2_gate_pass'])}",
        "",
        "| base | tag | loss | seed | JS_pre | JS_post | CE_pre | CE_post | pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows_out:
        lines.append(
            f"| {r['base']} | {r['tag']} | {r['loss_path']} | {r['seed']} | "
            f"{r['JS_name_pre']:.4f} | {r['JS_name_post']:.4f} | "
            f"{r['CE_T_on_S_pre']:.4f} | {r['CE_T_on_S_post']:.4f} | {r['v2_gate_pass']} |"
        )
    (out / "CLEAN_MICRO_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"n": len(rows_out), "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
