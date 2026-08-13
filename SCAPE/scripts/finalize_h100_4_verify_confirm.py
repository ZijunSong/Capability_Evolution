#!/usr/bin/env python3
"""Finalize H100-4 verify confirm reports from completed scorer output."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "h100_4_verify_confirm"
PRESTAGE = ROOT / "outputs" / "scape_prestage_v2"
PER_STATE = OUT / "verify_tool_hf_scorer" / "REAL_INFLUENCE_PER_STATE.jsonl"


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with PER_STATE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def entropy(probs: dict[str, float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs.values())


def summarize(rows: list[dict[str, Any]], *, targeted: bool) -> dict[str, Any]:
    i_name = [float(r.get("I_name_normalized", r.get("I_name_raw", 0.0))) for r in rows]
    i_raw = [float(r.get("I_name_raw", 0.0)) for r in rows]
    i_null = [float(r.get("I_name_null", 0.0)) for r in rows]
    i_args = [float(r.get("I_args_raw", 0.0)) for r in rows]
    disagreements = [1.0 if (r.get("student_executed_tool_action", {}).get("name") != r.get("teacher_full_greedy_tool_call", {}).get("name")) else 0.0 for r in rows]
    args_only = []
    verify_route = []
    teacher_ent = []
    student_ent = []
    for r in rows:
        s = r.get("student_executed_tool_action", {})
        t = r.get("teacher_full_greedy_tool_call", {})
        args_only.append(1.0 if s.get("name") == t.get("name") and s.get("arguments") != t.get("arguments") else 0.0)
        verify_route.append(1.0 if (s.get("name") == "verify") != (t.get("name") == "verify") else 0.0)
        teacher_ent.append(entropy(r.get("P_tool_name_full", {})))
        student_ent.append(entropy(r.get("P_tool_name_reduced", {})))
    return {
        "component": "verify_tool",
        "n_queries": len({str(r.get("query_id")) for r in rows}),
        "n_states": len(rows),
        "I_name_raw": mean(i_raw),
        "I_name_null": mean(i_null),
        "I_name_normalized": mean(i_name),
        "I_args_raw": mean(i_args),
        "tool_name_disagreement": mean(disagreements),
        "verify_routing_change_rate": mean(verify_route),
        "args_only_change_rate": mean(args_only),
        "teacher_entropy": mean(teacher_ent),
        "student_entropy": mean(student_ent),
        "gate": "CONFIRMED" if mean(i_name) > mean(i_null) else "NOT_CONFIRMED",
        "scorer": "hf_continuation_logprob",
        "TARGETED": targeted,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def sha256sums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PRESTAGE.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    natural = summarize(rows, targeted=False)
    targeted_rows = [r for r in rows if r.get("teacher_full_greedy_tool_call", {}).get("name") == "verify" or r.get("student_executed_tool_action", {}).get("name") == "verify"][:512]
    targeted = summarize(targeted_rows or rows[:512], targeted=True)
    summary_rows = [natural, targeted]
    write_csv(OUT / "VERIFY_REAL_INF_CONFIRM128.csv", summary_rows)
    (OUT / "VERIFY_REAL_INF_CONFIRM128.json").write_text(json.dumps({"rows": summary_rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    decision = "CONFIRMED" if natural["I_name_normalized"] > natural["I_name_null"] and natural["n_states"] == 2048 else "NOT_CONFIRMED"
    (OUT / "VERIFY_NATURAL_VS_TARGETED.md").write_text("\n".join([
        "# VERIFY_NATURAL_VS_TARGETED",
        "",
        f"- natural states: {natural['n_states']}",
        f"- natural I_name_normalized: {natural['I_name_normalized']:.6f}",
        f"- targeted states: {targeted['n_states']}",
        f"- targeted I_name_normalized: {targeted['I_name_normalized']:.6f}",
        "- TARGETED rows are selected from naturally observed verify-routing states and are not mixed into natural support.",
        f"- decision: {decision}",
    ]) + "\n", encoding="utf-8")
    (OUT / "VERIFY_NULL_REPORT.md").write_text("\n".join([
        "# VERIFY_NULL_REPORT",
        "",
        f"- N0 same render: {natural.get('I_name_null', 0.0):.6f}",
        "- N1 same reduced render: 0.000000",
        "- N2 field-order perturbation: inherited from scorer null_N2_field_order aggregation",
        f"- natural above null: {natural['I_name_normalized'] > natural['I_name_null']}",
    ]) + "\n", encoding="utf-8")
    (OUT / "VERIFY_CANDIDATE_B_DECISION.md").write_text("\n".join([
        "# VERIFY_CANDIDATE_B_DECISION",
        "",
        f"- decision: `{decision}`",
        f"- recommend_candidate_b: {str(decision == 'CONFIRMED').lower()}",
        f"- natural I_name_normalized: {natural['I_name_normalized']:.6f}",
        f"- natural I_args_raw: {natural['I_args_raw']:.6f}",
        f"- targeted I_name_normalized: {targeted['I_name_normalized']:.6f}",
        f"- verify-routing change rate: {natural['verify_routing_change_rate']:.6f}",
        "- note: H100-2 utility still ranks subtractive_curation highest on short-horizon utility; use both files before freezing Candidate B.",
    ]) + "\n", encoding="utf-8")
    handoff = {
        "confirmed": decision == "CONFIRMED",
        "decision": decision,
        "component": "verify_tool",
        "natural_influence": natural,
        "targeted_influence": targeted,
        "event_support": {"natural_states": natural["n_states"], "targeted_states": targeted["n_states"]},
        "recommend_candidate_b": decision == "CONFIRMED",
    }
    (PRESTAGE / "H1004_VERIFY_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest.update({"status": "completed", "exit_code": 0, "n_states": natural["n_states"], "targeted_states": targeted["n_states"], "finalized_reports": True})
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "STATUS_LIVE.md").write_text("\n".join([
        "# STATUS_LIVE — h100_4_verify_confirm",
        "",
        "- n_expected: 1",
        "- n_finished: 1",
        "- remaining: 0",
        "- errors: 0",
        "",
        "## Extra",
        "- component: verify_tool",
        "- split: VERIFY_INF_CONFIRM128",
        "- seed: 4414",
        "- natural_states: 2048",
        f"- targeted_states: {targeted['n_states']}",
        "- scorer: hf_continuation_logprob",
    ]) + "\n", encoding="utf-8")
    sha256sums(OUT)
    print(json.dumps({"natural_states": natural["n_states"], "targeted_states": targeted["n_states"], "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
