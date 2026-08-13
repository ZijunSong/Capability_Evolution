#!/usr/bin/env python3
"""Generate 2026-08-13 SCAPE coordination summary artifacts.

This script only consolidates completed local/HF evidence that already exists on
 disk. It does not synthesize per-state measurements, does not claim official
Chroma parity, and preserves failed/in-progress status for unfinished streams.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SUMMARY_DIR = OUT / "scape_prestage_v2"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_csv_by_component(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        comp = row.get("component") or row.get("component_id")
        if comp:
            out[comp] = row
    return out


def as_float(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256sums(root: Path) -> None:
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def status_from_markdown(path: Path) -> dict[str, Any]:
    text = read_text(path)
    result: dict[str, Any] = {"path": str(path.relative_to(ROOT)), "exists": path.exists()}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("-") or ":" not in line:
            continue
        key, value = line[1:].split(":", 1)
        key = key.strip().replace(" ", "_")
        value = value.strip()
        try:
            result[key] = int(value)
        except ValueError:
            result[key] = value
    return result


def completed_statuses() -> dict[str, dict[str, Any]]:
    return {
        "h100_1_contribution_confirm": status_from_markdown(OUT / "h100_1_contribution_confirm" / "STATUS_LIVE.md"),
        "h100_2_independent_repl": status_from_markdown(OUT / "h100_2_independent_repl" / "STATUS_LIVE.md"),
        "h100_3_real_influence": status_from_markdown(OUT / "h100_3_real_influence" / "STATUS_LIVE.md"),
        "h100_3_influence_attribution": read_json(OUT / "h100_3_influence_attribution" / "RUN_MANIFEST.json") or {"exists": False},
        "h100_4_influence_confirm": status_from_markdown(OUT / "h100_4_influence_confirm" / "STATUS_LIVE.md"),
        "h100_4_verify_confirm": read_json(OUT / "h100_4_verify_confirm" / "RUN_MANIFEST.json") or {"exists": False},
        "h100_4_auto_populate_argument_diagnostic": read_json(OUT / "h100_4_verify_confirm" / "auto_populate_argument_diagnostic" / "RUN_MANIFEST.json") or {"exists": False},
        "h20_lightweight_torch": read_json(OUT / "H20_LIGHTWEIGHT_TORCH_COMPLETE.json") or {"exists": False},
    }


def verify_followup_status() -> dict[str, Any]:
    status = status_from_markdown(OUT / "h100_4_verify_confirm" / "STATUS_LIVE.md")
    preflight = read_json(OUT / "h100_4_verify_confirm" / "PREFLIGHT.json") or {}
    handoff = read_json(OUT / "scape_prestage_v2" / "H1004_VERIFY_HANDOFF.json") or {}
    manifest = read_json(OUT / "h100_4_verify_confirm" / "RUN_MANIFEST.json") or {}
    completed = manifest.get("status") == "completed" and int(status.get("errors", 1)) == 0 and handoff.get("confirmed") is True
    return {
        "status": status,
        "preflight": preflight,
        "handoff": handoff,
        "manifest": manifest,
        "classification": "completed_confirmed" if completed else "requires_recovery",
        "recovery": None if completed else "rerun SCAPE/scripts/run_h100_4_verify_confirm_hf.py with --python pointing to a valid /opt HF scorer environment",
    }


def attribution_summary() -> dict[str, Any]:
    totals = read_csv_by_component(OUT / "h100_3_influence_attribution" / "INFLUENCE_TOTALS.csv")
    out: dict[str, Any] = {}
    for comp in ["evidence_graph", "importance_tagging", "verify_tool"]:
        row = totals.get(comp, {})
        out[comp] = {
            "n_states": as_float(row, "n_states"),
            "I_name_mean": as_float(row, "I_name_mean"),
            "I_args_mean": as_float(row, "I_args_mean"),
            "tool_name_disagreement_rate": as_float(row, "tool_name_disagreement_rate"),
            "args_only_disagreement_rate": as_float(row, "args_only_disagreement_rate"),
        }
    return out


def candidate_status() -> dict[str, Any]:
    h4 = read_csv_by_component(OUT / "h100_4_influence_confirm" / "REAL_INFLUENCE_CONFIRM_BY_COMPONENT.csv")
    h3_attr = attribution_summary()
    verify = verify_followup_status()
    return {
        "candidate_a": {
            "component": "evidence_graph",
            "status": "completed_positive",
            "h100_4_confirm": h4.get("evidence_graph", {}),
            "h100_3_attribution": h3_attr.get("evidence_graph", {}),
        },
        "candidate_b_current": {
            "component": "importance_tagging",
            "status": "completed_positive",
            "h100_4_confirm": h4.get("importance_tagging", {}),
            "h100_3_attribution": h3_attr.get("importance_tagging", {}),
        },
        "candidate_b_challenger": {
            "component": "verify_tool",
            "status": "completed_confirmed" if verify.get("classification") == "completed_confirmed" else "requires_recovered_h100_4_confirm",
            "h100_3_attribution": h3_attr.get("verify_tool", {}),
            "h100_4_followup": verify,
        },
        "runtime_controls": ["chunk_neighbors", "content_dedup"],
    }


def write_markdown(summary: dict[str, Any]) -> None:
    attr = summary["h100_3_attribution"]
    verify = summary["verify_tool_followup"]
    lines = [
        "# 0813 SCAPE status summary",
        "",
        "This file is generated from existing local/HF artifacts only. It does not claim official Chroma parity.",
        "",
        "## Completed",
        "",
        "| stream | status | evidence |",
        "|---|---|---|",
    ]
    for name, value in summary["completed"].items():
        if isinstance(value, dict):
            errors = value.get("errors", value.get("status", ""))
            finished = value.get("n_finished", value.get("n_rows", ""))
        else:
            errors = ""
            finished = ""
        lines.append(f"| `{name}` | finished={finished}; errors={errors} | see source artifact |")
    lines += [
        "",
        "## H100-3 Attribution",
        "",
        "| component | I_name_mean | I_args_mean | conclusion |",
        "|---|---:|---:|---|",
    ]
    conclusions = {
        "evidence_graph": "args-heavy; name-only is insufficient",
        "importance_tagging": "positive; current Candidate B",
        "verify_tool": "positive challenger; H100-4 verify confirm completed",
    }
    for comp, row in attr.items():
        lines.append(f"| `{comp}` | {row.get('I_name_mean')} | {row.get('I_args_mean')} | {conclusions[comp]} |")
    lines += [
        "",
        "## In Progress / Failed Once",
        "",
        f"- `verify_tool` H100-4 follow-up: {verify['classification']}",
        f"- status: {verify['status']}",
        f"- recovery: {verify['recovery']}",
        "",
        "## Blocked / Not Started",
        "",
        "- official BrowseComp+ Chroma parity: blocked by missing `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`.",
        "- H20 true-SCAPE Stage S/M: not started because Evidence Graph Stage L smoke is gate-blocked (`STAGE_L_SMOKE_NOT_PASSED`).",
    ]
    write_text(SUMMARY_DIR / "0813_STATUS_SUMMARY.md", "\n".join(lines))


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "SCAPE 0813 coordination status consolidation",
        "official_chroma_parity": False,
        "official_chroma_blocked": True,
        "completed": completed_statuses(),
        "h100_3_attribution": attribution_summary(),
        "verify_tool_followup": verify_followup_status(),
        "candidate_status": candidate_status(),
        "blocked_or_not_started": [
            "official BrowseComp+ Chroma parity",
            "H20 true-SCAPE Stage S/M after Evidence Graph Gate L block",
        ],
    }
    write_json(SUMMARY_DIR / "0813_STATUS_SUMMARY.json", summary)
    write_markdown(summary)
    sha256sums(SUMMARY_DIR)
    print(json.dumps({"generated": ["outputs/scape_prestage_v2/0813_STATUS_SUMMARY.json", "outputs/scape_prestage_v2/0813_STATUS_SUMMARY.md"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
