"""Append stage sections to result-record.md."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def format_stage_section(
    *,
    stage: str,
    setting: Mapping[str, Any],
    results: Mapping[str, Any],
    paired: Mapping[str, Any] | None = None,
    gate: str = "UNRESOLVED",
    decision: str = "",
    date: str | None = None,
) -> str:
    day = date or datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    lines = [
        f"## {day} SCAPE {stage}",
        "",
        "### Setting",
    ]
    for k, v in setting.items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "### Results", "| metric | value |", "|---|---:|"])
    for k, v in results.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.6g} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("### Paired")
    if paired:
        for k, v in paired.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (none)")
    lines.extend(["", "### Gate", str(gate), "", "### Decision", decision or "(none)", ""])
    return "\n".join(lines)


def append_result_record(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = path.read_text(encoding="utf-8") if path.exists() else "# SCAPE result-record\n\n"
    if not prev.endswith("\n"):
        prev += "\n"
    path.write_text(prev + "\n" + section, encoding="utf-8")
