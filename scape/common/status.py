"""STATUS_LIVE.md writer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def render_status_live(
    *,
    stage: str,
    run_id: str,
    n_expected: int,
    n_finished: int,
    errors: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    errs = errors or []
    lines = [
        f"# STATUS_LIVE — {stage}",
        "",
        f"- updated: {now}",
        f"- run_id: `{run_id}`",
        f"- n_expected: {n_expected}",
        f"- n_finished: {n_finished}",
        f"- remaining: {max(0, n_expected - n_finished)}",
        f"- errors: {len(errs)}",
    ]
    if errs:
        lines.append("")
        lines.append("## Errors")
        for e in errs[:20]:
            lines.append(f"- {e}")
    if extra:
        lines.append("")
        lines.append("## Extra")
        for k, v in extra.items():
            lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def write_status_live(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_status_live(**kwargs), encoding="utf-8")
