from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


def event_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    active = sum(bool(r.get("component_event_active")) for r in rows)
    projectable = sum(bool(r.get("projection_valid")) for r in rows)
    valid_args = sum(bool(r.get("visibility_valid")) for r in rows)
    terminal = sum((r.get("terminal_reward") or 0) != 0 for r in rows)
    return {
        "n_queries": len({str(r.get("query_id")) for r in rows}),
        "n_states": n,
        "n_event_active": active,
        "event_rate": active / max(1, n),
        "n_projectable": projectable,
        "n_valid_args": valid_args,
        "n_terminal_reward": terminal,
    }


def write_event_support_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = event_support(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return metrics


def write_sha256sums(root: Path) -> None:
    lines = []
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {p.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
