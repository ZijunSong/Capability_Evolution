"""Validate that a Harness-G result directory is an official (non-fallback) run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.eval.harness_g_contract import (
    EXPECTED_CORPUS_SCOPES,
    audit_trace_metrics,
    is_corpus_scope,
    is_episode_scope,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _find_per_query(run_dir: Path) -> Path | None:
    for cand in (
        run_dir / "PER_QUERY.jsonl",
        run_dir / "harness" / "PER_QUERY.jsonl",
    ):
        if cand.is_file():
            return cand
    matches = list(run_dir.glob("**/PER_QUERY.jsonl"))
    return matches[0] if matches else None


def validate_official_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    failures: list[str] = []
    launch_path = root / "LAUNCH.json"
    launch: dict[str, Any] = {}
    if launch_path.is_file():
        launch = _load_json(launch_path)
    else:
        failures.append("LAUNCH.json missing")

    graph_path = str(launch.get("graph_index_path") or "").strip()
    scope = str(launch.get("graph_scope") or "")
    fingerprint = str(launch.get("graph_fingerprint") or "")
    if not graph_path:
        failures.append("graph_index_path empty")
    if is_episode_scope(scope) or (scope and not is_corpus_scope(scope)):
        failures.append(f"graph_scope={scope}")
    if not fingerprint:
        failures.append("graph_fingerprint empty")

    pq = _find_per_query(root)
    rows: list[dict[str, Any]] = []
    if pq is None:
        failures.append("PER_QUERY.jsonl missing")
    else:
        rows = _load_jsonl(pq)
        fps = {str(r.get("graph_fingerprint") or "") for r in rows}
        scopes = {str(r.get("graph_scope") or "") for r in rows}
        if any(not r.get("graph_enabled") for r in rows):
            failures.append(f"{sum(1 for r in rows if not r.get('graph_enabled'))}/{len(rows)} rows graph_enabled!=true")
        if any(is_episode_scope(s) or s not in EXPECTED_CORPUS_SCOPES for s in scopes):
            failures.append(f"non-corpus graph_scope in traces: {sorted(scopes)}")
        if len({fp for fp in fps if fp}) > 1:
            failures.append("graph fingerprint inconsistent across rows")
        for row in rows:
            failures.extend(audit_trace_metrics(row))

    fingerprints: set[str] = set()
    for fp_path in root.glob("**/CONTRACT_FINGERPRINT.json"):
        payload = _load_json(fp_path)
        fingerprints.add(str(payload.get("graph_fingerprint") or ""))
    if len({x for x in fingerprints if x}) > 1:
        failures.append("graph fingerprint inconsistent across shards")

    ok = not failures
    return {
        "ok": ok,
        "status": "OFFICIAL_RUN_VALID" if ok else "OFFICIAL_RUN_INVALID",
        "failures": failures,
        "n_queries": len(rows),
        "graph_scope": scope,
        "graph_fingerprint": fingerprint,
    }


def assert_official_run(run_dir: str | Path) -> dict[str, Any]:
    result = validate_official_run(run_dir)
    if not result["ok"]:
        detail = "; ".join(result["failures"][:12])
        raise RuntimeError(f"Refusing to generate official summary from invalid Harness-G run: {detail}")
    return result
