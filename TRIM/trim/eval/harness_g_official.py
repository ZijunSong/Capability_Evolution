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


def _merged_per_query(run_dir: Path) -> Path | None:
    """Group-level trace only. A shard file must not stand in for the merged run."""
    for cand in (
        run_dir / "PER_QUERY.jsonl",
        run_dir / "harness" / "PER_QUERY.jsonl",
    ):
        if cand.is_file():
            return cand
    return None


def _find_per_query(run_dir: Path) -> Path | None:
    return _merged_per_query(run_dir)


def validate_official_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    failures: list[str] = []
    checks: dict[str, Any] = {
        "output_complete": True,
        "metric_formula_ok": True,
        "execution_healthy": True,
        "component_contract_ok": True,
    }
    launch_path = root / "LAUNCH.json"
    launch: dict[str, Any] = {}
    if launch_path.is_file():
        launch = _load_json(launch_path)
    else:
        failures.append("LAUNCH.json missing")
        checks["output_complete"] = False

    graph_path = str(launch.get("graph_index_path") or "").strip()
    scope = str(launch.get("graph_scope") or "")
    fingerprint = str(launch.get("graph_fingerprint") or "")
    if not graph_path:
        failures.append("graph_index_path empty")
        checks["component_contract_ok"] = False
    if is_episode_scope(scope) or (scope and not is_corpus_scope(scope)):
        failures.append(f"graph_scope={scope}")
        checks["component_contract_ok"] = False
    if not fingerprint:
        failures.append("graph_fingerprint empty")
        checks["component_contract_ok"] = False

    planned = [str(x) for x in (launch.get("planned_query_ids") or launch.get("query_ids") or []) if str(x)]
    n_expected = launch.get("n_expected") or launch.get("n_queries")
    benchmark = str(launch.get("benchmark") or "")
    if str(benchmark) in {"bcplus_test_50", "bcplus_50", "test_50"}:
        n_expected = int(n_expected or 50)

    shard_traces = list(root.glob("shards/**/PER_QUERY.jsonl"))
    pq = _merged_per_query(root)
    rows: list[dict[str, Any]] = []
    if pq is None:
        if shard_traces:
            failures.append(
                "merged PER_QUERY.jsonl missing; refusing to substitute a shard PER_QUERY for the group"
            )
        else:
            failures.append("PER_QUERY.jsonl missing")
        checks["output_complete"] = False
    else:
        rows = _load_jsonl(pq)
        if not rows:
            failures.append("PER_QUERY.jsonl empty")
            checks["output_complete"] = False
        ids = [str(r.get("query_id") or "") for r in rows]
        if any(not qid for qid in ids):
            failures.append("blank query_id in PER_QUERY.jsonl")
            checks["output_complete"] = False
        if len(ids) != len(set(ids)):
            dup = sorted({qid for qid in ids if ids.count(qid) > 1})
            failures.append(f"duplicate query_id: {dup[:8]}")
            checks["output_complete"] = False
        if n_expected is not None and len(rows) != int(n_expected):
            failures.append(f"expected {int(n_expected)} rows, got {len(rows)}")
            checks["output_complete"] = False
        if planned:
            got = set(ids)
            want = set(planned)
            missing = sorted(want - got)
            extra = sorted(got - want)
            if missing:
                failures.append(f"missing planned query_id: {missing[:8]}")
                checks["output_complete"] = False
            if extra:
                failures.append(f"unexpected query_id: {extra[:8]}")
                checks["output_complete"] = False
        row_fps = [str(r.get("graph_fingerprint") or "") for r in rows]
        scopes = {str(r.get("graph_scope") or "") for r in rows}
        if rows and any(not r.get("graph_enabled") for r in rows):
            failures.append(f"{sum(1 for r in rows if not r.get('graph_enabled'))}/{len(rows)} rows graph_enabled!=true")
            checks["component_contract_ok"] = False
        if rows and any(is_episode_scope(s) or s not in EXPECTED_CORPUS_SCOPES for s in scopes):
            failures.append(f"non-corpus graph_scope in traces: {sorted(scopes)}")
            checks["component_contract_ok"] = False
        if "" in row_fps:
            failures.append("empty graph_fingerprint in rows")
            checks["component_contract_ok"] = False
        if len(set(row_fps)) > 1:
            failures.append("graph fingerprint inconsistent across rows")
            checks["component_contract_ok"] = False
        if fingerprint and row_fps and any(fp != fingerprint for fp in row_fps):
            failures.append("row graph_fingerprint != LAUNCH")
            checks["component_contract_ok"] = False
        launch_contract = str(launch.get("contract_sha256") or "")
        launch_model = str(launch.get("model_name") or "")
        for row in rows:
            row_contract = str(row.get("contract_sha256") or "")
            if launch_contract and row_contract and row_contract != launch_contract:
                failures.append(f"{row.get('query_id')} contract_sha256 != LAUNCH")
                checks["component_contract_ok"] = False
                break
            row_model = str(row.get("model_name") or "")
            if launch_model and row_model and row_model != launch_model:
                failures.append(f"{row.get('query_id')} model_name != LAUNCH")
                checks["component_contract_ok"] = False
                break
        for row in rows:
            metric_fails = audit_trace_metrics(row)
            if metric_fails:
                checks["metric_formula_ok"] = False
                failures.extend(metric_fails)
            n_gen = int(row.get("n_generated") or 0)
            n_att = len(row.get("attempt_events") or [])
            if n_gen and n_att and n_gen != n_att:
                failures.append(f"{row.get('query_id')} attempt events {n_att} != n_generated {n_gen}")
                checks["execution_healthy"] = False
            elif n_gen and not n_att:
                failures.append(f"{row.get('query_id')} missing attempt_events for n_generated={n_gen}")
                checks["execution_healthy"] = False
            if int(row.get("n_infrastructure_failures") or 0) > 0:
                failures.append(f"{row.get('query_id')} infrastructure_failure={row.get('n_infrastructure_failures')}")
                checks["execution_healthy"] = False

    fingerprints: set[str] = set()
    for fp_path in root.glob("**/CONTRACT_FINGERPRINT.json"):
        payload = _load_json(fp_path)
        fingerprints.add(str(payload.get("graph_fingerprint") or ""))
    if fingerprints and ("" in fingerprints or len(fingerprints) > 1):
        failures.append("graph fingerprint inconsistent across shards")
        checks["component_contract_ok"] = False
    if fingerprint and fingerprints and any(fp != fingerprint for fp in fingerprints):
        failures.append("shard graph_fingerprint != LAUNCH")
        checks["component_contract_ok"] = False

    health_ok = not failures
    diagnostic_only = (not checks["execution_healthy"]) or (not checks["output_complete"])
    return {
        "ok": health_ok,
        "status": "OFFICIAL_RUN_VALID" if health_ok else "OFFICIAL_RUN_INVALID",
        "health_ok": health_ok,
        "diagnostic_only": diagnostic_only and not health_ok,
        "failures": failures,
        "checks": checks,
        "n_queries": len(rows),
        "n_expected": int(n_expected) if n_expected is not None else None,
        "graph_scope": scope,
        "graph_fingerprint": fingerprint,
    }


def write_partial_summary(run_dir: str | Path, validation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Diagnostic coverage record. Never a substitute for an official score."""
    root = Path(run_dir)
    launch: dict[str, Any] = {}
    launch_path = root / "LAUNCH.json"
    if launch_path.is_file():
        launch = _load_json(launch_path)
    planned = [str(x) for x in (launch.get("planned_query_ids") or launch.get("query_ids") or []) if str(x)]
    pq = _merged_per_query(root)
    rows = _load_jsonl(pq) if pq is not None else []
    got = [str(r.get("query_id") or "") for r in rows if str(r.get("query_id") or "")]
    got_set = set(got)
    missing = [qid for qid in planned if qid not in got_set]
    n_planned = len(planned) if planned else launch.get("n_expected")
    payload = {
        "official": False,
        "status": "PARTIAL_SUMMARY",
        "completed": len(got_set),
        "planned": int(n_planned) if n_planned is not None else None,
        "missing": len(missing) if planned else None,
        "missing_query_ids": missing,
        "n_rows": len(rows),
        "duplicate_query_ids": sorted({qid for qid in got if got.count(qid) > 1}),
        "note": (
            "Not an official score. Official summary requires the planned query-id set "
            "exactly, with one scorable row per id and a matching contract."
        ),
        "validation_failures": list((validation or {}).get("failures") or []),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "PARTIAL_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def assert_official_run(run_dir: str | Path) -> dict[str, Any]:
    result = validate_official_run(run_dir)
    if not result["ok"]:
        write_partial_summary(run_dir, result)
        detail = "; ".join(result["failures"][:12])
        raise RuntimeError(f"Refusing to generate official summary from invalid Harness-G run: {detail}")
    return result
