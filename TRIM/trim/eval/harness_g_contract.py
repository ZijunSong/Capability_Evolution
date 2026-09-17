"""Official Harness-G eval contract: gold metrics, corpus graph, provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_CORPUS_SCOPES = frozenset({"corpus", "corpus_graph", "bcplus_corpus", "global_corpus"})
EPISODE_SCOPES = frozenset({"episode_doc_store", "local_episode", "episode"})


def is_corpus_scope(scope: str | None) -> bool:
    return str(scope or "").strip() in EXPECTED_CORPUS_SCOPES


def is_episode_scope(scope: str | None) -> bool:
    return str(scope or "").strip() in EPISODE_SCOPES


def is_formal_harness_g_eval(
    *,
    harness: str | None = None,
    benchmark: str | None = None,
    smoke: bool = False,
    component_ids: Any = None,
    mask: Mapping[str, bool] | None = None,
) -> bool:
    from trim.adapters.harness_profiles import is_harness_g

    if smoke:
        return False
    if not is_harness_g(harness=harness, component_ids=component_ids, mask=mask):
        return False
    return str(benchmark or "").strip() in {"bcplus_test_50", "bcplus_50", "test_50"}


def require_graph_path(path: str | None, *, required: bool) -> str | None:
    text = str(path or "").strip()
    if required and not text:
        raise RuntimeError(
            "Official Harness-G evaluation requires --graph-index-path. "
            "Refusing episode_doc_store fallback."
        )
    return text or None


def require_graph_exists(path: str) -> Path:
    dest = Path(path)
    if not dest.exists():
        raise FileNotFoundError(f"Harness-G graph index not found: {dest}")
    return dest


def graph_metadata(graph: Any, *, path: str | None = None, required: bool = False) -> dict[str, Any]:
    meta_fn = getattr(graph, "metadata", None)
    payload = dict(meta_fn()) if callable(meta_fn) else {}
    scope = str(payload.get("graph_scope") or getattr(graph, "scope", "") or "")
    fingerprint = str(payload.get("graph_fingerprint") or "")
    if not fingerprint and hasattr(graph, "content_fingerprint"):
        fingerprint = str(graph.content_fingerprint() or "")
    payload.update(
        {
            "graph_index_path": str(path) if path else payload.get("graph_index_path"),
            "graph_scope": scope,
            "graph_fingerprint": fingerprint,
            "graph_enabled": is_corpus_scope(scope),
            "graph_required": bool(required),
        }
    )
    return payload


def validate_loaded_graph(graph: Any, *, required: bool, path: str | None = None) -> dict[str, Any]:
    meta = graph_metadata(graph, path=path, required=required)
    if not required:
        return meta
    scope = str(meta.get("graph_scope") or "")
    fingerprint = str(meta.get("graph_fingerprint") or "")
    if not scope:
        raise RuntimeError("Loaded Harness-G graph has empty scope")
    if is_episode_scope(scope) or not is_corpus_scope(scope):
        raise RuntimeError(
            f"Official Harness-G evaluation received non-corpus graph scope: {scope}"
        )
    if not fingerprint:
        raise RuntimeError("Harness-G graph fingerprint is empty")
    return meta


def official_metric_counts(
    *,
    selected: Sequence[str] | None,
    observed: Sequence[str] | None,
    gold: Sequence[str] | None,
    evidence: Sequence[str] | None,
) -> dict[str, Any]:
    from trim.eval.harness1_metrics import _id_set, set_recall, set_precision

    selected_ids = _id_set(selected)
    observed_ids = _id_set(observed)
    gold_ids = _id_set(gold)
    evidence_ids = _id_set(evidence)
    official = gold_ids or evidence_ids
    recall = set_recall(selected_ids, official)
    traj = set_recall(observed_ids | selected_ids if observed_ids else selected_ids, official)
    gold_recall = set_recall(selected_ids, gold_ids) if gold_ids else recall
    traj_gold = set_recall(observed_ids | selected_ids if observed_ids else selected_ids, gold_ids) if gold_ids else traj
    evidence_recall = set_recall(selected_ids, evidence_ids) if evidence_ids else 0.0
    traj_evidence = (
        set_recall(observed_ids | selected_ids if observed_ids else selected_ids, evidence_ids)
        if evidence_ids
        else 0.0
    )
    return {
        "recall": recall,
        "trajectory_recall": traj,
        "precision": set_precision(selected_ids, official),
        "gold_recall": gold_recall,
        "trajectory_gold_recall": traj_gold,
        "evidence_recall": evidence_recall,
        "trajectory_evidence_recall": traj_evidence,
        "n_gold": len(gold_ids),
        "n_gold_selected": len(selected_ids & gold_ids),
        "n_gold_observed": len((observed_ids | selected_ids) & gold_ids),
        "n_evidence": len(evidence_ids),
        "n_evidence_selected": len(selected_ids & evidence_ids),
        "n_evidence_observed": len((observed_ids | selected_ids) & evidence_ids),
    }


def audit_trace_metrics(row: Mapping[str, Any], *, tol: float = 1e-9) -> list[str]:
    failures: list[str] = []
    gold = row.get("gold_docids") or []
    if not gold:
        return ["missing gold_docids"]
    expected = official_metric_counts(
        selected=row.get("selected_docids") or [],
        observed=row.get("observed_docids") or [],
        gold=gold,
        evidence=row.get("evidence_docids") or [],
    )
    qid = str(row.get("query_id") or "?")
    for key in ("recall", "trajectory_recall"):
        got = float(row.get(key) or 0.0)
        want = float(expected[key])
        if abs(got - want) > tol:
            failures.append(f"{qid} {key} got={got} expected={want}")
    return failures
