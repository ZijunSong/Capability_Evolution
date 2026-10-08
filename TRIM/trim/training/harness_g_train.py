"""Harness-G training graph. Rollouts use the corpus index, not an episode fallback.

Non-smoke runs fail closed when the index is missing. Smoke runs may omit it
and build a per-query graph from the doc store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trim.adapters.harness_profiles import is_harness_g

_TRIM_ROOT = Path(__file__).resolve().parents[2]


def graph_path_candidates() -> list[Path]:
    return [
        _TRIM_ROOT.parent / "SCOPE" / "external" / "BrowseComp-Plus" / "indexes" / "harness_g_corpus_graph.pkl",
    ]


def default_graph_path() -> str | None:
    for path in graph_path_candidates():
        if path.is_file():
            return str(path)
    return None


def resolve_train_graph_path(args: Any) -> str | None:
    explicit = str(getattr(args, "graph_index_path", None) or "").strip()
    if explicit:
        return explicit
    return default_graph_path()


def load_train_graph(args: Any) -> Any | None:
    """Load the shared corpus graph for Harness-G rollouts. Harness-1 returns None."""
    if not is_harness_g(
        getattr(args, "harness", None),
        component_ids=getattr(args, "component", None),
    ):
        return None
    smoke = bool(getattr(args, "smoke", False))
    path = resolve_train_graph_path(args)
    if not path:
        if smoke:
            return None
        raise SystemExit(
            "Harness-G training requires a corpus graph. Pass --graph-index-path, "
            "or place harness_g_corpus_graph.pkl under "
            "SCOPE/external/BrowseComp-Plus/indexes/. "
            "--smoke may omit the index and use the per-query episode graph."
        )
    from trim.eval.harness_g_contract import require_graph_exists, validate_loaded_graph
    from trim.eval.harness_g_graph import load_graph_index

    require_graph_exists(path)
    graph = load_graph_index(path)
    if getattr(graph, "source_path", None) in {None, ""}:
        graph.source_path = path
    validate_loaded_graph(graph, required=not smoke, path=path)
    return graph
