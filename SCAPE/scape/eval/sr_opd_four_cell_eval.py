"""Strict four-cell evaluator for sr_opd_ce + CISPO adapters.

Reports Legal action rate, Test Evidence Recall@5, and tool cost.
Never reads legacy EasyOPD adapter paths unless the caller passes them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scape.eval.adapter_reload_audit import audit_saved_adapter, write_reload_audit
from scape.eval.browsecomp_retrieval import RetrievalBackend, evidence_recall, open_retrieval
from scape.eval.official_query_pool import load_official_384
from scape.training.action_codec import STUDENT_NATIVE_TOOLS


def legal_rate(tool_names: list[str]) -> float:
    if not tool_names:
        return 0.0
    return sum(1 for n in tool_names if n in STUDENT_NATIVE_TOOLS) / len(tool_names)


def summarize_traces(traces: list[dict[str, Any]], *, setting: str, retrieval_name: str) -> dict[str, Any]:
    n = max(1, len(traces))
    legal = [legal_rate(t.get("tool_names") or []) for t in traces]
    rec5 = [float(t.get("evidence_recall_at_5") or 0.0) for t in traces]
    rec100 = [float(t.get("evidence_recall_at_100") or 0.0) for t in traces]
    tools = [float(t.get("n_tool_calls") or 0.0) for t in traces]
    searches = [float(t.get("n_search_calls") or 0.0) for t in traces]
    return {
        "setting": setting,
        "n_queries": len(traces),
        "legal_action_rate": sum(legal) / n,
        "test_evidence_recall_at_5": sum(rec5) / n if retrieval_name != "none" else None,
        "test_evidence_recall_at_100": sum(rec100) / n if retrieval_name != "none" else None,
        "mean_tool_calls_per_query": sum(tools) / n,
        "tool_search_cost": sum(searches) / n,
        "retrieval": retrieval_name,
        "student_inference_privilege": False,
        "eval_harness": "H_min",
        "legacy_tool_token_kl_used": False,
        "opd_loss": "sr_opd_ce",
        "rl_loss": "cispo",
    }


def search_metrics(searcher: RetrievalBackend, query: str, evidence: list[str]) -> dict[str, Any]:
    hits5 = searcher.search(query, 5)
    hits100 = searcher.search(query, 100) if searcher.name != "none" else []
    return {
        "retrieved_at_5": [h.docid for h in hits5],
        "retrieved_at_100": [h.docid for h in hits100],
        "evidence_recall_at_5": evidence_recall([h.docid for h in hits5], evidence),
        "evidence_recall_at_100": evidence_recall([h.docid for h in hits100], evidence),
    }


def write_eval_outputs(
    out: Path,
    *,
    component_id: str,
    summaries: list[dict[str, Any]],
    adapter_audits: list[dict[str, Any]],
    pool_meta: dict[str, Any],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    reload = write_reload_audit(out / "ADAPTER_RELOAD_AUDIT.json", adapter_audits)
    payload = {
        "status": "SR_OPD_CISPO_FOUR_CELL_EVAL",
        "component": component_id,
        "opd_loss": "sr_opd_ce",
        "rl_loss_fn": "cispo",
        "legacy_tool_token_kl_hook_used": False,
        "protocol_complete_rl_opd": True,
        "pool": pool_meta,
        "settings": summaries,
        "adapter_reload": reload,
    }
    (out / "FOUR_CELL_OFFICIAL_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def load_eval_pool(manifest: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_official_384(manifest=manifest)


def audit_adapter_map(adapter_map: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for cell, path in adapter_map.items():
        if not path:
            rows.append({"cell": cell, "adapter_dir": None, "reload_ready": cell == "before", "exists": False})
            continue
        rows.append(audit_saved_adapter(Path(path), cell=cell))
    return rows
