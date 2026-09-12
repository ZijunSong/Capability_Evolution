"""Strict four-cell evaluator for sr_opd_ce + CISPO adapters.

Reports Legal action rate, Test Evidence Recall@5, and tool cost.
Never reads legacy EasyOPD adapter paths unless the caller passes them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trim.eval.adapter_reload_audit import audit_saved_adapter, write_reload_audit
from trim.eval.browsecomp_retrieval import RetrievalBackend, evidence_recall, open_retrieval
from trim.eval.harness1_metrics import format_summary_table, summarize_quality_and_timing
from trim.eval.official_query_pool import (
    load_bcplus_830_split,
    load_official_384,
    SCORE_SPLIT_830,
    is_full_score_split,
)
from trim.training.action_codec import HARNESS_G_STUDENT_NATIVE_TOOLS, STUDENT_NATIVE_TOOLS


def legal_rate(tool_names: list[str], *, harness_g: bool = False) -> float:
    if not tool_names:
        return 0.0
    allowed = set(HARNESS_G_STUDENT_NATIVE_TOOLS) if harness_g else set(STUDENT_NATIVE_TOOLS)
    return sum(1 for n in tool_names if n in allowed) / len(tool_names)


def summarize_traces(
    traces: list[dict[str, Any]],
    *,
    setting: str,
    retrieval_name: str,
    harness_g: bool = False,
) -> dict[str, Any]:
    n = max(1, len(traces))
    legal = [
        legal_rate(t.get("tool_names") or t.get("names") or [], harness_g=harness_g) for t in traces
    ]
    rec5 = [float(t.get("evidence_recall_at_5") or 0.0) for t in traces]
    rec100 = [float(t.get("evidence_recall_at_100") or 0.0) for t in traces]
    tools = [float(t.get("n_tool_calls") or 0.0) for t in traces]
    searches = [float(t.get("n_search_calls") or 0.0) for t in traces]
    payload = {
        "setting": setting,
        "n_queries": len(traces),
        "legal_action_rate": sum(legal) / n,
        "test_evidence_recall_at_5": sum(rec5) / n if retrieval_name != "none" else None,
        "test_evidence_recall_at_100": sum(rec100) / n if retrieval_name != "none" else None,
        "mean_tool_calls_per_query": sum(tools) / n,
        "tool_search_cost": sum(searches) / n,
        "retrieval": retrieval_name,
        "student_inference_privilege": False,
        "eval_harness": "Harness-G" if harness_g else "H_min",
        "legacy_tool_token_kl_used": False,
        "opd_loss": "sr_opd_ce",
        "rl_loss": "cispo",
        **summarize_quality_and_timing(traces),
    }
    payload["n_queries"] = len(traces)
    return payload


def search_metrics(searcher: RetrievalBackend, query: str, evidence: list[str]) -> dict[str, Any]:
    hits5 = searcher.search(query, 5)
    hits100 = searcher.search(query, 100) if searcher.name != "none" else []
    normalize = getattr(searcher, "normalize_id", None)
    return {
        "retrieved_at_5": [h.docid for h in hits5],
        "retrieved_at_100": [h.docid for h in hits100],
        "evidence_recall_at_5": evidence_recall([h.docid for h in hits5], evidence, normalize=normalize),
        "evidence_recall_at_100": evidence_recall([h.docid for h in hits100], evidence, normalize=normalize),
    }


def split_summaries(
    traces: list[dict[str, Any]],
    *,
    setting: str,
    retrieval_name: str,
    eval_rows: list[dict[str, Any]],
    harness_g: bool = False,
) -> dict[str, Any]:
    by_id = {r["query_id"]: r for r in eval_rows}
    for tr in traces:
        rec = by_id.get(tr["query_id"]) or {}
        tr["official_split"] = rec.get("official_split") or "train"
    all_pool = summarize_traces(traces, setting=setting, retrieval_name=retrieval_name, harness_g=harness_g)
    test_traces = [t for t in traces if t.get("official_split") == "test"]
    official = summarize_traces(test_traces, setting=setting, retrieval_name=retrieval_name, harness_g=harness_g)
    official["split"] = "official_test"
    official["n_expected"] = len(test_traces)
    return {"setting": setting, "all_pool": all_pool, "official_test": official, "primary_split": "official_test"}


def pack_closed_loop_summary(
    split: dict[str, Any],
    *,
    leak: float,
    n_rows: int,
    primary_split: str = "official_test",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick the primary eval surface: 166-test or full BC+ 830."""
    all_pool = dict(split["all_pool"])
    official_test = dict(split["official_test"])
    use_full = is_full_score_split(primary_split) or str(primary_split) in {"all_pool"}
    payload = dict(all_pool if use_full else official_test)
    payload["primary_split"] = SCORE_SPLIT_830 if use_full else str(primary_split or "official_test")
    payload["n_expected"] = int(n_rows) if use_full else int(official_test.get("n_expected") or official_test.get("n_queries") or n_rows)
    payload["teacher_leak_rate"] = float(leak) / max(1, n_rows)
    payload["all_pool"] = all_pool
    payload["official_test"] = official_test
    payload["n_all_pool"] = all_pool.get("n_queries")
    payload["n_official_test"] = official_test.get("n_queries")
    if extra:
        payload.update(extra)
    return payload


def write_upstream_api_eval_outputs(
    out: Path,
    *,
    component_id: str,
    summaries: list[dict[str, Any]],
    pool_meta: dict[str, Any],
) -> dict[str, Any]:
    """Official result schema for base-model upstream_api / Harness-1 API eval."""
    out.mkdir(parents=True, exist_ok=True)
    primary = dict(summaries[0]) if summaries else {}
    coverage = dict(primary.get("coverage") or {})
    n_planned = int(primary.get("n_planned") or coverage.get("planned") or primary.get("n_queries") or 0)
    n_missing = int(coverage.get("missing") or primary.get("dropped_queries") or 0)
    n_infra = int(primary.get("n_infra_error") or 0)
    coverage_complete = n_planned > 0 and n_missing == 0
    infra_clean = n_infra == 0
    execution_complete = bool(primary.get("execution_complete", coverage_complete and infra_clean))
    formal_eligible = coverage_complete and infra_clean and execution_complete
    payload = {
        "status": "UPSTREAM_API_BASELINE_EVAL",
        "run_kind": "upstream_api_base_model",
        "component": component_id,
        "evaluation_path": primary.get("evaluation_path") or "upstream_api",
        "formal_eligible": formal_eligible,
        "execution_complete": execution_complete,
        "coverage_complete": coverage_complete,
        "infra_clean": infra_clean,
        "protocol_accounted": True,
        "claim_usable_for_full_vs_zero": formal_eligible,
        "claim_status": "FORMAL_ELIGIBLE" if formal_eligible else "PARTIAL_OR_INFRA_INCOMPLETE",
        "pool": pool_meta,
        "coverage": {
            "planned": n_planned,
            "completed": int(coverage.get("completed") or primary.get("n_queries") or 0),
            "missing": n_missing,
        },
        "settings": summaries,
        "note": (
            "Base-model Harness-1 API eval. Not an SR-OPD/CISPO training run. "
            "final_answer_recall is gold-document recall from curated set, not answer accuracy."
        ),
    }
    (out / "FOUR_CELL_OFFICIAL_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for summary in summaries:
        print(format_summary_table(str(summary.get("setting") or component_id), summary), flush=True)
    return payload


def write_eval_outputs(
    out: Path,
    *,
    component_id: str,
    summaries: list[dict[str, Any]],
    adapter_audits: list[dict[str, Any]],
    pool_meta: dict[str, Any],
    runtime_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    reload = write_reload_audit(out / "ADAPTER_RELOAD_AUDIT.json", adapter_audits)
    claim_usable = False
    claim_status = "NOT_USABLE_FOR_FULL_VS_ZERO"
    if runtime_audit:
        from trim.eval.runtime_effect_audit import write_runtime_audit

        write_runtime_audit(out, runtime_audit)
        claim_usable = bool(runtime_audit.get("claim_usable_for_full_vs_zero"))
        claim_status = str(runtime_audit.get("claim_status") or claim_status)
        (out / "CLAIM.json").write_text(
            json.dumps(
                {
                    "claim_usable_for_full_vs_zero": claim_usable,
                    "claim_status": claim_status,
                    "reason": runtime_audit.get("reason"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    status = "SR_OPD_CISPO_FOUR_CELL_EVAL"
    if runtime_audit and not runtime_audit.get("pass"):
        status = "RUNTIME_EFFECT_AUDIT_FAILED"
    payload = {
        "status": status,
        "component": component_id,
        "opd_loss": "sr_opd_ce",
        "rl_loss_fn": "cispo",
        "legacy_tool_token_kl_hook_used": False,
        "protocol_complete_rl_opd": True,
        "claim_usable_for_full_vs_zero": claim_usable,
        "claim_status": claim_status,
        "pool": pool_meta,
        "settings": summaries,
        "adapter_reload": reload,
        "runtime_effect_audit": runtime_audit,
    }
    (out / "FOUR_CELL_OFFICIAL_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for summary in summaries:
        print(format_summary_table(str(summary.get("setting") or component_id), summary), flush=True)
    return payload


def load_eval_pool(manifest: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if manifest is not None:
        return load_official_384(manifest=manifest)
    _train, test_rows, meta = load_bcplus_830_split()
    return test_rows, meta


def audit_adapter_map(adapter_map: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for cell, path in adapter_map.items():
        if not path:
            rows.append({"cell": cell, "adapter_dir": None, "reload_ready": cell == "before", "exists": False})
            continue
        rows.append(audit_saved_adapter(Path(path), cell=cell))
    return rows
