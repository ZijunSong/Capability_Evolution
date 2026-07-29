"""E0 global and capability-specific metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from harness.capability.capability_id import CapabilityId, parse_capability_id

GLOBAL_METRICS = (
    "recall",
    "trajectory_recall",
    "final_answer_recall",
    "precision",
    "reward",
    "turns",
    "n_curated",
    "n_pool",
    "search_calls",
    "open_calls",
    "errors",
)


def _episode_metric(ep: dict[str, Any], metric: str) -> float:
    if metric == "errors":
        return 1.0 if ep.get("error") else 0.0
    if metric == "search_calls":
        return float(ep.get("search_calls", ep.get("metrics", {}).get("search_calls", 0)))
    if metric == "open_calls":
        return float(ep.get("open_calls", ep.get("metrics", {}).get("open_calls", 0)))
    return float(ep.get(metric, 0.0))


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {m: 0.0 for m in GLOBAL_METRICS}
    out: dict[str, float] = {}
    for m in GLOBAL_METRICS:
        out[m] = sum(_episode_metric(ep, m) for ep in episodes) / len(episodes)
    return out


def capability_specific_metrics(
    capability_id: str,
    episodes: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    cap = parse_capability_id(capability_id)
    events = events or []
    n = max(1, len(episodes))

    if cap == CapabilityId.DUPLICATE_EVIDENCE:
        dup_rates = [float(ep.get("dup_curate_rate", 0.0)) for ep in episodes]
        repeated = [float(ep.get("repeated_evidence_rate", 0.0)) for ep in episodes]
        unique_ratio = [float(ep.get("unique_curated_ratio", 0.0)) for ep in episodes]
        redundant = [float(ep.get("redundant_context_rate", 0.0)) for ep in episodes]
        return {
            "duplicate_curate_rate": sum(dup_rates) / n,
            "repeated_evidence_rate": sum(repeated) / n,
            "unique_curated_evidence_ratio": sum(unique_ratio) / n,
            "redundant_context_rate": sum(redundant) / n,
        }

    if cap == CapabilityId.STOP_DECISION:
        premature = [float(ep.get("premature_stop", 0.0)) for ep in episodes]
        valid_stop = [float(ep.get("valid_stop_recall", 0.0)) for ep in episodes]
        over_search = [float(ep.get("over_search", 0.0)) for ep in episodes]
        extra_calls = [float(ep.get("extra_calls_after_sufficient", 0.0)) for ep in episodes]
        turns = [float(ep.get("turns", 0.0)) for ep in episodes]
        return {
            "premature_stop_rate": sum(premature) / n,
            "valid_stop_recall": sum(valid_stop) / n,
            "over_search_rate": sum(over_search) / n,
            "mean_turns": sum(turns) / n,
            "mean_extra_calls_after_sufficient_evidence": sum(extra_calls) / n,
        }

    if cap == CapabilityId.EVIDENCE_CURATION:
        return {
            "supporting_evidence_recall": sum(
                float(ep.get("supporting_evidence_recall", 0.0)) for ep in episodes
            )
            / n,
            "evidence_precision": sum(float(ep.get("evidence_precision", 0.0)) for ep in episodes)
            / n,
            "unique_useful_curated_count": sum(
                float(ep.get("unique_useful_curated", 0.0)) for ep in episodes
            )
            / n,
            "curated_set_size": sum(float(ep.get("n_curated", 0.0)) for ep in episodes) / n,
        }

    if cap in {CapabilityId.VERIFICATION_DECISION, CapabilityId.EXTERNAL_VERIFICATION}:
        verify_calls = [float(ep.get("verification_calls", 0.0)) for ep in episodes]
        verified_rate = [float(ep.get("verified_claim_rate", 0.0)) for ep in episodes]
        conflict = [float(ep.get("conflict_resolution_rate", 0.0)) for ep in episodes]
        unsupported = [float(ep.get("unsupported_answer_rate", 0.0)) for ep in episodes]
        return {
            "verification_calls": sum(verify_calls) / n,
            "verified_claim_rate": sum(verified_rate) / n,
            "conflict_resolution_rate": sum(conflict) / n,
            "unsupported_answer_rate": sum(unsupported) / n,
            "external_call_count": sum(
                float(ev.get("external_call", 0)) for ev in events
            )
            / max(1, len(events)),
            "new_information_count": sum(
                float(ev.get("new_information", 0)) for ev in events
            )
            / max(1, len(events)),
            "downstream_use_count": sum(
                float(ev.get("downstream_use", 0)) for ev in events
            )
            / max(1, len(events)),
        }

    if cap == CapabilityId.DETERMINISTIC_TRUNCATION:
        return {
            "context_overflow_rate": sum(
                float(ep.get("context_overflow", 0.0)) for ep in episodes
            )
            / n,
            "truncation_events": sum(
                float(ep.get("truncation_events", 0.0)) for ep in episodes
            )
            / n,
        }

    return {}


def enrich_episode_metrics(
    capability_id: str,
    episode: dict[str, Any],
    turn_records: list[Any] | None = None,
) -> dict[str, Any]:
    """Add capability-specific per-episode fields from rollout result."""
    out = dict(episode)
    metrics = dict(out.get("metrics") or {})
    turns = int(out.get("turns", 0))
    n_curated = int(out.get("n_curated", 0))
    n_pool = int(out.get("n_pool", 0))
    early_blocks = int(out.get("early_end_blocks", 0))

    cap = parse_capability_id(capability_id)
    if cap == CapabilityId.DUPLICATE_EVIDENCE:
        dup_blocked = int(metrics.get("dup_skipped", 0))
        out["dup_curate_rate"] = dup_blocked / max(1, n_curated + dup_blocked)
        out["repeated_evidence_rate"] = float(metrics.get("repeated_evidence_rate", 0.0))
        out["unique_curated_ratio"] = n_curated / max(1, n_pool)
        out["redundant_context_rate"] = float(metrics.get("redundant_context_rate", 0.0))

    if cap == CapabilityId.STOP_DECISION:
        out["premature_stop"] = 1.0 if turns < 8 and out.get("recall", 0) == 0 else 0.0
        out["valid_stop_recall"] = float(out.get("recall", 0.0))
        out["over_search"] = 1.0 if turns >= 34 else 0.0
        out["extra_calls_after_sufficient"] = max(0, early_blocks)

    if cap in {CapabilityId.VERIFICATION_DECISION, CapabilityId.EXTERNAL_VERIFICATION}:
        out["verification_calls"] = float(metrics.get("verify_calls", 0.0))
        out["verified_claim_rate"] = float(metrics.get("verified_claim_rate", 0.0))
        out["unsupported_answer_rate"] = 1.0 if out.get("recall", 0) == 0 else 0.0

    if cap == CapabilityId.DETERMINISTIC_TRUNCATION:
        out["context_overflow"] = float(metrics.get("context_overflow", 0.0))
        out["truncation_events"] = float(metrics.get("truncation_events", 0.0))

    out["search_calls"] = float(metrics.get("search_calls", metrics.get("n_search", 0.0)))
    out["open_calls"] = float(metrics.get("open_calls", metrics.get("n_open", 0.0)))
    return out


def episodes_by_query(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        qid = str(ep.get("query_id", ""))
        if qid:
            out[qid] = ep
    return out
