"""Harness-1 closed-loop quality metrics plus e2e / model / harness timing.

Document-level formulas match SearchDataset (BrowseComp-Plus):
  recall            = |curated ∩ relevant| / |relevant|
  trajectory_recall = |pool ∪ curated ∩ relevant| / |relevant|
  precision         = |curated ∩ relevant| / |curated|
  final_answer_*    uses gold / final-answer document ids
  f1                = harmonic mean of precision and recall
  f_beta            = Fβ with β=2 (Harness-1 v3)
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

RECALL_BETA = 2.0

# Per-query fields that mirror evaluate_harness1.py + compute_reward + SearchTask.
HARNESS1_QUALITY_KEYS: tuple[str, ...] = (
    "recall",
    "precision",
    "f1",
    "f_beta",
    "trajectory_recall",
    "final_answer_recall",
    "trajectory_fa_recall",
    "final_answer_found",
    "reward",
    "n_curated",
    "n_pool",
    "num_turns",
    "tool_diversity",
    "total_curate_calls",
    "curate_rate",
    "used_curate",
    "no_error",
    "error",
    "max_turns_reached",
)

TIMING_KEYS: tuple[str, ...] = ("e2e_sec", "model_sec", "harness_sec")

TRACE_EXPORT_KEYS: tuple[str, ...] = HARNESS1_QUALITY_KEYS + TIMING_KEYS + (
    "gold_recall",
    "n_tool_calls",
    "n_search_calls",
    "elapsed_s",
    "search_query",
    "n_turns",
    "ended",
)

# Summary aliases matching the original eval table labels.
SUMMARY_LABELS: dict[str, str] = {
    "recall": "Curated Recall",
    "trajectory_recall": "Pool Recall",
    "final_answer_recall": "Final-Answer Recall",
    "trajectory_fa_recall": "Pool FA Recall",
    "precision": "Precision",
    "f1": "F1",
    "f_beta": "F-beta (β=2)",
    "reward": "Reward",
    "n_curated": "Curated Docs",
    "n_pool": "Pool Docs",
    "num_turns": "Turns",
    "tool_diversity": "Tool Diversity",
    "legal_action_rate": "Legal Action Rate",
    "test_evidence_recall_at_5": "Evidence Recall@5",
    "test_evidence_recall_at_100": "Evidence Recall@100",
    "mean_tool_calls_per_query": "Tool Calls",
    "tool_search_cost": "Search Calls",
    "mean_e2e_sec": "E2E Time (s)",
    "mean_model_sec": "Model Time (s)",
    "mean_harness_sec": "Harness Time (s)",
    "error_rate": "Error Rate",
}


def _id_set(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, Mapping):
        return {str(k) for k in value if str(k)}
    if isinstance(value, (list, tuple, set)):
        return {str(x) for x in value if str(x)}
    return {str(value)}


def set_recall(got: Iterable[str], relevant: Iterable[str]) -> float:
    gold = {str(x) for x in relevant if str(x)}
    if not gold:
        return 0.0
    have = {str(x) for x in got if str(x)}
    return len(gold & have) / len(gold)


def set_precision(got: Iterable[str], relevant: Iterable[str]) -> float:
    have = {str(x) for x in got if str(x)}
    if not have:
        return 0.0
    gold = {str(x) for x in relevant if str(x)}
    return len(gold & have) / len(have)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def f_beta_score(precision: float, recall: float, *, beta: float = RECALL_BETA) -> float:
    if precision + recall <= 0:
        return 0.0
    beta_sq = beta * beta
    return (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall)


@dataclass
class EpisodeTiming:
    e2e_start: float = field(default_factory=time.perf_counter)
    model_sec: float = 0.0
    harness_sec: float = 0.0

    def add_model(self, dt: float) -> None:
        self.model_sec += max(0.0, float(dt))

    def add_harness(self, dt: float) -> None:
        self.harness_sec += max(0.0, float(dt))

    def snapshot(self) -> dict[str, float]:
        e2e = max(0.0, time.perf_counter() - self.e2e_start)
        return {
            "e2e_sec": e2e,
            "model_sec": self.model_sec,
            "harness_sec": self.harness_sec,
            "elapsed_s": e2e,
        }


@contextmanager
def timed_section(timing: EpisodeTiming | None, kind: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if timing is None:
            return
        dt = time.perf_counter() - t0
        if kind == "model":
            timing.add_model(dt)
        else:
            timing.add_harness(dt)


def episode_quality_metrics(
    state: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    tool_names: Sequence[str],
    valids: Sequence[bool],
    reward: float,
    max_turns: int | None = None,
    timing: Mapping[str, float] | None = None,
    actions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    curated = _id_set(state.get("curated"))
    pool = _id_set(state.get("pool"))
    traversed = pool | curated
    relevant = _id_set(row.get("evidence_docids")) or _id_set(row.get("gold_docids"))
    gold = _id_set(row.get("gold_docids"))

    recall = set_recall(curated, relevant)
    precision = set_precision(curated, relevant)
    traj_recall = set_recall(traversed, relevant)
    fa_recall = set_recall(curated, gold) if gold else 0.0
    traj_fa = set_recall(traversed, gold) if gold else 0.0
    n_turns = len(list(tool_names))
    n_curate = sum(1 for name in tool_names if name == "curate")
    n_unique = len({name for name in tool_names if name and name not in {"unknown", "None"}})
    invalid = int(state.get("invalid_tools") or 0)
    error = invalid > 0 or (bool(valids) and not any(valids))
    ended = bool(state.get("ended"))
    max_turns_reached = 0.0
    if max_turns is not None and n_turns >= int(max_turns) and not ended:
        max_turns_reached = 1.0

    payload: dict[str, Any] = {
        "recall": recall,
        "gold_recall": recall,
        "precision": precision,
        "f1": f1_score(precision, recall),
        "f_beta": f_beta_score(precision, recall),
        "trajectory_recall": traj_recall,
        "final_answer_recall": fa_recall,
        "trajectory_fa_recall": traj_fa,
        "final_answer_found": 1.0 if fa_recall > 0 else 0.0,
        "reward": float(reward),
        "n_curated": float(len(curated)),
        "n_pool": float(len(pool)),
        "num_turns": float(n_turns),
        "n_turns": n_turns,
        "tool_diversity": float(n_unique),
        "total_curate_calls": float(n_curate),
        "curate_rate": n_curate / max(n_turns, 1),
        "used_curate": 1.0 if n_curate > 0 or curated else 0.0,
        "no_error": 0.0 if error else 1.0,
        "error": bool(error),
        "max_turns_reached": max_turns_reached,
        "n_tool_calls": int(state.get("n_tool_calls") or n_turns),
        "n_search_calls": int(state.get("n_search_calls") or 0),
        "names": list(tool_names),
        "ended": ended,
        "search_query": next(
            (
                str((a.get("arguments") or {}).get("query") or "")
                for a in (actions or [])
                if a.get("name") == "search_corpus" and (a.get("arguments") or {}).get("query")
            ),
            str(row.get("query") or ""),
        ),
        "prune_accuracy": None,
        "rerank_recall": None,
        "rerank_dropped_relevant_count": None,
    }
    if timing:
        payload.update({k: float(timing[k]) for k in TIMING_KEYS if k in timing})
        if "elapsed_s" in timing:
            payload["elapsed_s"] = float(timing["elapsed_s"])
    return payload


def trace_fields(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {key: stats[key] for key in TRACE_EXPORT_KEYS if key in stats}


def _mean(vals: list[float]) -> float:
    return sum(vals) / max(1, len(vals))


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    idx = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[idx]


def summarize_quality_and_timing(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(traces)
    out: dict[str, Any] = {}
    for key in HARNESS1_QUALITY_KEYS:
        if key == "error":
            continue
        vals = [float(t.get(key) or 0.0) for t in traces]
        out[key] = _mean(vals) if n else 0.0
    errors = [bool(t.get("error")) for t in traces]
    out["errors"] = int(sum(1 for e in errors if e))
    out["error_rate"] = (sum(1 for e in errors if e) / n) if n else 0.0
    out["used_curate_rate"] = _mean([float(t.get("used_curate") or 0.0) for t in traces]) if n else 0.0
    out["recall_gt0_rate"] = _mean([1.0 if float(t.get("recall") or 0.0) > 0 else 0.0 for t in traces]) if n else 0.0
    for key in TIMING_KEYS:
        vals = [float(t[key]) for t in traces if t.get(key) is not None]
        out[f"mean_{key}"] = _mean(vals) if vals else None
        out[f"p50_{key}"] = _percentile(vals, 50) if vals else None
        out[f"p95_{key}"] = _percentile(vals, 95) if vals else None
        out[f"sum_{key}"] = sum(vals) if vals else None
    out["elapsed_s"] = out.get("mean_e2e_sec")
    return out


def format_summary_table(name: str, summary: Mapping[str, Any]) -> str:
    n = int(summary.get("n_queries") or 0)
    errors = summary.get("errors")
    lines = [
        "",
        "=" * 80,
        f"  {name}",
        "=" * 80,
        f"  n: {n}  errors: {errors}",
    ]
    order = [
        "recall",
        "trajectory_recall",
        "final_answer_recall",
        "trajectory_fa_recall",
        "precision",
        "f1",
        "f_beta",
        "reward",
        "n_curated",
        "n_pool",
        "num_turns",
        "tool_diversity",
        "legal_action_rate",
        "test_evidence_recall_at_5",
        "test_evidence_recall_at_100",
        "mean_tool_calls_per_query",
        "tool_search_cost",
        "mean_e2e_sec",
        "mean_model_sec",
        "mean_harness_sec",
        "error_rate",
    ]
    for key in order:
        val = summary.get(key)
        if val is None:
            continue
        label = SUMMARY_LABELS.get(key, key)
        lines.append(f"  {label:<28} {float(val):>10.4f}")
    extra = [
        ("p50_e2e_sec", "E2E p50 (s)"),
        ("p95_e2e_sec", "E2E p95 (s)"),
        ("p50_model_sec", "Model p50 (s)"),
        ("p95_model_sec", "Model p95 (s)"),
        ("p50_harness_sec", "Harness p50 (s)"),
        ("p95_harness_sec", "Harness p95 (s)"),
        ("sum_e2e_sec", "E2E sum (s)"),
        ("sum_model_sec", "Model sum (s)"),
        ("sum_harness_sec", "Harness sum (s)"),
    ]
    for key, label in extra:
        val = summary.get(key)
        if val is None:
            continue
        lines.append(f"  {label:<28} {float(val):>10.4f}")
    lines.append("=" * 80)
    return "\n".join(lines)
