"""Harness-1 closed-loop quality metrics plus e2e / model / harness timing.

Official document-level formulas (gold_docids wins when present):
  recall            = |selected ∩ gold| / |gold|
  trajectory_recall = |observed ∪ selected ∩ gold| / |gold|
  precision         = |selected ∩ gold| / |selected|
  evidence_recall   = |selected ∩ evidence| / |evidence|   (diagnostic only)
  final_answer_recall is the same selected-vs-gold quantity, NOT answer-string accuracy.
  f1 / f_beta use official precision/recall.
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

# Child stages. Parents stay in TIMING_KEYS. Summaries must not add a child into its parent.
# Missing keys on old traces stay missing; do not impute 0.
TIMING_SCHEMA_VERSION = "harness_g_timing_v1"
STAGE_TIMING_KEYS: tuple[str, ...] = (
    "queue_wait_sec",
    "prompt_sec",
    "parse_sec",
    "execute_sec",
    "menu_sec",
    "init_retrieve_sec",
    "lookup_candidates_sec",
    "lookup_rank_sec",
    "frontier_sec",
    "bridge_sec",
    "serialize_sec",
    "freeze_sec",
    "snapshot_sec",
    "model_sec_allocated",
)

METRIC_SPLIT_KEYS: tuple[str, ...] = (
    "gold_recall",
    "trajectory_gold_recall",
    "evidence_recall",
    "trajectory_evidence_recall",
    "n_gold",
    "n_gold_selected",
    "n_gold_observed",
    "n_evidence",
    "n_evidence_selected",
    "n_evidence_observed",
)

TRACE_EXPORT_KEYS: tuple[str, ...] = HARNESS1_QUALITY_KEYS + TIMING_KEYS + METRIC_SPLIT_KEYS + (
    "n_tool_calls",
    "n_search_calls",
    "elapsed_s",
    "search_query",
    "n_turns",
    "ended",
    "runtime_effects",
    "auto_seed",
    "observed_docids",
    "observed_sids",
    "selected_docids",
    "selected_sids",
    "tool_history",
    "successful_select",
    "unique_selected_docs",
    "n_generated",
    "n_parse_ok",
    "n_schema_ok",
    "n_menu_ok",
    "n_execution_ok",
    "n_attempt_events",
    "n_infrastructure_failures",
    "gold_docids",
    "evidence_docids",
    "official_split",
    "candidate_docids",
    "displayed_sids",
    "prompt_visible_sids",
    "legacy_retrieved_union_docids",
    "attempt_events",
    "termination_kind",
    "termination_diagnostics",
    "component_funnel",
    "graph_scope",
    "graph_fingerprint",
    "graph_enabled",
    "graph_index_path",
    "graph_lookups",
    "real_search_calls",
    "timing_schema_version",
    "model_sec_kind",
    "generate_batches",
    "stage_counts",
    "stage_cpu_sec",
) + STAGE_TIMING_KEYS

# Summary aliases matching the original eval table labels.
SUMMARY_LABELS: dict[str, str] = {
    "recall": "Official Recall (selected vs gold)",
    "trajectory_recall": "Official Trajectory Recall (observed vs gold)",
    "final_answer_recall": "Gold-document Recall (alias of official selected recall)",
    "trajectory_fa_recall": "Gold Trajectory Recall (alias of official observed recall)",
    "gold_recall": "Gold Recall",
    "trajectory_gold_recall": "Gold Trajectory Recall",
    "evidence_recall": "Evidence/Support Recall (diagnostic)",
    "trajectory_evidence_recall": "Evidence Trajectory Recall (diagnostic)",
    "precision": "Precision",
    "f1": "F1",
    "f_beta": "F-beta (β=2)",
    "reward": "Reward",
    "n_curated": "Curated Docs",
    "n_pool": "Pool Docs",
    "num_turns": "Turns",
    "tool_diversity": "Tool Diversity",
    "legal_action_rate": "Legal Action Rate (name, macro)",
    "legal_action_rate_micro": "Legal Action Rate (name, micro)",
    "execution_ok_rate_micro": "Execution-OK Rate (micro)",
    "test_evidence_recall_at_5": "Initial BM25 Recall@5",
    "test_evidence_recall_at_100": "Initial BM25 Recall@100",
    "initial_bm25_recall_at_5": "Initial BM25 Evidence Recall@5 (compat alias)",
    "initial_bm25_recall_at_100": "Initial BM25 Evidence Recall@100 (compat alias)",
    "initial_bm25_gold_recall_at_5": "Initial BM25 Gold Recall@5",
    "initial_bm25_gold_recall_at_100": "Initial BM25 Gold Recall@100",
    "initial_bm25_evidence_recall_at_5": "Initial BM25 Evidence Recall@5",
    "initial_bm25_evidence_recall_at_100": "Initial BM25 Evidence Recall@100",
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
    finished_at: float | None = None
    model_sec: float = 0.0
    harness_sec: float = 0.0
    freeze_sec: float = 0.0
    menu_sec: float = 0.0
    parse_sec: float = 0.0
    execute_sec: float = 0.0
    prompt_sec: float = 0.0
    snapshot_sec: float = 0.0
    queue_wait_sec: float = 0.0
    init_retrieve_sec: float = 0.0
    lookup_candidates_sec: float = 0.0
    lookup_rank_sec: float = 0.0
    frontier_sec: float = 0.0
    bridge_sec: float = 0.0
    serialize_sec: float = 0.0
    peak_rss_mb: float | None = None
    generate_batches: list[dict[str, Any]] = field(default_factory=list)
    stage_cpu_sec: dict[str, float] = field(default_factory=dict)

    def add_model(self, dt: float) -> None:
        self.model_sec += max(0.0, float(dt))

    def note_generate_batch(self, batch_id: str, wall_sec: float, n_requests: int) -> None:
        self.generate_batches.append(
            {
                "batch_id": str(batch_id),
                "wall_sec": max(0.0, float(wall_sec)),
                "n_requests": int(n_requests),
            }
        )

    def add_harness(self, dt: float) -> None:
        self.harness_sec += max(0.0, float(dt))

    def mark_finished(self) -> None:
        if self.finished_at is None:
            self.finished_at = time.perf_counter()

    def snapshot(self) -> dict[str, float]:
        end = self.finished_at if self.finished_at is not None else time.perf_counter()
        e2e = max(0.0, end - self.e2e_start)
        return {
            "e2e_sec": e2e,
            "model_sec": self.model_sec,
            "harness_sec": self.harness_sec,
            "freeze_sec": self.freeze_sec,
            "menu_sec": self.menu_sec,
            "parse_sec": self.parse_sec,
            "execute_sec": self.execute_sec,
            "prompt_sec": self.prompt_sec,
            "snapshot_sec": self.snapshot_sec,
            "queue_wait_sec": self.queue_wait_sec,
            "init_retrieve_sec": self.init_retrieve_sec,
            "lookup_candidates_sec": self.lookup_candidates_sec,
            "lookup_rank_sec": self.lookup_rank_sec,
            "frontier_sec": self.frontier_sec,
            "bridge_sec": self.bridge_sec,
            "serialize_sec": self.serialize_sec,
            "model_sec_allocated": self.model_sec,
            "model_sec_kind": "allocated_share",
            "timing_schema_version": TIMING_SCHEMA_VERSION,
            "generate_batches": list(self.generate_batches),
            "stage_cpu_sec": dict(self.stage_cpu_sec),
            "peak_rss_mb": self.peak_rss_mb,
            "elapsed_s": e2e,
        }


def _add_stage_wall(timing: EpisodeTiming, kind: str, dt: float) -> None:
    dt = max(0.0, float(dt))
    if kind == "model":
        timing.add_model(dt)
    elif kind == "freeze":
        timing.freeze_sec += dt
    elif kind == "menu":
        timing.menu_sec += dt
    elif kind == "parse":
        timing.parse_sec += dt
    elif kind == "execute":
        # Inclusive parent is harness_sec; execute_sec is the child interval.
        timing.execute_sec += dt
        timing.add_harness(dt)
    elif kind == "prompt":
        timing.prompt_sec += dt
    elif kind == "snapshot":
        timing.snapshot_sec += dt
    elif kind == "queue_wait":
        timing.queue_wait_sec += dt
    elif kind == "init_retrieve":
        timing.init_retrieve_sec += dt
    elif kind == "lookup_candidates":
        timing.lookup_candidates_sec += dt
    elif kind == "lookup_rank":
        timing.lookup_rank_sec += dt
    elif kind == "frontier":
        timing.frontier_sec += dt
    elif kind == "bridge":
        timing.bridge_sec += dt
    elif kind == "serialize":
        timing.serialize_sec += dt
    else:
        timing.add_harness(dt)


@contextmanager
def timed_section(timing: EpisodeTiming | None, kind: str) -> Iterator[None]:
    t0 = time.perf_counter()
    cpu0 = time.thread_time()
    try:
        yield
    finally:
        if timing is None:
            return
        dt = time.perf_counter() - t0
        cpu = max(0.0, time.thread_time() - cpu0)
        _add_stage_wall(timing, kind, dt)
        timing.stage_cpu_sec[kind] = float(timing.stage_cpu_sec.get(kind) or 0.0) + cpu


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
    selected = _id_set(state.get("selected_docids")) or _id_set(state.get("curated"))
    curated = selected
    pool = _id_set(state.get("pool"))
    observed = _id_set(state.get("observed_docids"))
    if observed:
        traversed = observed | curated
    else:
        traversed = pool | curated
    evidence = _id_set(row.get("evidence_docids"))
    gold = _id_set(row.get("gold_docids"))
    g_state = bool(state.get("sentences") is not None or state.get("graph") is not None or state.get("action_map"))
    if g_state:
        official_relevant = gold
        missing_gold = not gold
    else:
        official_relevant = gold or evidence
        missing_gold = False

    recall = set_recall(curated, official_relevant)
    precision = set_precision(curated, official_relevant)
    traj_recall = set_recall(traversed, official_relevant)
    gold_recall = set_recall(curated, gold) if gold else recall
    traj_gold = set_recall(traversed, gold) if gold else traj_recall
    evidence_recall = set_recall(curated, evidence) if evidence else 0.0
    traj_evidence = set_recall(traversed, evidence) if evidence else 0.0
    # final_answer_recall is selected-vs-gold, not an independent answer-string score.
    fa_recall = gold_recall
    traj_fa = traj_gold
    n_turns = len(list(tool_names))
    n_curate = sum(1 for name in tool_names if name == "curate")
    n_unique = len({name for name in tool_names if name and name not in {"unknown", "None", "truncated"}})
    invalid = int(state.get("invalid_tools") or 0)
    error = invalid > 0 or (bool(valids) and not any(valids))
    ended = bool(state.get("ended"))
    max_turns_reached = 0.0
    if max_turns is not None and n_turns >= int(max_turns) and not ended:
        max_turns_reached = 1.0
    hist = list(state.get("tool_history") or [])
    attempts = list(state.get("attempt_events") or [])
    n_generated = len(attempts) if attempts else n_turns
    if attempts:
        n_parse_ok = sum(1 for a in attempts if a.get("parse_ok"))
        n_schema_ok = sum(1 for a in attempts if a.get("schema_ok"))
        n_menu_ok = sum(1 for a in attempts if a.get("menu_ok"))
        n_exec_ok = sum(1 for a in attempts if a.get("execution_ok"))
    else:
        n_parse_ok = sum(1 for h in hist if h.get("parse_ok"))
        n_schema_ok = sum(1 for h in hist if h.get("schema_ok"))
        n_menu_ok = sum(1 for h in hist if h.get("target_ok") or h.get("menu_ok"))
        n_exec_ok = sum(1 for h in hist if h.get("execution_ok"))
    n_infra = sum(1 for a in attempts if str(a.get("error_class") or a.get("error_type") or "") in {"infrastructure", "infrastructure_failure"} or a.get("error_stage") == "execute" and a.get("traceback_ref"))
    n_select_ok = sum(1 for h in hist if str(h.get("name") or "") == "select" and h.get("execution_ok"))

    payload: dict[str, Any] = {
        "recall": recall,
        "gold_recall": gold_recall,
        "trajectory_gold_recall": traj_gold,
        "evidence_recall": evidence_recall,
        "trajectory_evidence_recall": traj_evidence,
        "n_gold": float(len(gold)),
        "n_gold_selected": float(len(curated & gold)),
        "n_gold_observed": float(len(traversed & gold)),
        "n_evidence": float(len(evidence)),
        "n_evidence_selected": float(len(curated & evidence)),
        "n_evidence_observed": float(len(traversed & evidence)),
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
        "total_curate_calls": 0.0 if g_state else float(n_curate),
        "curate_rate": 0.0 if g_state else n_curate / max(n_turns, 1),
        "used_curate": 0.0 if g_state else (1.0 if n_curate > 0 or curated else 0.0),
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
        "runtime_effects": dict(state.get("runtime_effects") or {}),
        "auto_seed": list(state.get("auto_seed") or []) if state.get("auto_seed") else None,
        "observed_docids": list(state.get("observed_docids") or []),
        "observed_sids": list(state.get("observed_sids") or []),
        "selected_docids": list(state.get("selected_docids") or []),
        "selected_sids": list(state.get("selected_sids") or []),
        "tool_history": hist,
        "successful_select": float(n_select_ok) if g_state else None,
        "unique_selected_docs": float(len(_id_set(state.get("selected_docids")))) if g_state else None,
        "n_generated": n_generated,
        "n_parse_ok": n_parse_ok,
        "n_schema_ok": n_schema_ok,
        "n_menu_ok": n_menu_ok,
        "n_execution_ok": n_exec_ok,
        "n_attempt_events": len(attempts),
        "n_infrastructure_failures": n_infra,
        "gold_docids": sorted(gold),
        "evidence_docids": sorted(evidence),
        "official_split": row.get("official_split"),
        "missing_gold_docids": bool(missing_gold),
        "candidate_docids": list(state.get("candidate_docids") or []),
        "displayed_sids": list(state.get("displayed_sids") or []),
        "prompt_visible_sids": list(state.get("prompt_visible_sids") or []),
        "legacy_retrieved_union_docids": list(state.get("legacy_retrieved_union_docids") or []),
        "attempt_events": attempts,
        "termination_kind": state.get("termination_kind") or (state.get("end_reason") if ended else None),
        "termination_diagnostics": dict(state.get("termination_diagnostics") or {}),
        "component_funnel": dict(state.get("component_funnel") or {}),
        "graph_scope": state.get("graph_scope"),
        "graph_fingerprint": state.get("graph_fingerprint"),
        "graph_enabled": bool(state.get("graph_enabled")),
        "graph_index_path": state.get("graph_index_path"),
        "graph_lookups": int(state.get("graph_lookups") or 0),
        "real_search_calls": int(state.get("real_search_calls") or 0),
        "prefetch_docs": int(state.get("prefetch_docs") or 0),
    }
    stage_counts = state.get("_stage_counts") or state.get("stage_counts")
    if isinstance(stage_counts, Mapping) and stage_counts:
        payload["stage_counts"] = {str(k): dict(v) if isinstance(v, Mapping) else v for k, v in stage_counts.items()}
    if timing:
        payload.update({k: float(timing[k]) for k in TIMING_KEYS if k in timing and timing[k] is not None})
        for extra_key in STAGE_TIMING_KEYS:
            if extra_key in timing and timing[extra_key] is not None:
                payload[extra_key] = float(timing[extra_key])
        if "elapsed_s" in timing and timing["elapsed_s"] is not None:
            payload["elapsed_s"] = float(timing["elapsed_s"])
        if timing.get("model_sec_kind"):
            payload["model_sec_kind"] = timing["model_sec_kind"]
        if timing.get("timing_schema_version"):
            payload["timing_schema_version"] = timing["timing_schema_version"]
        if timing.get("generate_batches"):
            payload["generate_batches"] = list(timing["generate_batches"])
        if isinstance(timing.get("stage_cpu_sec"), Mapping) and timing.get("stage_cpu_sec"):
            payload["stage_cpu_sec"] = dict(timing["stage_cpu_sec"])
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
    for key in HARNESS1_QUALITY_KEYS + METRIC_SPLIT_KEYS:
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
        vals = [float(t[key]) for t in traces if key in t and t.get(key) is not None]
        out[f"mean_{key}"] = _mean(vals) if vals else None
        out[f"p50_{key}"] = _percentile(vals, 50) if vals else None
        out[f"p95_{key}"] = _percentile(vals, 95) if vals else None
        out[f"sum_{key}"] = sum(vals) if vals else None
    for key in STAGE_TIMING_KEYS:
        if not any(key in t and t.get(key) is not None for t in traces):
            continue
        vals = [float(t[key]) for t in traces if key in t and t.get(key) is not None]
        out[f"mean_{key}"] = _mean(vals) if vals else None
        out[f"sum_{key}"] = sum(vals) if vals else None
    batch_walls: dict[str, float] = {}
    for trace in traces:
        for batch in trace.get("generate_batches") or []:
            if not isinstance(batch, Mapping):
                continue
            batch_id = str(batch.get("batch_id") or "")
            if not batch_id:
                continue
            batch_walls[batch_id] = float(batch.get("wall_sec") or 0.0)
    if batch_walls:
        out["n_generate_batches"] = len(batch_walls)
        out["sum_generate_batch_wall_sec"] = sum(batch_walls.values())
    out["elapsed_s"] = out.get("mean_e2e_sec")
    schema = next((t.get("timing_schema_version") for t in traces if t.get("timing_schema_version")), None)
    if schema:
        out["timing_schema_version"] = schema
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
        "gold_recall",
        "evidence_recall",
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
