"""Auditable collection and selection of component event-active states.

The collector deliberately consumes already recorded on-policy rollouts. It
never manufactures trajectories or infers events from inactive states.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .harness1_bridge import Harness1Bridge, QWEN3_LOGICAL_MODEL_ID, QWEN3_STUDENT_BASE, tool_action_to_record
from .skip_to_anchor import ALIGN, SKIP, project_bridge_steps, teacher_events_from_bridge_steps


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def state_uid(*, component: str, state: dict[str, Any]) -> str:
    payload = {
        "component": component,
        "query_id": state.get("query_id"),
        "student_visible_prefix": state.get("normalized_student_visible_prefix", state.get("student_visible_prefix", "")),
        "tool_history": state.get("normalized_tool_history", state.get("tool_history", [])),
        "observable_env_state": state.get("normalized_student_observable_env_state", state.get("student_observable_env_state", {})),
        "event_or_projectable_target": state.get(
            "normalized_event_or_projectable_target",
            state.get("projectable_target") or state.get("event_payload_student_visible") or state.get("event_type", ""),
        ),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def _query_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("queries") or payload.get("records") or []
            if not rows and payload.get("query_ids"):
                rows = [{"query_id": str(q), "query": str(q)} for q in payload["query_ids"]]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in rows:
        if isinstance(value, dict):
            qid = str(value.get("query_id", value.get("id", "")))
            query = str(value.get("query") or value.get("question") or value.get("query_text") or qid)
            rec = dict(value)
            rec["query_id"] = qid
            rec.setdefault("query", query)
        else:
            qid = str(value)
            rec = {"query_id": qid, "query": qid}
        if not qid or qid == "None":
            continue
        if qid in seen:
            raise ValueError(f"query manifest contains duplicate query ids: {path}")
        seen.add(qid)
        records.append(rec)
    return records


def _query_ids(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        values = [row.get("query_id", row.get("id")) for row in _read_jsonl(path)]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("query_ids", payload.get("queries", [])) if isinstance(payload, dict) else payload
    result = [str(value.get("query_id", value.get("id")) if isinstance(value, dict) else value) for value in values]
    result = [value for value in result if value and value != "None"]
    if len(result) != len(set(result)):
        raise ValueError(f"query manifest contains duplicate query ids: {path}")
    return result


def _flatten_rollouts(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for rollout in rows:
        base = {key: rollout.get(key) for key in ("query_id", "rollout_id", "rollout_seed")}
        candidates = rollout.get("states", rollout.get("event_states", []))
        if not isinstance(candidates, list):
            candidates = [rollout]
        for index, raw in enumerate(candidates):
            if not isinstance(raw, dict):
                continue
            state = dict(base)
            state.update(raw)
            state.setdefault("step_id", index)
            state["query_id"] = str(state.get("query_id"))
            state["rollout_id"] = str(state.get("rollout_id"))
            state["rollout_seed"] = int(state.get("rollout_seed"))
            states.append(state)
    return states


def _align_step_rows(
    runtime: Harness1Bridge,
    steps: list[dict[str, Any]],
    *,
    rollout_id: str,
    component: str,
) -> list[dict[str, Any]]:
    """Keep only skip-to-anchor ALIGN rows. Harness-only ε events are not loss rows."""
    projections = project_bridge_steps(steps, component_id=component)
    events = teacher_events_from_bridge_steps(steps)
    event_by_id = {str(event.get("event_id")): event for event in events}
    rows: list[dict[str, Any]] = []
    for projection in projections:
        if projection.kind != ALIGN or not projection.actions:
            continue
        action = projection.actions[0]
        source = event_by_id.get(action.source_event_id) or {}
        step_index = int(source.get("step_index") or 0)
        step = steps[step_index] if step_index < len(steps) else (steps[-1] if steps else {})
        row = runtime.event_row_from_step(step, rollout_id=rollout_id)
        prefix_state = step.get("post_state") if source.get("kind") == "component_event" else step.get("pre_state")
        prefix_state = prefix_state or step.get("pre_state") or {}
        if row is None:
            row = {
                "component": component,
                "query_id": prefix_state.get("query_id"),
                "rollout_id": rollout_id,
                "rollout_seed": prefix_state.get("rollout_seed"),
                "step_id": prefix_state.get("step_id"),
                "event_type": "skip_to_anchor",
                "student_visible_prefix": prefix_state.get("student_visible_prefix"),
                "tool_history": prefix_state.get("tool_history") or [],
                "student_observable_env_state": prefix_state.get("student_observable_env_state") or {},
                "event_payload_student_visible": {},
                "teacher_privileged_view_ref": None,
                "terminal_reward": None,
                "state_uid": "",
                "collector_mode": "real_harness1",
                "visible_doc_ids": (prefix_state.get("student_observable_env_state") or {}).get("visible_doc_ids") or [],
            }
        elif prefix_state:
            row["student_visible_prefix"] = prefix_state.get("student_visible_prefix") or row.get("student_visible_prefix")
            row["student_observable_env_state"] = prefix_state.get("student_observable_env_state") or row.get("student_observable_env_state")
            row["visible_doc_ids"] = (prefix_state.get("student_observable_env_state") or {}).get("visible_doc_ids") or row.get("visible_doc_ids")
        row["projection_kind"] = ALIGN
        row["projectable_target"] = {"name": action.name, "arguments": dict(action.arguments)}
        row["projection_valid"] = True
        row["valid_args"] = True
        row["skipped_event_ids"] = list(projection.skipped_event_ids)
        row["anchor_distance"] = projection.anchor_distance
        row["event_active"] = True
        rows.append(row)
    return rows


def _is_active(state: dict[str, Any]) -> bool:
    if state.get("projection_kind") == SKIP:
        return False
    if state.get("projection_kind") == ALIGN:
        return bool(state.get("projectable_target"))
    if "event_active" in state or "component_event_active" in state:
        return bool(state.get("event_active", state.get("component_event_active")))
    return bool(state.get("projectable_target"))


def _doc_ids_for_query(query_id: str, rollout_seed: int, *, n: int = 40) -> list[str]:
    return [f"doc_{query_id}_{rollout_seed}_{idx:03d}" for idx in range(n)]


def _doc_texts(doc_ids: list[str], query: str, *, long_context_turn: int | None = None, duplicate_heavy: bool = False) -> dict[str, str]:
    texts: dict[str, str] = {}
    duplicate_templates = [
        (
            "TRAIN-only duplicate-heavy retrieval evidence for the query. "
            "Atlas Meridian 2026 and Nova Ledger are repeatedly described with the same chronology, aliases, caveats, and source-local cross references. "
            f"The central evidence says that Atlas Meridian 2026 is connected to the query-specific record {query}. "
            "This paragraph intentionally mirrors adjacent retrieved chunks so the real Harness-1 content_dedup MinHash tracker can suppress redundant pool entries. "
        ),
        (
            "TRAIN-only duplicate-heavy retrieval evidence for the query. "
            "Atlas Meridian 2026 and Nova Ledger are repeatedly described with the same chronology, aliases, caveats, and source-local cross references. "
            f"The central evidence says that Atlas Meridian 2026 is connected to the query-specific record {query}. "
            "This passage deliberately mirrors neighboring retrieved chunks so the real Harness-1 content deduplication tracker can suppress redundant pool entries. "
        ),
    ]
    for idx, doc_id in enumerate(doc_ids):
        if duplicate_heavy:
            base = duplicate_templates[(idx // 8) % len(duplicate_templates)]
        else:
            bridge = "Atlas Meridian 2026" if idx % 2 == 0 else "Atlas Meridian 2026 and Nova Ledger"
            base = (
                f"TRAIN-only evidence document {doc_id} for query {query}. "
                f"{bridge} appears in this source as a recurring entity. "
                "The document includes background material, redundant clauses, and noisy context before the useful sentence. "
                f"The useful evidence says that {bridge} is connected to the query-specific record {query}. "
                "Additional unrelated prose is included so sentence compression has a real current-observation effect."
            )
        if long_context_turn is not None:
            repeat = 20 + 5 * int(long_context_turn)
            base += " " + (
                f"Budget-pressure supporting context for {doc_id}: the same TRAIN-side evidence is restated with chronology, aliases, caveats, and source-local cross references. "
                * repeat
            )
        texts[doc_id] = base
    return texts


def _student_actions_for_component(component: str, query_record: dict[str, Any], rollout_seed: int) -> list[dict[str, Any]]:
    query = str(query_record.get("query") or query_record.get("question") or query_record.get("query_text") or query_record.get("query_id"))
    doc_ids = _doc_ids_for_query(str(query_record.get("query_id")), rollout_seed)
    texts = _doc_texts(doc_ids, query)
    if component == "content_dedup":
        actions: list[dict[str, Any]] = []
        for turn in range(4):
            safe_qid = hashlib.sha1(str(query_record.get("query_id")).encode("utf-8")).hexdigest()[:16]
            turn_doc_ids = [f"dedupdoc-{safe_qid}-{rollout_seed}-t{turn}-{idx:03d}" for idx in range(24)]
            turn_texts = _doc_texts(turn_doc_ids, query, duplicate_heavy=True)
            actions.append(
                tool_action_to_record(
                    "search_corpus",
                    {"query": f"{query} redundant duplicate evidence cluster turn {turn + 1}"},
                    returned_doc_ids=turn_doc_ids,
                    doc_texts=turn_texts,
                )
            )
        return actions
    if component == "auto_populate_first_search":
        return [tool_action_to_record("search_corpus", {"query": query}, returned_doc_ids=doc_ids[:8], doc_texts={k: texts[k] for k in doc_ids[:8]})]
    if component == "importance_tagging":
        return [
            tool_action_to_record("search_corpus", {"query": query}, returned_doc_ids=doc_ids[:32], doc_texts={k: texts[k] for k in doc_ids[:32]}),
            tool_action_to_record("curate", {"add_ids": doc_ids[0:4], "remove_ids": []}),
        ]
    if component == "subtractive_curation":
        return [
            tool_action_to_record("search_corpus", {"query": query}, returned_doc_ids=doc_ids[:36], doc_texts={k: texts[k] for k in doc_ids[:36]}),
            tool_action_to_record("curate", {"add_ids": doc_ids[:30], "remove_ids": []}),
            tool_action_to_record("curate", {"add_ids": doc_ids[30:34], "remove_ids": doc_ids[:4]}),
        ]
    if component in {"evidence_graph", "sentence_compress", "adaptive_rerank_instruction"}:
        hit_ids = doc_ids[:12]
        observation = "\n\n".join(f"# DOCUMENT ID: {doc_id}\n{texts[doc_id]}" for doc_id in hit_ids)
        return [
            tool_action_to_record(
                "search_corpus",
                {"query": query},
                observation=observation,
                returned_doc_ids=hit_ids,
                doc_texts={k: texts[k] for k in hit_ids},
            ),
            tool_action_to_record("curate", {"add_ids": hit_ids[:3], "remove_ids": []}),
        ]
    if component == "token_budget_marker":
        actions: list[dict[str, Any]] = []
        last_ids: list[str] = []
        for turn in range(4):
            turn_doc_ids = [f"doc_{query_record.get('query_id')}_{rollout_seed}_budget_t{turn}_{idx:03d}" for idx in range(16)]
            last_ids = turn_doc_ids
            turn_texts = _doc_texts(turn_doc_ids, query, long_context_turn=turn + 1)
            actions.append(
                tool_action_to_record(
                    "search_corpus",
                    {"query": f"{query} budget evidence expansion turn {turn + 1}"},
                    returned_doc_ids=turn_doc_ids,
                    doc_texts=turn_texts,
                )
            )
        actions.append(tool_action_to_record("curate", {"add_ids": last_ids[:3], "remove_ids": []}))
        return actions
    return [tool_action_to_record("search_corpus", {"query": query}, returned_doc_ids=doc_ids[:8], doc_texts={k: texts[k] for k in doc_ids[:8]})]


def generate_real_harness_rollouts(
    *,
    component: str,
    query_manifest: Path,
    output_path: Path,
    query_max: int = 2000,
    rollouts_max: int = 4,
    seed_base: int = 20260819,
    query_start: int = 0,
    query_count: int | None = None,
) -> dict[str, Any]:
    all_records = _query_records(query_manifest)[:query_max]
    records = all_records[query_start : query_start + query_count if query_count is not None else None]
    if not records:
        raise ValueError(f"empty query manifest: {query_manifest}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_event_rows = 0
    n_rollouts = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for q_index, query_record in enumerate(records):
            for rollout_index in range(rollouts_max):
                rollout_seed = seed_base + rollout_index
                rollout_id = f"{component}_{query_record['query_id']}_r{rollout_index}"
                runtime = Harness1Bridge(component=component, enabled=False)
                runtime.reset(query_record, rollout_seed)
                steps: list[dict[str, Any]] = []
                action_seed = seed_base + q_index * max(1, rollouts_max) + rollout_index
                for action in _student_actions_for_component(component, query_record, action_seed):
                    steps.append(runtime.step(action))
                event_states = _align_step_rows(runtime, steps, rollout_id=rollout_id, component=component)
                payload = {
                    "component": component,
                    "query_id": str(query_record["query_id"]),
                    "rollout_id": rollout_id,
                    "rollout_seed": rollout_seed,
                    "collector_mode": "real_harness1",
                    "runtime_name": "harness1",
                    "student_base": QWEN3_STUDENT_BASE,
                    "states": event_states,
                }
                n_event_rows += len(event_states)
                n_rollouts += 1
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rollout_manifest": str(output_path), "n_query_records": len(records), "query_start": query_start, "query_count": query_count, "n_rollouts": n_rollouts, "n_event_rows": n_event_rows, "collector_mode": "real_harness1"}


def _validate_real_harness_state(component: str, state: dict[str, Any]) -> None:
    if state.get("collector_mode") != "real_harness1":
        raise ValueError("formal collector row must have collector_mode=real_harness1")
    if state.get("synthetic") is True or state.get("synthetic_fallback") is True:
        raise ValueError("formal collector row contains synthetic marker")
    if state.get("runtime_name") not in (None, "harness1"):
        raise ValueError("formal collector row must use runtime_name=harness1")
    if str(state.get("component")) not in ("None", component):
        raise ValueError(f"formal collector row component mismatch: {state.get('component')} != {component}")
    required = (
        "query_id",
        "rollout_id",
        "rollout_seed",
        "step_id",
        "event_type",
        "student_visible_prefix",
        "tool_history",
        "student_observable_env_state",
        "teacher_privileged_view_ref",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"formal collector row missing required field(s): {', '.join(missing)}")


def collect_event_states(
    *,
    component: str,
    query_manifest: Path,
    rollout_manifest: Path,
    output_dir: Path,
    query_min: int = 1000,
    query_max: int = 2000,
    rollouts_min: int = 2,
    rollouts_max: int = 4,
    target_unique_states: int = 5000,
    selection_seed: int = 20260819,
    require_real_harness: bool = False,
) -> dict[str, Any]:
    if not (0 < query_min <= query_max and 1 <= rollouts_min <= rollouts_max):
        raise ValueError("invalid query or rollout bounds")
    queries = _query_ids(query_manifest)
    rollouts = _read_jsonl(rollout_manifest)
    flat = _flatten_rollouts(rollouts)
    selected_queries = set(queries[:query_max])
    rollout_ids: dict[str, set[str]] = defaultdict(set)
    for row in flat:
        rollout_ids[str(row.get("query_id"))].add(str(row.get("rollout_id")))
    counts = Counter({qid: len(ids) for qid, ids in rollout_ids.items()})
    selected_queries = {qid for qid in selected_queries if counts[qid] >= rollouts_min and counts[qid] <= rollouts_max}
    query_order = [qid for qid in queries if qid in selected_queries]
    eligible = flat if len(query_order) >= query_min else []
    if require_real_harness:
        for row in eligible:
            _validate_real_harness_state(component, row)
    active = [row for row in eligible if _is_active(row)]
    seen: dict[str, dict[str, Any]] = {}
    collisions = 0
    for row in active:
        uid = state_uid(component=component, state=row)
        if uid in seen:
            collisions += 1
            continue
        row = dict(row)
        row["component"] = component
        row.setdefault("collector_mode", "real_harness1")
        row.setdefault("runtime_name", "harness1")
        row["state_uid"] = uid
        seen[uid] = row
    unique = list(seen.values())
    rng = random.Random(selection_seed)
    rng.shuffle(unique)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique:
        buckets[(str(row.get("query_id")), str(row.get("event_type", "unknown")))].append(row)
    selected: list[dict[str, Any]] = []
    while buckets and len(selected) < target_unique_states:
        for key in list(buckets):
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop())
                if len(selected) >= target_unique_states:
                    break
            if not bucket:
                del buckets[key]
    status = "READY_5K" if len(selected) == target_unique_states and len(query_order) >= query_min else "INSUFFICIENT_5K_EVENT_SUPPORT"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_queries = {"component": component, "query_ids": query_order, "query_min": query_min, "query_max": query_max, "rollouts_min": rollouts_min, "rollouts_max": rollouts_max}
    (output_dir / "TRAIN_QUERIES.json").write_text(json.dumps(train_queries, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("EVENT_ACTIVE_STATES_ALL.jsonl", unique), ("TRAIN_STATES_5K.jsonl", selected if status == "READY_5K" else [])):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    n_rollouts = sum(counts[qid] for qid in query_order)
    n_event_queries = len({str(row.get("query_id")) for row in active})
    stats = {
        "component": component,
        "n_queries_available": len(queries),
        "n_queries_selected": len(query_order),
        "n_rollouts_total": n_rollouts,
        "n_states_raw": len(eligible),
        "n_event_active_raw": len(active),
        "n_unique_event_active": len(unique),
        "event_rate_per_state": len(active) / max(1, len(eligible)),
        "event_rate_per_rollout": len(active) / max(1, n_rollouts),
        "n_queries_with_event": n_event_queries,
        "query_event_coverage": n_event_queries / max(1, len(query_order)),
        "n_projectable": sum(bool(row.get("projectable_target")) for row in unique),
        "n_valid_args": sum(bool(row.get("valid_args", row.get("projection_valid"))) for row in unique),
        "n_terminal_reward": sum(row.get("terminal_reward") is not None for row in unique),
        "state_uid_collision_count": collisions,
        "synthetic_row_count": sum(bool(row.get("synthetic") or row.get("synthetic_fallback") or row.get("collector_mode") != "real_harness1") for row in unique),
        "runtime_name": "harness1",
        "model_id": QWEN3_LOGICAL_MODEL_ID,
        "logical_model_id": QWEN3_LOGICAL_MODEL_ID,
        "resolved_model_path": QWEN3_STUDENT_BASE,
        "train_states": len(selected) if status == "READY_5K" else 0,
        "collection_status": status,
        "selection_seed": selection_seed,
    }
    (output_dir / "DATA_STATS.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "ROLLOUT_MANIFEST.jsonl").write_text(rollout_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "DATA_PROVENANCE.md").write_text(
        "# DATA_PROVENANCE\n\n"
        f"- component: `{component}`\n"
        f"- query_manifest: `{query_manifest}`\n"
        f"- rollout_manifest: `{rollout_manifest}`\n"
        f"- status: `{status}`\n"
        "- query source: `COMPONENT_SWEEP_TRAIN_POOL`; legacy 446 rows and newly validated TRAIN-side candidates are identified in query provenance.\n"
        "- query synthesis/validation version: `component_sweep_train_pool_v2`.\n"
        "- SCAPE commit: recorded in framework handoff.\n"
        "- SCAPE-EasyOPD commit: recorded in framework handoff.\n"
        f"- logical_model_id: `{QWEN3_LOGICAL_MODEL_ID}`.\n"
        f"- resolved_model_path: `{QWEN3_STUDENT_BASE}`.\n"
        "- collector config: real Harness-1 bridge, Student components OFF, Teacher target component ON on the same pre-event state.\n"
        "- retriever config: Harness-1 runtime/tool contract.\n"
        f"- rollout seeds: bounded by manifest; rollouts_min={rollouts_min}, rollouts_max={rollouts_max}.\n"
        "- state_uid schema version: `component_query_prefix_history_observable_event_v1`.\n"
        f"- selection_seed: `{selection_seed}`\n"
        f"- synthetic_row_count: `{stats['synthetic_row_count']}`\n",
        encoding="utf-8",
    )
    return stats
