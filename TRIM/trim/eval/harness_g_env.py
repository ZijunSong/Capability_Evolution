"""TRIM-native Harness-G episode over a graph index + optional searcher.

Always-on runtime: INIT / SELECT / LOOKUP / PAGE / ANSWER. Advanced components
(answer_with, bridges, synonyms, neighbors, hybrid INIT, lexical hints)
are mask-gated. Public protocol correctness (menu refresh, JSON, order,
readability) is shared by zero and all.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import traceback
from typing import Any, Mapping

from trim.adapters.harness_g_components import RUNTIME_TOOLS
from trim.adapters.harness_profiles import is_harness_g, zero_mask_for
from trim.eval.harness_g_contract import is_corpus_scope
from trim.eval.harness_g_graph import (
    GRAPH_MAX_ENTITY_DOCS,
    GRAPH_MAX_ENTITY_SIDS,
    HarnessGGraphIndex,
    _ENTITY_RE,
    begin_scan_bucket,
    build_graph_from_documents,
    canonical_eid,
    drain_scan_bucket,
    is_metadata_only_line,
    keep_entity_surface,
    lexical_score,
    take_lookup_stats,
    mixquery_text,
    normalize_entity_surface,
    text_to_sentence_parts,
)

_BAD_LOOKUP = re.compile(
    r"^(\d+|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|american|british|french|german|chinese|"
    r"japanese|russian|indian|canadian)$",
    re.I,
)
MAX_IDENTICAL_FAILURES = 3
SELECT_LOOKUP_K = 8
NAV_LOOKUP_K = 4
CORPUS_NAV_LOOKUP_K = 32
VISIBLE_K = 6
GRAPH_MAX_FRONTIER_SIDS = 2000

# Backward-compatible names used by existing tests.
_WM_PREVIEW_CHARS = 10**9
_SELECT_PREVIEW_CHARS = 10**9
_text_to_sentence_parts = text_to_sentence_parts
_is_metadata_only_line = is_metadata_only_line


def _graph(state: Mapping[str, Any]) -> HarnessGGraphIndex:
    graph = state.get("graph")
    if isinstance(graph, HarnessGGraphIndex):
        return graph
    return build_graph_from_documents(state.get("doc_store") or {})


def _sync_graph_views(state: dict[str, Any]) -> None:
    graph = _graph(state)
    state["graph"] = graph
    state["sentences"] = graph.sentences
    state["entities"] = graph.entities
    state["graph_scope"] = graph.scope
    state["graph_fingerprint"] = graph.content_fingerprint()
    state["graph_enabled"] = is_corpus_scope(graph.scope)


def _sentences_from_store(doc_store: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return build_graph_from_documents(doc_store).sentences


def _entities_from_sentences(sentences: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_doc: dict[str, list[str]] = {}
    for sent in sentences.values():
        did = str(sent.get("doc_id") or "d")
        by_doc.setdefault(did, []).append(str(sent.get("text") or ""))
    store = {did: {"id": did, "text": " ".join(parts)} for did, parts in by_doc.items() if did}
    return build_graph_from_documents(store).entities


def _rebuild_index(state: dict[str, Any]) -> None:
    graph = _graph(state)
    if not is_corpus_scope(getattr(graph, "scope", None)):
        graph.ingest_documents(state.get("doc_store") or {})
    _sync_graph_views(state)


def _merge_search_hits(state: dict[str, Any], hits: list[Any]) -> int:
    store = dict(state.get("doc_store") or {})
    added = 0
    incoming: dict[str, Any] = {}
    for hit in hits or []:
        did = str(getattr(hit, "docid", "") or "")
        if not did:
            continue
        text = str(getattr(hit, "text", "") or "")
        score = float(getattr(hit, "score", 0.0) or 0.0)
        if did not in store:
            added += 1
        prev = store.get(did) or {}
        prev_text = str(prev.get("text") or "") if isinstance(prev, dict) else str(prev)
        if len(text) >= len(prev_text):
            rec = {"id": did, "text": text, "score": score}
            store[did] = rec
            incoming[did] = rec
    state["doc_store"] = store
    graph = _graph(state)
    if not is_corpus_scope(getattr(graph, "scope", None)):
        graph.ingest_documents(incoming or store)
        _sync_graph_views(state)
    return added


def _sort_sids_by_idx(sids: list[str], sentences: Mapping[str, Mapping[str, Any]]) -> list[str]:
    def _key(sid: str) -> tuple[int, str]:
        sent = sentences.get(sid) or {}
        return (int(sent.get("idx", 0)), sid)

    return sorted(sids, key=_key)


def _rank_sids(query: str, sentences: Mapping[str, Mapping[str, Any]], k: int) -> list[str]:
    scored: list[tuple[float, str]] = []
    for sid, sent in sentences.items():
        scored.append((lexical_score(query, str(sent.get("text") or "")), sid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [sid for _, sid in scored[:k]]


def _rrf_fuse(lists: list[list[str]], *, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for lst in lists:
        seen: set[str] = set()
        rank = 0
        for sid in lst:
            if sid in seen:
                continue
            seen.add(sid)
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank + 1)
            rank += 1
    return sorted(scores.keys(), key=lambda s: (-scores[s], s))


def _hybrid_rank_sids(
    query: str,
    sentences: Mapping[str, Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, Any]],
    k: int,
    *,
    doc_order: list[str] | None = None,
) -> list[str]:
    lexical = _rank_sids(query, sentences, k=max(k * 3, 12))
    q_low = query.lower()
    entity_hits: list[str] = []
    for rec in entities.values():
        if str(rec.get("surface") or "").lower() in q_low:
            entity_hits.extend(rec.get("sids") or [])
    bm25_sids: list[str] = []
    if doc_order:
        by_doc: dict[str, list[str]] = {}
        for sid, sent in sentences.items():
            by_doc.setdefault(str(sent.get("doc_id")), []).append(sid)
        for did in doc_order:
            bm25_sids.extend(_sort_sids_by_idx(list(by_doc.get(did) or []), sentences))
    fused = _rrf_fuse([lexical, entity_hits, bm25_sids])
    return fused[:k]


def new_state(
    query: str,
    doc_store: dict[str, Any],
    *,
    harness_mask: Mapping[str, bool] | None = None,
    visible_k: int = 6,
    graph_index: HarnessGGraphIndex | None = None,
    graph_index_path: str | None = None,
) -> dict[str, Any]:
    del visible_k
    mask = dict(harness_mask) if harness_mask is not None else zero_mask_for("Harness-G")
    if graph_index is not None:
        if is_corpus_scope(graph_index.scope):
            # Official corpus graph is complete. BM25 snippets must not drop/rebuild docs.
            graph = graph_index
        else:
            graph = graph_index.clone_overlay()
            graph.ingest_documents(doc_store or {})
    else:
        graph = build_graph_from_documents(doc_store or {}, scope="episode_doc_store")
    path = graph_index_path or getattr(graph_index, "source_path", None) or getattr(graph, "source_path", None)
    return {
        "query": query,
        "step": 0,
        "ended": False,
        "end_reason": None,
        "initialized": False,
        "doc_store": doc_store or {},
        "graph": graph,
        "graph_scope": graph.scope,
        "graph_fingerprint": graph.content_fingerprint(),
        "graph_enabled": is_corpus_scope(graph.scope),
        "graph_index_path": path,
        "sentences": graph.sentences,
        "entities": graph.entities,
        "visible_sids": [],
        "selected_sids": [],
        "visited_eids": [],
        "frontier_eids": [],
        "action_map": {},
        "harness_mask": mask,
        "tool_history": [],
        "turn_events": [],
        "n_tool_calls": 0,
        "invalid_tools": 0,
        "pool": {},
        "curated": {},
        "search_count": 0,
        "n_search_calls": 0,
        "real_search_calls": 0,
        "graph_lookups": 0,
        "prefetch_docs": len(doc_store or {}),
        "observed_docids": [],
        "observed_sids": [],
        "displayed_sids": [],
        "displayed_docids": [],
        "prompt_visible_sids": [],
        "candidate_docids": [],
        "candidate_sids": [],
        "legacy_retrieved_union_docids": [],
        "selected_docids": [],
        "graph_expanded_docids": [],
        "lookup_candidates": [],
        "lookup_window_offset": 0,
        "nav_frontier_eids": [],
        "attempt_events": [],
        "component_funnel": {},
        "runtime_effects": {},
        "last_mixquery": None,
        "last_mixquery_meta": {},
        "fail_counts": {},
        "protocol_failure": False,
        "infrastructure_failure": False,
        "termination_kind": None,
    }


def _mask_on(state: Mapping[str, Any], component_id: str) -> bool:
    return bool((state.get("harness_mask") or {}).get(component_id, False))


def _is_bad_lookup(entity: Mapping[str, Any]) -> bool:
    surface = str(entity.get("surface") or "")
    return bool(_BAD_LOOKUP.match(surface.strip()))


def _select_score(query: str, text: str) -> float:
    return lexical_score(query, text)


def _note_effect(state: dict[str, Any], key: str, n: int = 1) -> None:
    effects = dict(state.get("runtime_effects") or {})
    effects[key] = int(effects.get(key) or 0) + int(n)
    state["runtime_effects"] = effects


def _parent_docid(state: Mapping[str, Any], sid: str) -> str:
    graph = state.get("graph")
    if isinstance(graph, HarnessGGraphIndex):
        did = graph.parent_docid(sid)
        if did:
            return did
    sent = (state.get("sentences") or {}).get(sid) or {}
    return str(sent.get("parent_docid") or sent.get("doc_id") or (sid.split(":")[0] if ":" in sid else sid))


def allowed_menu_pairs(state: Mapping[str, Any]) -> set[tuple[str, str | None]]:
    pairs: set[tuple[str, str | None]] = set()
    for action in (state.get("action_map") or {}).values():
        typ = str(action.get("type") or action.get("name") or "").upper()
        if typ in {"SELECT", "ANSWER_WITH"} and action.get("sid"):
            pairs.add((typ, str(action["sid"])))
        elif typ == "LOOKUP" and action.get("eid"):
            pairs.add((typ, str(action["eid"])))
        elif typ == "PAGE":
            pairs.add((typ, str(action.get("direction") or "next")))
        elif typ in {"ANSWER", "INIT"}:
            pairs.add((typ, None))
    return pairs


def allowed_menu_targets(state: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Executable sid/eid targets from the current action menu."""
    sids: set[str] = set()
    eids: set[str] = set()
    for typ, target in allowed_menu_pairs(state):
        if typ in {"SELECT", "ANSWER_WITH"} and target:
            sids.add(target)
        if typ == "LOOKUP" and target:
            eids.add(target)
    return sids, eids


def _menu_hint(state: Mapping[str, Any], *, want: str | None = None) -> str:
    items: list[str] = []
    for aid, action in (state.get("action_map") or {}).items():
        typ = str(action.get("type") or "")
        if want and typ.upper() != want.upper() and typ.lower() != want.lower():
            continue
        if action.get("sid"):
            items.append(f'{aid} {typ} sid="{action["sid"]}"')
        elif action.get("eid"):
            items.append(f'{aid} {typ} eid="{action["eid"]}"')
        else:
            items.append(f"{aid} {typ}")
        if len(items) >= 8:
            break
    if not items:
        return "Legal alternative: call init if not initialized, otherwise choose an actions entry."
    return "Legal alternatives: " + "; ".join(items)


def _frontier_from_sids(state: Mapping[str, Any], sids: list[str]) -> list[str]:
    graph = _graph(state)
    frontier: list[str] = []
    for sid in sids:
        for rec in graph.get_entities_for_sentence(sid):
            eid = str(rec.get("eid") or "")
            if eid and eid not in frontier:
                frontier.append(eid)
    return frontier


def _lookup_specs_from_visible(state: dict[str, Any], sids: list[str]) -> list[str]:
    return _frontier_from_sids(state, sids)


def _note_funnel(state: dict[str, Any], component_id: str, **fields: Any) -> None:
    funnel = dict(state.get("component_funnel") or {})
    rec = dict(funnel.get(component_id) or {})
    rec.setdefault("enabled", _mask_on(state, component_id))
    for key, value in fields.items():
        if key.endswith("_delta") or key in {"invoked", "eligible"}:
            rec[key] = int(rec.get(key) or 0) + int(value or 0)
        else:
            rec[key] = value
    funnel[component_id] = rec
    state["component_funnel"] = funnel


def _candidate_doc_list(state: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in ("candidate_docids", "initial_bm25_docids", "graph_expanded_docids", "displayed_docids"):
        for did in state.get(key) or []:
            did = str(did or "")
            if did and did not in seen:
                seen.add(did)
                out.append(did)
    return out


def _observed_doc_list(state: Mapping[str, Any]) -> list[str]:
    """Docs eligible for navigation menus. Uses candidates, not hidden-as-observed."""
    return _candidate_doc_list(state)


def _query_eids(query: str) -> set[str]:
    out: set[str] = set()
    for surface in _ENTITY_RE.findall(query or ""):
        surface = normalize_entity_surface(surface)
        if keep_entity_surface(surface):
            out.add(canonical_eid(surface))
    return out


def _stage_add(state: dict[str, Any], key: str, dt: float) -> None:
    bucket = state.setdefault("_stage_sec", {})
    bucket[key] = float(bucket.get(key) or 0.0) + max(0.0, float(dt))


def _merge_scan_counts(state: dict[str, Any]) -> None:
    scans = drain_scan_bucket()
    if not scans:
        return
    counts = state.setdefault("_stage_counts", {})
    for kind, rec in scans.items():
        dest = counts.setdefault(kind, {})
        for key, value in rec.items():
            dest[key] = int(dest.get(key) or 0) + int(value)


def _frontier_from_observed_docs(state: Mapping[str, Any], *, cap: int) -> list[str]:
    t0 = time.perf_counter()
    begin_scan_bucket()
    graph = _graph(state)
    query = str(state.get("query") or "")
    selected_sids = list(state.get("selected_sids") or [])
    selected_text = " ".join(
        str(((state.get("sentences") or {}).get(sid) or {}).get("text") or "")
        for sid in selected_sids[-4:]
    )
    mix = f"{query} {selected_text}".strip()
    docs = list(_candidate_doc_list(state))
    counted = [str(x) for x in (state.get("_frontier_counted_docs") or [])]
    local_sids = {str(k): int(v) for k, v in (state.get("_frontier_entity_counts") or {}).items()}
    doc_set = set(docs)
    if any(did not in doc_set for did in counted):
        local_sids = {}
        counted = []
    counted_set = set(counted)
    new_docs = [did for did in docs if did not in counted_set]
    for did in new_docs:
        for eid, n_mentions in graph.entity_mention_counts(did).items():
            local_sids[str(eid)] = local_sids.get(str(eid), 0) + int(n_mentions)
        counted.append(did)
    if isinstance(state, dict):
        state["_frontier_counted_docs"] = list(counted)
        state["_frontier_entity_counts"] = dict(local_sids)
        state["_frontier_last_new_docs"] = len(new_docs)
    scored: list[tuple[float, str, dict[str, float]]] = []
    for eid, n_local in local_sids.items():
        rec = graph.entities.get(eid) or {}
        n_sids = len(rec.get("sids") or [])
        if not rec or n_sids > GRAPH_MAX_FRONTIER_SIDS:
            continue
        surface = str(rec.get("surface") or "")
        q_score = lexical_score(mix, surface)
        ev_score = lexical_score(selected_text, surface) if selected_text else 0.0
        hub_pen = min(4.0, math.log1p(max(0, n_sids)) / 3.0)
        external = 1.0 if n_sids > n_local else 0.0
        score = 3.0 * q_score + 1.5 * ev_score + 0.25 * external - 0.4 * hub_pen
        scored.append((score, eid, {"q": q_score, "ev": ev_score, "hub": hub_pen, "ext": external}))
    scored.sort(key=lambda x: (-x[0], x[1]))
    # Return the full ranked pool. Eligibility (visited/invalid) is applied in
    # build_action_map before the menu budget, so visited-prefix rows can refill.
    eids = [eid for _, eid, _ in scored]
    if isinstance(state, dict):
        state["_frontier_cache"] = {
            "eids": list(eids),
            "n_candidates": len(scored),
            "n_new_docs": len(new_docs),
            "rank_breakdown": [
                {"eid": eid, "score": score, **parts} for score, eid, parts in scored[: max(1, int(cap))]
            ],
        }
        _merge_scan_counts(state)
        _stage_add(state, "frontier_sec", time.perf_counter() - t0)
    return eids


def _attach_lexical_hint(state: dict[str, Any], menu: dict[str, dict[str, Any]]) -> None:
    if not _mask_on(state, "snc_frontier"):
        return
    query = str(state.get("query") or "")
    sentences = state.get("sentences") or {}
    entities = state.get("entities") or {}
    _note_effect(state, "lexical_frontier_hint_applied")
    for action in menu.values():
        typ = action.get("type")
        if typ == "SELECT":
            text = str((sentences.get(action.get("sid")) or {}).get("text") or "")
            hint = round(_select_score(query, text), 4)
        elif typ == "LOOKUP":
            rec = entities.get(action.get("eid")) or {}
            hint = round(_select_score(query, str(rec.get("surface") or "")), 4)
        else:
            hint = 0.0
        action["lexical_frontier_hint"] = hint
        action["snc_preview"] = hint  # compatibility alias; not official SNC


def build_action_map(
    state: dict[str, Any],
    *,
    include_answer: bool,
    lookup_cap: int | None = None,
    lookup_eids: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    t_menu = time.perf_counter()
    bridge_before = float((state.get("_stage_sec") or {}).get("bridge_sec") or 0.0)
    menu: dict[str, dict[str, Any]] = {}
    n = 0
    selected = set(state.get("selected_sids") or [])
    sentences = state.get("sentences") or {}
    entities = state.get("entities") or {}
    cap = int(lookup_cap if lookup_cap is not None else (SELECT_LOOKUP_K if include_answer else NAV_LOOKUP_K))

    for sid in state.get("visible_sids") or []:
        if sid in selected or sid not in sentences:
            continue
        menu[f"A{n}"] = {"type": "SELECT", "sid": sid, "name": "select"}
        n += 1
        if _mask_on(state, "answer_with"):
            menu[f"A{n}"] = {
                "type": "ANSWER_WITH",
                "sid": sid,
                "sids": [sid],
                "name": "answer_with",
                "evidence_preview": str(sentences[sid].get("text") or "")[:80],
            }
            n += 1
            _note_funnel(state, "answer_with", enabled=True, eligible=1)

    lookup_eids = list(lookup_eids if lookup_eids is not None else (state.get("frontier_eids") or []))
    pairs_before = {(str(a.get("type") or "").upper(), a.get("sid") or a.get("eid")) for a in menu.values()}
    bridges: list[dict[str, Any]] = []
    bridge_eids: list[str] = []
    if _mask_on(state, "bridge_entities"):
        graph = _graph(state)
        _note_funnel(state, "bridge_entities", enabled=True, invoked=1)
        begin_scan_bucket()
        t_bridge = time.perf_counter()
        bridges = graph.propose_bridge_entities(
            lookup_eids,
            str(state.get("query") or ""),
            state.get("selected_sids") or [],
            topm=5,
        )
        _stage_add(state, "bridge_sec", time.perf_counter() - t_bridge)
        _merge_scan_counts(state)
        _note_funnel(state, "bridge_entities", eligible=len(bridges), candidate_delta=len(bridges))
        for cand in bridges:
            eid = str(cand.get("target_eid") or cand.get("eid") or "")
            if eid:
                bridge_eids.append(eid)

    visited = set(state.get("visited_eids") or [])
    skip_reasons: dict[str, str] = {}

    def _eligible(eid: str) -> tuple[bool, str | None]:
        rec = entities.get(eid)
        if not rec:
            return False, "missing_entity"
        if _mask_on(state, "lookup_dedup") and eid in visited:
            return False, "visited"
        if _mask_on(state, "invalid_target_filter") and _is_bad_lookup(rec):
            return False, "invalid_target"
        return True, None

    reserved = 0
    if bridge_eids:
        reserved = min(2, max(1, cap // 4), len(bridge_eids), cap)
    normal_budget = max(0, cap - reserved)
    chosen: list[str] = []
    for eid in lookup_eids:
        ok, reason = _eligible(eid)
        if not ok:
            skip_reasons[eid] = str(reason)
            if reason == "invalid_target":
                _note_effect(state, "invalid_target_filtered")
                _note_funnel(state, "invalid_target_filter", invoked=1, candidate_delta=1)
            if reason == "visited":
                _note_funnel(state, "lookup_dedup", invoked=1)
            continue
        if eid in chosen:
            continue
        if len(chosen) >= normal_budget:
            continue
        chosen.append(eid)

    added_bridge_temp = 0
    added_bridge_menu = 0
    for eid in bridge_eids:
        added_bridge_temp += 1
        if eid in chosen:
            continue
        ok, reason = _eligible(eid)
        if not ok:
            skip_reasons[eid] = str(reason)
            continue
        if len(chosen) >= cap:
            continue
        chosen.append(eid)
        added_bridge_menu += 1

    if len(chosen) < cap:
        for eid in lookup_eids:
            if len(chosen) >= cap:
                break
            if eid in chosen:
                continue
            ok, reason = _eligible(eid)
            if ok:
                chosen.append(eid)

    for eid in chosen:
        rec = entities.get(eid) or {}
        menu[f"A{n}"] = {
            "type": "LOOKUP",
            "eid": eid,
            "name": "lookup",
            "entity_surface": rec.get("surface"),
        }
        n += 1

    candidates = list(state.get("lookup_candidates") or [])
    offset = int(state.get("lookup_window_offset") or 0)
    if candidates and offset + VISIBLE_K < len(candidates):
        menu[f"A{n}"] = {
            "type": "PAGE",
            "name": "page",
            "direction": "next",
            "window_offset": offset,
            "n_candidates": len(candidates),
        }
        n += 1

    if include_answer:
        menu[f"A{n}"] = {"type": "ANSWER", "name": "answer"}
        n += 1

    pairs_after = set()
    for action in menu.values():
        typ = str(action.get("type") or "").upper()
        if typ in {"SELECT", "ANSWER_WITH"} and action.get("sid"):
            pairs_after.add((typ, str(action["sid"])))
        elif typ == "LOOKUP" and action.get("eid"):
            pairs_after.add((typ, str(action["eid"])))
        elif typ in {"ANSWER", "INIT", "PAGE"}:
            pairs_after.add((typ, str(action.get("direction") or "")))
    bridge_targets = {("LOOKUP", eid) for eid in bridge_eids}
    actual_bridge_delta = len((pairs_after - pairs_before) & bridge_targets)
    if _mask_on(state, "bridge_entities"):
        _note_effect(state, "bridge_entities_candidate_delta", added_bridge_temp)
        _note_effect(state, "bridge_entities_menu_delta", actual_bridge_delta)
        _note_funnel(
            state,
            "bridge_entities",
            menu_delta=actual_bridge_delta,
            skip_reason=None if actual_bridge_delta else ("cap_full" if added_bridge_temp else "no_candidates"),
        )
    if _mask_on(state, "invalid_target_filter"):
        _note_funnel(state, "invalid_target_filter", menu_delta=len(skip_reasons), enabled=True)
    if _mask_on(state, "lookup_dedup"):
        _note_funnel(state, "lookup_dedup", enabled=True, menu_delta=sum(1 for r in skip_reasons.values() if r == "visited"))
    state["last_menu_skip_reasons"] = skip_reasons
    state["last_frontier_diagnostics"] = {
        "n_lookup_pool": len(lookup_eids),
        "n_chosen": len(chosen),
        "cap": cap,
        "reserved_bridge": reserved,
        "skip_reasons": skip_reasons,
    }
    _attach_lexical_hint(state, menu)
    if _mask_on(state, "snc_frontier"):
        _note_funnel(state, "snc_frontier", enabled=True, invoked=1, menu_delta=0)
    bridge_after = float((state.get("_stage_sec") or {}).get("bridge_sec") or 0.0)
    _stage_add(state, "menu_sec", time.perf_counter() - t_menu - (bridge_after - bridge_before))
    return menu


def _set_nav_menu(state: dict[str, Any], *, include_answer: bool, lookup_cap: int | None = None) -> None:
    corpus = is_corpus_scope(state.get("graph_scope"))
    cap = lookup_cap if lookup_cap is not None else (CORPUS_NAV_LOOKUP_K if corpus else NAV_LOOKUP_K)
    pool_cap = max(int(cap), CORPUS_NAV_LOOKUP_K) * 4
    if corpus:
        state["frontier_eids"] = _frontier_from_observed_docs(state, cap=pool_cap)
    else:
        state["frontier_eids"] = _lookup_specs_from_visible(state, list(state.get("visible_sids") or []))
    state["nav_frontier_eids"] = list(state.get("frontier_eids") or [])
    state["action_map"] = build_action_map(
        state,
        include_answer=include_answer,
        lookup_cap=cap,
        lookup_eids=list(state.get("frontier_eids") or []),
    )


def _set_select_menu(state: dict[str, Any], sid: str) -> None:
    selected_frontier = _frontier_from_sids(state, [sid])
    nav_frontier = list(state.get("nav_frontier_eids") or state.get("frontier_eids") or [])
    merged = list(dict.fromkeys(list(selected_frontier) + list(nav_frontier)))
    state["frontier_eids"] = merged
    state["action_map"] = build_action_map(
        state,
        include_answer=True,
        lookup_cap=SELECT_LOOKUP_K,
        lookup_eids=merged,
    )


def _sync_curated(state: dict[str, Any]) -> None:
    curated: dict[str, Any] = {}
    store = state.get("doc_store") or {}
    sentences = state.get("sentences") or {}
    pool: dict[str, Any] = {}
    observed_docids: list[str] = list(state.get("observed_docids") or [])
    observed_sids: list[str] = list(state.get("observed_sids") or [])
    selected_docids: list[str] = []
    observed_doc_set = set(observed_docids)
    observed_sid_set = set(observed_sids)
    selected_sid_set: set[str] = set()

    selected_sids = list(state.get("selected_sids") or [])
    deduped_selected: list[str] = []
    for sid in selected_sids:
        if sid in selected_sid_set:
            continue
        selected_sid_set.add(sid)
        deduped_selected.append(sid)
        sent = sentences.get(sid) or {}
        did = _parent_docid(state, sid) or str(sent.get("doc_id") or "")
        rec = store.get(did) or {"id": did, "text": sent.get("text") or ""}
        curated[did] = rec if isinstance(rec, dict) else {"id": did, "text": str(rec)}
        pool[did] = curated[did]
        if did and did not in selected_docids:
            selected_docids.append(did)
        if sid not in observed_sid_set:
            observed_sids.append(sid)
            observed_sid_set.add(sid)
        if did and did not in observed_doc_set:
            observed_docids.append(did)
            observed_doc_set.add(did)

    for sid in state.get("visible_sids") or []:
        sent = sentences.get(sid) or {}
        did = _parent_docid(state, sid) or str(sent.get("doc_id") or "")
        if sid not in observed_sid_set:
            observed_sids.append(sid)
            observed_sid_set.add(sid)
        if did and did not in observed_doc_set:
            observed_docids.append(did)
            observed_doc_set.add(did)
        if did not in pool:
            rec = store.get(did) or {"id": did, "text": sent.get("text") or ""}
            pool[did] = rec if isinstance(rec, dict) else {"id": did, "text": str(rec)}

    displayed_sids = list(dict.fromkeys(list(state.get("visible_sids") or []) + list(state.get("selected_sids") or [])[-8:]))
    displayed_docids = []
    displayed_doc_set: set[str] = set()
    for sid in displayed_sids:
        did = _parent_docid(state, sid)
        if did and did not in displayed_doc_set:
            displayed_docids.append(did)
            displayed_doc_set.add(did)

    candidate_docids = list(state.get("candidate_docids") or [])
    candidate_set = set(candidate_docids)
    for did in list(state.get("initial_bm25_docids") or []) + list(state.get("graph_expanded_docids") or []) + displayed_docids:
        did = str(did or "")
        if did and did not in candidate_set:
            candidate_docids.append(did)
            candidate_set.add(did)
            if did not in pool:
                rec = store.get(did) or {"id": did, "text": ""}
                pool[did] = rec if isinstance(rec, dict) else {"id": did, "text": str(rec)}

    candidate_sids = list(state.get("candidate_sids") or [])
    cand_sid_set = set(candidate_sids)
    for sid in list(state.get("visible_sids") or []) + list(row.get("sid") for row in (state.get("lookup_candidates") or []) if isinstance(row, dict)):
        sid = str(sid or "")
        if sid and sid not in cand_sid_set:
            candidate_sids.append(sid)
            cand_sid_set.add(sid)

    legacy_union = list(dict.fromkeys(list(candidate_docids) + list(observed_docids)))

    state["curated"] = curated
    state["pool"] = pool
    state["curated_ids"] = list(curated)
    state["observed_docids"] = observed_docids
    state["observed_sids"] = observed_sids
    state["displayed_sids"] = displayed_sids
    state["displayed_docids"] = displayed_docids
    state["candidate_docids"] = candidate_docids
    state["candidate_sids"] = candidate_sids
    state["legacy_retrieved_union_docids"] = legacy_union
    state["selected_docids"] = selected_docids
    state["selected_sids"] = deduped_selected


def _full_sent(state: Mapping[str, Any], sid: str) -> str:
    return str(((state.get("sentences") or {}).get(sid) or {}).get("text") or "")


def wm_text(state: dict[str, Any], *, auto_on: bool = False) -> str:
    del auto_on
    entities = state.get("entities") or {}
    selected = list(state.get("selected_sids") or [])
    visible = list(state.get("visible_sids") or [])
    n_sel = len(selected)
    shown_sel = selected[-8:]
    hidden_sel = max(0, n_sel - len(shown_sel))
    lines = [
        "[Harness-G Working Memory]",
        "Execute only the actions and targets listed under actions. Selected ids are known evidence, not SELECT targets unless they reappear as SELECT/ANSWER_WITH.",
        f"step={state.get('step', 0)} initialized={bool(state.get('initialized'))} graph_scope={state.get('graph_scope')}",
        f"n_selected={n_sel} n_visible={len(visible)}",
        f"selected (showing last {len(shown_sel)} of {n_sel}" + (f", {hidden_sel} older hidden" if hidden_sel else "") + "):",
    ]
    if not shown_sel:
        lines.append("  - none")
    for sid in shown_sel:
        lines.append(f'  - sid="{sid}": {_full_sent(state, sid)}')
    lines.append("visible:")
    if not visible:
        lines.append("  - none")
    for sid in visible[:8]:
        lines.append(f'  - sid="{sid}": {_full_sent(state, sid)}')
    lines.append("actions:")
    for aid, action in (state.get("action_map") or {}).items():
        if action.get("sid"):
            extra = f'sid="{action["sid"]}"'
        elif action.get("eid"):
            eid = action["eid"]
            surface = action.get("entity_surface") or (entities.get(eid) or {}).get("surface") or eid
            extra = f'eid="{eid}" surface="{surface}"'
        else:
            extra = ""
        hint = action.get("lexical_frontier_hint", action.get("snc_preview"))
        score = f" lexical_hint={hint}" if hint is not None else ""
        lines.append(f"  {aid} = {action.get('type')} {extra}{score}".rstrip())
    hist = state.get("tool_history") or []
    lines.append("tool_history: " + ", ".join(str(h.get("name")) for h in hist[-8:]))
    return "\n".join(lines)


def _lookup_mixquery(state: dict[str, Any]) -> str:
    sentences = state.get("sentences") or {}
    evidence = [
        str((sentences.get(sid) or {}).get("text") or "")
        for sid in (state.get("selected_sids") or [])
    ]
    meta = mixquery_text(str(state.get("query") or ""), evidence)
    state["last_mixquery"] = meta["query"]
    state["last_mixquery_meta"] = meta
    return str(meta["query"])


def _lookup_query(state: dict[str, Any], eid: str) -> str:
    rec = (state.get("entities") or {}).get(eid) or {}
    surface = str(rec.get("surface") or eid)
    mix = _lookup_mixquery(state)
    return f"{surface} {mix}".strip()


def _init_visible(state: dict[str, Any], *, searcher: Any | None, search_k: int) -> list[str]:
    query = str(state.get("query") or "")
    doc_order: list[str] = list((state.get("doc_store") or {}).keys())
    if searcher is not None and getattr(searcher, "name", "none") != "none":
        hits = searcher.search(query, int(search_k))
        added = _merge_search_hits(state, hits)
        state["real_search_calls"] = int(state.get("real_search_calls") or 0) + 1
        _note_effect(state, "init_retrieval_hits", len(hits))
        _note_effect(state, "init_new_docs", added)
        hit_ids = [str(h.docid) for h in hits]
        doc_order = hit_ids + [d for d in doc_order if d not in set(hit_ids)]
        state["initial_bm25_docids"] = hit_ids
    graph = _graph(state)
    hybrid = _mask_on(state, "hybrid_init_retrieve")
    if hybrid:
        _note_effect(state, "hybrid_init_used")
    t_init = time.perf_counter()
    begin_scan_bucket()
    ranked = graph.rank_init_sids(query, topk=VISIBLE_K, doc_order=doc_order, hybrid=hybrid)
    _stage_add(state, "init_retrieve_sec", time.perf_counter() - t_init)
    _merge_scan_counts(state)
    if hybrid:
        channels = getattr(graph, "last_hybrid_channels", {}) or {}
        _note_funnel(
            state,
            "hybrid_init_retrieve",
            enabled=True,
            invoked=1,
            candidate_delta=len(channels.get("global_outside_whitelist") or []),
            displayed_delta=len(ranked),
            backend=channels.get("backend"),
        )
        state["init_hybrid_channels"] = channels
    return ranked


def _lookup_sids(
    state: dict[str, Any],
    eid: str,
    *,
    new_doc_ids: list[str] | None = None,
) -> list[str]:
    graph = _graph(state)
    mix = _lookup_mixquery(state)
    offset = int(state.get("lookup_window_offset") or 0)
    candidates: list[dict[str, Any]] = []
    begin_scan_bucket()
    rows = graph.lookup_entity(
        eid,
        mix,
        topk=VISIBLE_K,
        use_synonyms=_mask_on(state, "entity_synonyms"),
        use_neighbors=_mask_on(state, "sentence_neighbors"),
        extra_sids=None,
        observed_sids=state.get("observed_sids") or [],
        new_doc_ids=new_doc_ids,
        window_offset=offset,
        candidates_out=candidates,
    )
    lookup_stats = take_lookup_stats()
    _stage_add(state, "lookup_candidates_sec", float(lookup_stats.get("lookup_candidates_sec") or 0.0))
    _stage_add(state, "lookup_rank_sec", float(lookup_stats.get("lookup_rank_sec") or 0.0))
    _merge_scan_counts(state)
    counts = state.setdefault("_stage_counts", {})
    lookup_counts = counts.setdefault("lookup", {})
    lookup_counts["n_candidates"] = int(lookup_counts.get("n_candidates") or 0) + int(lookup_stats.get("n_candidates") or 0)
    lookup_counts["n_selected"] = int(lookup_counts.get("n_selected") or 0) + int(lookup_stats.get("n_selected") or 0)
    state["lookup_candidates"] = candidates
    state["last_lookup_n_candidates"] = len(candidates)
    if _mask_on(state, "entity_synonyms"):
        syn_n = sum(1 for row in rows if row.get("entity_source") == "synonym")
        syn_cand = sum(1 for row in candidates if row.get("entity_source") == "synonym")
        _note_effect(state, "entity_synonyms_expanded", max(1, syn_n) if syn_n else 0)
        _note_effect(state, "entity_synonyms_candidate_delta", syn_cand)
        _note_effect(state, "entity_synonyms_visible_delta", syn_n)
        _note_funnel(
            state,
            "entity_synonyms",
            enabled=True,
            invoked=1,
            candidate_delta=syn_cand,
            displayed_delta=syn_n,
        )
    if _mask_on(state, "sentence_neighbors"):
        nb = sum(1 for row in rows if row.get("entity_source") == "sentence_neighbor")
        nb_cand = sum(1 for row in candidates if row.get("entity_source") == "sentence_neighbor")
        _note_effect(state, "sentence_neighbors_added", nb)
        _note_effect(state, "sentence_neighbors_visible_delta", nb)
        _note_funnel(
            state,
            "sentence_neighbors",
            enabled=True,
            invoked=1,
            candidate_delta=nb_cand,
            displayed_delta=nb,
        )
    state["last_lookup_rows"] = [{"sid": r["sid"], "score": r.get("score"), "source": r.get("entity_source"), "rank": r.get("rank")} for r in rows]
    return [str(r["sid"]) for r in rows]


def _tool_history_append(
    state: dict[str, Any],
    *,
    name: str,
    args: dict[str, Any],
    parse_ok: bool,
    schema_ok: bool,
    target_ok: bool,
    execution_ok: bool,
    error_code: str | None = None,
) -> None:
    state["tool_history"].append(
        {
            "name": name,
            "legal": parse_ok and schema_ok,
            "parse_ok": parse_ok,
            "schema_ok": schema_ok,
            "target_ok": target_ok,
            "menu_ok": target_ok,
            "execution_ok": execution_ok,
            "args": dict(args or {}),
            "error_code": error_code,
        }
    )


def _evidence_sig(state: Mapping[str, Any]) -> str:
    payload = {
        "visible": list(state.get("visible_sids") or []),
        "selected": list(state.get("selected_sids") or []),
        "frontier": list(state.get("frontier_eids") or []),
        "visited": list(state.get("visited_eids") or []),
        "initialized": bool(state.get("initialized")),
        "ended": bool(state.get("ended")),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _menu_hash(state: Mapping[str, Any]) -> str:
    blob = json.dumps(state.get("action_map") or {}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _append_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    events = list(state.get("turn_events") or [])
    events.append(event)
    state["turn_events"] = events


def _enter_failed_terminal(state: dict[str, Any], *, kind: str, code: str) -> None:
    state["ended"] = True
    state["end_reason"] = str(kind)
    state["termination_kind"] = str(kind)
    state["action_map"] = {}
    if kind == "protocol_failure":
        state["protocol_failure"] = True
    if kind == "infrastructure_failure":
        state["infrastructure_failure"] = True
        state["protocol_failure"] = True


def _fail(
    state: dict[str, Any],
    name: str,
    args: dict[str, Any],
    *,
    code: str,
    msg: str,
    parse_ok: bool = True,
    schema_ok: bool = True,
    target_ok: bool = False,
    evidence_sig: str | None = None,
    error_class: str = "protocol",
    traceback_ref: str | None = None,
) -> tuple[dict[str, Any], str, bool]:
    key = json.dumps(
        {
            "sig": evidence_sig or _evidence_sig(state),
            "name": str(name or "").lower(),
            "args": {k: args.get(k) for k in sorted(args or {}) if k in {"sid", "eid", "id", "sids", "direction"}},
            "code": code,
            "class": error_class,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    counts = dict(state.get("fail_counts") or {})
    counts[key] = int(counts.get(key) or 0) + 1
    state["fail_counts"] = counts
    terminal = False
    if error_class == "infrastructure":
        code = "infrastructure_failure"
        state["infrastructure_failure"] = True
        msg = f"internal error during `{name}`: {msg}"
        _enter_failed_terminal(state, kind="infrastructure_failure", code=code)
        terminal = True
    elif counts[key] >= MAX_IDENTICAL_FAILURES:
        code = "protocol_failure"
        msg = (
            f"same invalid `{name}` failed {counts[key]} times with {args}. "
            f"{_menu_hint(state, want=None)}"
        )
        _enter_failed_terminal(state, kind="protocol_failure", code=code)
        terminal = True
    hint = _menu_hint(state)
    if hint not in msg:
        msg = f"{msg} {hint}"
    state["invalid_tools"] = int(state.get("invalid_tools") or 0) + 1
    _tool_history_append(
        state,
        name=name,
        args=args,
        parse_ok=parse_ok,
        schema_ok=schema_ok,
        target_ok=target_ok,
        execution_ok=False,
        error_code=code,
    )
    include_answer = bool(state.get("initialized"))
    if not state.get("ended"):
        if state.get("initialized"):
            state["action_map"] = build_action_map(
                state,
                include_answer=include_answer,
                lookup_cap=SELECT_LOOKUP_K if include_answer else NAV_LOOKUP_K,
            )
        else:
            state["action_map"] = {}
    _sync_curated(state)
    obs = f"ERROR [{code}]: {msg}"
    _append_event(
        state,
        {
            "turn": int(state.get("step") or 0),
            "phase": "execute",
            "name": name,
            "args": dict(args or {}),
            "parse_ok": parse_ok,
            "schema_ok": schema_ok,
            "menu_ok": target_ok,
            "execution_ok": False,
            "error_code": code,
            "error_class": error_class,
            "traceback_ref": traceback_ref,
            "menu_hash": _menu_hash(state),
            "mixquery": state.get("last_mixquery"),
            "termination_kind": state.get("termination_kind") if terminal else None,
        },
    )
    return state, obs, False


def _note_termination_diagnostics(state: dict[str, Any], *, reason: str) -> None:
    candidates = list(state.get("lookup_candidates") or [])
    visible = set(state.get("visible_sids") or [])
    unshown = [row.get("sid") for row in candidates if isinstance(row, dict) and row.get("sid") not in visible]
    state["termination_diagnostics"] = {
        "reason": reason,
        "termination_kind": state.get("termination_kind") or reason,
        "empty_selection": not bool(state.get("selected_sids")),
        "n_selected": len(state.get("selected_sids") or []),
        "n_visible": len(visible),
        "n_candidates": len(candidates),
        "n_unshown_candidates": len([sid for sid in unshown if sid]),
        "n_observed_sids": len(state.get("observed_sids") or []),
        "n_candidate_docs": len(state.get("candidate_docids") or []),
    }


def summarize_component_funnels(state: Mapping[str, Any]) -> dict[str, Any]:
    from trim.adapters.harness_g_components import COMPONENT_TAXONOMY

    mask = state.get("harness_mask") or {}
    funnel = dict(state.get("component_funnel") or {})
    out: dict[str, Any] = {}
    for cid in COMPONENT_TAXONOMY:
        rec = dict(funnel.get(cid) or {})
        rec.setdefault("enabled", bool(mask.get(cid)))
        rec.setdefault("eligible", 0)
        rec.setdefault("invoked", 0)
        rec.setdefault("candidate_delta", 0)
        rec.setdefault("menu_delta", 0)
        rec.setdefault("displayed_delta", 0)
        rec.setdefault("state_delta", 0)
        if not rec.get("enabled"):
            rec.setdefault("skip_reason", "disabled")
        elif not rec.get("invoked") and not rec.get("eligible"):
            rec.setdefault("skip_reason", rec.get("skip_reason") or "not_triggered")
        out[cid] = rec
    return out


def record_nonexecution_failure(
    state: dict[str, Any],
    name: str,
    args: dict[str, Any] | None,
    *,
    code: str,
    msg: str,
    parse_ok: bool = False,
    schema_ok: bool = False,
    error_class: str = "protocol",
    traceback_ref: str | None = None,
) -> tuple[dict[str, Any], str, bool]:
    st = dict(state)
    st["fail_counts"] = dict(state.get("fail_counts") or {})
    st["tool_history"] = list(state.get("tool_history") or [])
    st["turn_events"] = list(state.get("turn_events") or [])
    st["harness_mask"] = dict(state.get("harness_mask") or zero_mask_for("Harness-G"))
    st["action_map"] = dict(state.get("action_map") or {})
    st["invalid_tools"] = int(state.get("invalid_tools") or 0)
    return _fail(
        st,
        name,
        dict(args or {}),
        code=code,
        msg=msg,
        parse_ok=parse_ok,
        schema_ok=schema_ok,
        target_ok=False,
        error_class=error_class,
        traceback_ref=traceback_ref,
    )


def execute_tool(
    state: dict[str, Any],
    name: str | None,
    args: dict[str, Any] | None,
    *,
    searcher: Any | None = None,
    search_k: int = 10,
) -> tuple[dict[str, Any], str, bool]:
    st = dict(state)
    graph = _graph(state)
    st["graph"] = graph
    st["sentences"] = graph.sentences
    st["entities"] = graph.entities
    st["selected_sids"] = list(state.get("selected_sids") or [])
    st["visible_sids"] = list(state.get("visible_sids") or [])
    st["visited_eids"] = list(state.get("visited_eids") or [])
    st["frontier_eids"] = list(state.get("frontier_eids") or [])
    st["tool_history"] = list(state.get("tool_history") or [])
    st["turn_events"] = list(state.get("turn_events") or [])
    st["fail_counts"] = dict(state.get("fail_counts") or {})
    st["harness_mask"] = dict(state.get("harness_mask") or zero_mask_for("Harness-G"))
    st["runtime_effects"] = dict(state.get("runtime_effects") or {})
    st["observed_docids"] = list(state.get("observed_docids") or [])
    st["observed_sids"] = list(state.get("observed_sids") or [])
    st["displayed_sids"] = list(state.get("displayed_sids") or [])
    st["displayed_docids"] = list(state.get("displayed_docids") or [])
    st["candidate_docids"] = list(state.get("candidate_docids") or [])
    st["candidate_sids"] = list(state.get("candidate_sids") or [])
    st["graph_expanded_docids"] = list(state.get("graph_expanded_docids") or [])
    st["lookup_candidates"] = list(state.get("lookup_candidates") or [])
    st["nav_frontier_eids"] = list(state.get("nav_frontier_eids") or [])
    st["attempt_events"] = list(state.get("attempt_events") or [])
    st["component_funnel"] = dict(state.get("component_funnel") or {})
    st["doc_store"] = dict(state.get("doc_store") or {})
    evidence_sig = _evidence_sig(state)
    st["step"] = int(state.get("step") or 0) + 1
    st["n_tool_calls"] = int(state.get("n_tool_calls") or 0) + 1
    args = dict(args or {})
    visible_before = list(st.get("visible_sids") or [])
    selected_before = list(st.get("selected_sids") or [])
    try:
        return _execute_tool_dispatch(
            st,
            name,
            args,
            searcher=searcher,
            search_k=search_k,
            evidence_sig=evidence_sig,
            visible_before=visible_before,
            selected_before=selected_before,
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        return _fail(
            st,
            str(name or "unknown"),
            dict(args or {}),
            code="infrastructure_failure",
            msg=f"{type(exc).__name__}: {exc}",
            error_class="infrastructure",
            traceback_ref=tb[-4000:],
            evidence_sig=evidence_sig,
        )


def _execute_tool_dispatch(
    st: dict[str, Any],
    name: str | None,
    args: dict[str, Any],
    *,
    searcher: Any | None,
    search_k: int,
    evidence_sig: str,
    visible_before: list[str],
    selected_before: list[str],
) -> tuple[dict[str, Any], str, bool]:
    state_action_map = st.get("action_map") or {}

    if name and name.upper().startswith("A") and name[1:].isdigit():
        mapped = state_action_map.get(name) or state_action_map.get(str(name).upper())
        if mapped:
            name = str(mapped.get("name") or mapped.get("type") or name).lower()
            if mapped.get("sid") and "sid" not in args:
                args["sid"] = mapped["sid"]
            if mapped.get("eid") and "eid" not in args:
                args["eid"] = mapped["eid"]
            if mapped.get("sids") and "sids" not in args:
                args["sids"] = list(mapped["sids"])
            if mapped.get("direction") and "direction" not in args:
                args["direction"] = mapped["direction"]

    name = str(name or "").lower()
    if name == "answer_with" and not _mask_on(st, "answer_with"):
        return _fail(st, name, args, code="tool_disabled", msg="invalid tool `answer_with` (component off).", evidence_sig=evidence_sig)

    legal = set(RUNTIME_TOOLS) | ({"answer_with"} if _mask_on(st, "answer_with") else set())
    if name not in legal:
        return _fail(
            st,
            name,
            args,
            code="invalid_tool",
            msg=f"invalid tool `{name}`.",
            parse_ok=False,
            schema_ok=False,
            evidence_sig=evidence_sig,
        )

    if name == "init":
        if st.get("initialized"):
            return _fail(st, name, args, code="already_initialized", msg="environment already initialized.", evidence_sig=evidence_sig)
        st["initialized"] = True
        st["visible_sids"] = _init_visible(st, searcher=searcher, search_k=search_k)
        hit_ids = [str(x) for x in (st.get("initial_bm25_docids") or []) if str(x)]
        if not hit_ids:
            hit_ids = [str(x) for x in (st.get("doc_store") or {}) if str(x)]
        vis_docs = [did for did in (_parent_docid(st, sid) for sid in st["visible_sids"]) if did]
        st["candidate_docids"] = list(dict.fromkeys(list(st.get("candidate_docids") or []) + hit_ids))
        st["observed_docids"] = list(dict.fromkeys(list(st.get("observed_docids") or []) + vis_docs))
        st["n_search_calls"] = int(st.get("n_search_calls") or 0) + 1
        st["search_count"] = int(st.get("search_count") or 0) + 1
        _set_nav_menu(st, include_answer=False)
        _tool_history_append(
            st,
            name="init",
            args={},
            parse_ok=True,
            schema_ok=True,
            target_ok=True,
            execution_ok=True,
        )
        _sync_curated(st)
        _append_event(
            st,
            {
                "turn": st["step"],
                "phase": "execute",
                "name": "init",
                "execution_ok": True,
                "visible_delta": [s for s in st["visible_sids"] if s not in visible_before],
                "menu_hash": _menu_hash(st),
            },
        )
        return st, "INIT retrieved visible sentences.", True

    if not st.get("initialized"):
        return _fail(
            st,
            name,
            args,
            code="not_initialized",
            msg="call init before other tools.",
            target_ok=False,
            evidence_sig=evidence_sig,
        )

    pairs = allowed_menu_pairs(st)

    if name == "select":
        sid = str(args.get("sid") or args.get("id") or "")
        visible = set(st.get("visible_sids") or [])
        if not sid or sid not in st["sentences"]:
            return _fail(st, name, args, code="sid_not_found", msg=f"sid `{sid}` is not a known sentence.", evidence_sig=evidence_sig)
        if ("SELECT", sid) not in pairs:
            return _fail(
                st,
                name,
                args,
                code="target_not_in_menu",
                msg=f"sid `{sid}` is not a SELECT target in the current menu.",
                target_ok=False,
                evidence_sig=evidence_sig,
            )
        if sid not in visible:
            return _fail(st, name, args, code="sid_not_visible", msg=f"sid `{sid}` is not currently visible.", evidence_sig=evidence_sig)
        if sid not in st["selected_sids"]:
            st["selected_sids"].append(sid)
        sent = st["sentences"].get(sid) or {}
        _set_select_menu(st, sid)
        obs = f'SELECT sid="{sid}": {str(sent.get("text") or "")}'
        exec_ok = True
    elif name == "lookup":
        eid = str(args.get("eid") or args.get("id") or "")
        entities = st.get("entities") or {}
        if not eid:
            return _fail(st, name, args, code="missing_eid", msg="lookup requires eid.", evidence_sig=evidence_sig)
        if ("LOOKUP", eid) not in pairs:
            return _fail(
                st,
                name,
                args,
                code="target_not_in_menu",
                msg=f"eid `{eid}` is not a LOOKUP target in the current menu.",
                target_ok=False,
                evidence_sig=evidence_sig,
            )
        if eid not in entities:
            return _fail(st, name, args, code="eid_not_found", msg=f"eid `{eid}` is not a known entity.", evidence_sig=evidence_sig)
        if _mask_on(st, "lookup_dedup") and eid in set(st.get("visited_eids") or []):
            _note_effect(st, "lookup_dedup_blocked")
            return _fail(
                st,
                name,
                args,
                code="eid_already_visited",
                msg=f"eid `{eid}` was already looked up (lookup_dedup).",
                evidence_sig=evidence_sig,
            )
        if eid not in st["visited_eids"]:
            st["visited_eids"].append(eid)
        new_doc_ids: list[str] = []
        added = 0
        observed_before = set(st.get("observed_docids") or [])
        observed_sids_before = set(st.get("observed_sids") or [])
        candidate_before = set(st.get("candidate_docids") or [])
        corpus_mode = is_corpus_scope(st.get("graph_scope"))
        lexical_hits: list[Any] = []
        if (
            not corpus_mode
            and searcher is not None
            and getattr(searcher, "name", "none") != "none"
        ):
            lookup_q = _lookup_query(st, eid)
            lexical_hits = searcher.search(lookup_q, int(search_k)) or []
            for hit in lexical_hits:
                did = str(getattr(hit, "docid", "") or "")
                if did and did not in observed_before and did not in new_doc_ids:
                    new_doc_ids.append(did)
            added = _merge_search_hits(st, lexical_hits)
            st["real_search_calls"] = int(st.get("real_search_calls") or 0) + 1
            _note_effect(st, "lexical_lookup_calls")
            _note_effect(st, "lexical_candidate_docs", len(lexical_hits))
            _note_effect(st, "lexical_new_docs", added)
            _note_effect(st, "lookup_retrieval_hits", len(lexical_hits))
            _note_effect(st, "lookup_new_docs", added)
            _note_effect(st, "lookup_new_docids", len(new_doc_ids))
        st["n_search_calls"] = int(st.get("n_search_calls") or 0) + 1
        st["search_count"] = int(st.get("search_count") or 0) + 1
        st["lookup_window_offset"] = 0
        st["visible_sids"] = _lookup_sids(st, eid, new_doc_ids=new_doc_ids or None)
        if corpus_mode:
            st["graph_lookups"] = int(st.get("graph_lookups") or 0) + 1
            _note_effect(st, "graph_lookup_calls")
            _note_effect(st, "graph_expanded_entities")
            graph = _graph(st)
            rec = (st.get("entities") or {}).get(eid) or {}
            n_sids = len(rec.get("sids") or [])
            expanded: list[str] = []
            if n_sids <= GRAPH_MAX_ENTITY_SIDS:
                docs = graph.docs_for_entity(eid)
                if len(docs) <= GRAPH_MAX_ENTITY_DOCS:
                    expanded = [str(d) for d in docs if str(d)]
            prev_expanded = list(st.get("graph_expanded_docids") or [])
            seen_exp = set(prev_expanded)
            for did in expanded:
                if did not in seen_exp:
                    prev_expanded.append(did)
                    seen_exp.add(did)
            st["graph_expanded_docids"] = prev_expanded
            cand = list(st.get("candidate_docids") or [])
            cand_set = set(cand)
            for did in expanded:
                if did not in cand_set:
                    cand.append(did)
                    cand_set.add(did)
            st["candidate_docids"] = cand
            graph_docs = set(expanded) | {
                _parent_docid(st, sid)
                for sid in st["visible_sids"]
                if _parent_docid(st, sid)
            }
            graph_new_docs = [did for did in graph_docs if did not in candidate_before]
            graph_new_sids = [sid for sid in st["visible_sids"] if sid not in observed_sids_before]
            _note_effect(st, "graph_candidate_docs", len(graph_docs))
            _note_effect(st, "graph_new_docs", len(graph_new_docs))
            _note_effect(st, "graph_new_sids", len(graph_new_sids))
        _set_nav_menu(st, include_answer=True)
        rec = st["entities"].get(eid) or {}
        if not st["visible_sids"]:
            obs = (
                f'LOOKUP eid="{eid}" surface="{rec.get("surface")}": '
                f"no new sentences (added_docs={added})."
            )
        else:
            obs = (
                f'LOOKUP eid="{eid}" surface="{rec.get("surface")}": '
                f"{len(st['visible_sids'])} sentences (added_docs={added})."
            )
        exec_ok = True
    elif name == "page":
        if ("PAGE", "next") not in pairs and ("PAGE", "") not in pairs:
            direction = str(args.get("direction") or "next")
            if ("PAGE", direction) not in pairs:
                return _fail(
                    st,
                    name,
                    args,
                    code="target_not_in_menu",
                    msg="PAGE is not in the current menu.",
                    target_ok=False,
                    evidence_sig=evidence_sig,
                )
        candidates = list(st.get("lookup_candidates") or [])
        if not candidates:
            return _fail(st, name, args, code="no_lookup_candidates", msg="no LOOKUP candidates to page.", evidence_sig=evidence_sig)
        direction = str(args.get("direction") or "next")
        offset = int(st.get("lookup_window_offset") or 0)
        if direction == "prev":
            offset = max(0, offset - VISIBLE_K)
        else:
            offset = min(max(0, len(candidates) - VISIBLE_K), offset + VISIBLE_K)
        window = candidates[offset : offset + VISIBLE_K]
        if not window:
            return _fail(st, name, args, code="no_more_candidates", msg="no remaining LOOKUP candidates.", evidence_sig=evidence_sig)
        st["lookup_window_offset"] = offset
        st["visible_sids"] = [str(row.get("sid") or "") for row in window if row.get("sid")]
        _set_nav_menu(st, include_answer=True)
        obs = f"PAGE offset={offset} showing={len(st['visible_sids'])} of {len(candidates)} candidates."
        exec_ok = True
    elif name == "answer_with":
        sids_arg = list(args.get("sids") or [])
        sid = str(args.get("sid") or (sids_arg[0] if sids_arg else ""))
        if not sid or sid not in st["sentences"]:
            return _fail(st, name, args, code="sid_not_found", msg=f"answer_with sid `{sid}` is not valid.", evidence_sig=evidence_sig)
        if ("ANSWER_WITH", sid) not in pairs:
            already = sid in set(st.get("selected_sids") or [])
            extra = " If evidence is already selected, call ANSWER." if already else ""
            return _fail(
                st,
                name,
                args,
                code="target_not_in_menu",
                msg=(
                    f"ANSWER_WITH sid `{sid}` is not an unselected visible sentence in the current menu."
                    f"{extra}"
                ),
                target_ok=False,
                evidence_sig=evidence_sig,
            )
        if sid not in st["selected_sids"]:
            st["selected_sids"].append(sid)
        st["ended"] = True
        st["end_reason"] = "answer_with"
        st["termination_kind"] = "model_answer"
        st["action_map"] = {}
        _note_termination_diagnostics(st, reason="answer_with")
        _tool_history_append(
            st,
            name=name,
            args=args,
            parse_ok=True,
            schema_ok=True,
            target_ok=True,
            execution_ok=True,
        )
        _note_effect(st, "answer_with_used")
        _sync_curated(st)
        _append_event(
            st,
            {
                "turn": st["step"],
                "phase": "execute",
                "name": name,
                "execution_ok": True,
                "selected_delta": [s for s in st["selected_sids"] if s not in selected_before],
                "end_reason": "answer_with",
            },
        )
        return st, f"ANSWER_WITH selected={st['selected_sids'][:8]}", True
    elif name == "answer":
        if ("ANSWER", None) not in pairs:
            return _fail(
                st,
                name,
                args,
                code="target_not_in_menu",
                msg="ANSWER is not in the current menu. Select evidence first, or use ANSWER_WITH on a visible sentence.",
                target_ok=False,
                evidence_sig=evidence_sig,
            )
        st["ended"] = True
        st["end_reason"] = str(args.get("reason") or args.get("reasoning") or "answer")
        st["termination_kind"] = "model_answer"
        st["action_map"] = {}
        _note_termination_diagnostics(st, reason=str(st["end_reason"]))
        _tool_history_append(
            st,
            name=name,
            args=args,
            parse_ok=True,
            schema_ok=True,
            target_ok=True,
            execution_ok=True,
        )
        _sync_curated(st)
        _append_event(
            st,
            {
                "turn": st["step"],
                "phase": "execute",
                "name": name,
                "execution_ok": True,
                "end_reason": st["end_reason"],
            },
        )
        return st, f"ANSWER selected={st['selected_sids'][:8]}", True
    else:
        return _fail(st, name, args, code="unhandled_tool", msg=f"unhandled tool `{name}`.", evidence_sig=evidence_sig)

    if not st.get("ended") and name not in {"select", "lookup", "page"}:
        st["action_map"] = build_action_map(st, include_answer=True)
    _tool_history_append(
        st,
        name=name,
        args=args,
        parse_ok=True,
        schema_ok=True,
        target_ok=True,
        execution_ok=exec_ok,
    )
    _sync_curated(st)
    _append_event(
        st,
        {
            "turn": st["step"],
            "phase": "execute",
            "name": name,
            "args": dict(args or {}),
            "execution_ok": exec_ok,
            "menu_hash": _menu_hash(st),
            "mixquery": st.get("last_mixquery"),
            "visible_delta": [s for s in (st.get("visible_sids") or []) if s not in visible_before],
            "selected_delta": [s for s in (st.get("selected_sids") or []) if s not in selected_before],
            "last_lookup_rows": st.get("last_lookup_rows"),
        },
    )
    return st, obs, exec_ok


def curated_recall(state: dict[str, Any], gold_ids: list[str]) -> float | None:
    from trim.eval.local_search_env import curated_recall as _h1_recall

    return _h1_recall(state, gold_ids)


def is_g_state(state: Mapping[str, Any] | None) -> bool:
    if not state:
        return False
    if "sentences" in state or "action_map" in state or "graph" in state:
        return True
    return is_harness_g(mask=state.get("harness_mask"))
