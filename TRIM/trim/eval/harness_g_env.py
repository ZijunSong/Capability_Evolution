"""TRIM-native Harness-G episode over a per-query doc_store.

Mirrors ``local_search_env`` for Harness-1. The always-on runtime is
INIT / SELECT / LOOKUP / ANSWER. Advanced components (answer_with, bridges,
synonyms, neighbors, hybrid INIT, SNC previews, …) are mask-gated.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from trim.adapters.harness_g_components import RUNTIME_TOOLS
from trim.adapters.harness_profiles import is_harness_g, zero_mask_for
from trim.eval.local_search_env import _doc_text, _tokenize

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_BAD_LOOKUP = re.compile(
    r"^(\d+|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|american|british|french|german|chinese|"
    r"japanese|russian|indian|canadian)$",
    re.I,
)
_FRONT_MATTER_RE = re.compile(
    r"^(title|author|date|published|copyright|table of contents|references)\b",
    re.I,
)
_MAX_SENTS_PER_DOC = 48


def _sentences_from_store(doc_store: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sentences: dict[str, dict[str, Any]] = {}
    for did, rec in (doc_store or {}).items():
        text = _doc_text(rec)
        parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
        if not parts:
            parts = [text.strip()] if text.strip() else []
        kept = [p for p in parts if not _FRONT_MATTER_RE.match(p.strip())] or parts
        for i, sent in enumerate(kept[:_MAX_SENTS_PER_DOC]):
            sid = f"{did}:s{i}"
            sentences[sid] = {
                "sid": sid,
                "doc_id": str(did),
                "text": sent,
                "idx": i,
                "neighbors": [
                    f"{did}:s{j}"
                    for j in (i - 1, i + 1)
                    if 0 <= j < min(_MAX_SENTS_PER_DOC, len(kept))
                ],
            }
    return sentences


def _entities_from_sentences(sentences: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for sid, sent in sentences.items():
        for match in _ENTITY_RE.findall(str(sent.get("text") or "")):
            surface = match.strip()
            if len(surface) < 3:
                continue
            eid = "e:" + re.sub(r"\s+", "_", surface.lower())
            rec = entities.setdefault(
                eid,
                {"eid": eid, "surface": surface, "sids": [], "synonyms": []},
            )
            if sid not in rec["sids"]:
                rec["sids"].append(sid)
    by_token: dict[str, list[str]] = {}
    for eid, rec in entities.items():
        tok = rec["surface"].split()[0].lower()
        by_token.setdefault(tok, []).append(eid)
    for group in by_token.values():
        if len(group) < 2:
            continue
        for eid in group:
            entities[eid]["synonyms"] = [x for x in group if x != eid]
    return entities


def _rebuild_index(state: dict[str, Any]) -> None:
    store = dict(state.get("doc_store") or {})
    state["sentences"] = _sentences_from_store(store)
    state["entities"] = _entities_from_sentences(state["sentences"])


def _merge_search_hits(state: dict[str, Any], hits: list[Any]) -> int:
    store = dict(state.get("doc_store") or {})
    added = 0
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
            store[did] = {"id": did, "text": text, "score": score}
    state["doc_store"] = store
    _rebuild_index(state)
    return added


def _rank_sids(query: str, sentences: Mapping[str, Mapping[str, Any]], k: int) -> list[str]:
    q = _tokenize(query)
    scored: list[tuple[float, str]] = []
    for sid, sent in sentences.items():
        toks = _tokenize(str(sent.get("text") or ""))
        inter = len(q & toks)
        scored.append((inter / max(1, len(q)), sid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [sid for _, sid in scored[:k]]


def _rrf_fuse(lists: list[list[str]], *, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for lst in lists:
        for rank, sid in enumerate(lst):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank + 1)
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
        if rec["surface"].lower() in q_low:
            entity_hits.extend(rec.get("sids") or [])
    bm25_sids: list[str] = []
    if doc_order:
        by_doc: dict[str, list[str]] = {}
        for sid, sent in sentences.items():
            by_doc.setdefault(str(sent.get("doc_id")), []).append(sid)
        for did in doc_order:
            bm25_sids.extend(sorted(by_doc.get(did) or []))
    fused = _rrf_fuse([lexical, entity_hits, bm25_sids])
    return fused[:k]


def new_state(
    query: str,
    doc_store: dict[str, Any],
    *,
    harness_mask: Mapping[str, bool] | None = None,
    visible_k: int = 6,
) -> dict[str, Any]:
    del visible_k
    mask = dict(harness_mask) if harness_mask is not None else zero_mask_for("Harness-G")
    sentences = _sentences_from_store(doc_store)
    entities = _entities_from_sentences(sentences)
    return {
        "query": query,
        "step": 0,
        "ended": False,
        "end_reason": None,
        "initialized": False,
        "doc_store": doc_store or {},
        "sentences": sentences,
        "entities": entities,
        "visible_sids": [],
        "selected_sids": [],
        "visited_eids": [],
        "frontier_eids": [],
        "action_map": {},
        "harness_mask": mask,
        "tool_history": [],
        "n_tool_calls": 0,
        "invalid_tools": 0,
        "pool": {},
        "curated": {},
        "search_count": 0,
        "n_search_calls": 0,
        "real_search_calls": 0,
        "observed_docids": [],
        "observed_sids": [],
        "selected_docids": [],
        "runtime_effects": {},
    }


def _mask_on(state: Mapping[str, Any], component_id: str) -> bool:
    return bool((state.get("harness_mask") or {}).get(component_id, False))


def _is_bad_lookup(entity: Mapping[str, Any]) -> bool:
    surface = str(entity.get("surface") or "")
    return bool(_BAD_LOOKUP.match(surface.strip()))


def _select_score(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    return len(q & t) / max(1, len(q))


def _note_effect(state: dict[str, Any], key: str, n: int = 1) -> None:
    effects = dict(state.get("runtime_effects") or {})
    effects[key] = int(effects.get(key) or 0) + int(n)
    state["runtime_effects"] = effects


def build_action_map(state: dict[str, Any], *, include_answer: bool) -> dict[str, dict[str, Any]]:
    menu: dict[str, dict[str, Any]] = {}
    n = 0
    selected = set(state.get("selected_sids") or [])
    visible = set(state.get("visible_sids") or [])
    sentences = state.get("sentences") or {}
    entities = state.get("entities") or {}
    query = str(state.get("query") or "")

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

    lookup_eids: list[str] = []
    for eid in state.get("frontier_eids") or []:
        if eid not in lookup_eids:
            lookup_eids.append(eid)
    if _mask_on(state, "bridge_entities"):
        for rec in entities.values():
            if rec["eid"] in lookup_eids:
                continue
            if any(sid in visible for sid in rec.get("sids") or []):
                continue
            if any(sid in selected for sid in rec.get("sids") or []):
                lookup_eids.append(rec["eid"])

    visited = set(state.get("visited_eids") or [])
    added = 0
    for eid in lookup_eids:
        if added >= 8:
            break
        rec = entities.get(eid)
        if not rec:
            continue
        if _mask_on(state, "lookup_dedup") and eid in visited:
            continue
        if _mask_on(state, "invalid_target_filter") and _is_bad_lookup(rec):
            _note_effect(state, "invalid_target_filtered")
            continue
        menu[f"A{n}"] = {
            "type": "LOOKUP",
            "eid": eid,
            "name": "lookup",
            "entity_surface": rec.get("surface"),
        }
        n += 1
        added += 1

    if include_answer:
        menu[f"A{n}"] = {"type": "ANSWER", "name": "answer"}
        n += 1

    if _mask_on(state, "snc_frontier"):
        for action in menu.values():
            if action["type"] == "SELECT":
                text = str((sentences.get(action["sid"]) or {}).get("text") or "")
                action["snc_preview"] = round(_select_score(query, text), 4)
            elif action["type"] == "LOOKUP":
                rec = entities.get(action["eid"]) or {}
                texts = " ".join(
                    str((sentences.get(sid) or {}).get("text") or "") for sid in rec.get("sids") or []
                )
                action["snc_preview"] = round(_select_score(query, texts), 4)
            else:
                action["snc_preview"] = 0.0
    return menu


def _sync_curated(state: dict[str, Any]) -> None:
    curated: dict[str, Any] = {}
    store = state.get("doc_store") or {}
    sentences = state.get("sentences") or {}
    pool: dict[str, Any] = {}
    observed_docids: set[str] = set()
    observed_sids: set[str] = set()
    selected_docids: set[str] = set()
    selected_sids: set[str] = set()

    for sid in state.get("selected_sids") or []:
        sent = sentences.get(sid) or {}
        did = str(sent.get("doc_id") or sid.split(":")[0] if ":" in sid else sid)
        rec = store.get(did) or {"id": did, "text": sent.get("text") or ""}
        curated[did] = rec if isinstance(rec, dict) else {"id": did, "text": str(rec)}
        pool[did] = curated[did]
        selected_sids.add(sid)
        selected_docids.add(did)
        observed_sids.add(sid)
        observed_docids.add(did)

    for sid in state.get("visible_sids") or []:
        sent = sentences.get(sid) or {}
        did = str(sent.get("doc_id") or sid.split(":")[0] if ":" in sid else sid)
        observed_sids.add(sid)
        observed_docids.add(did)
        if did not in pool:
            rec = store.get(did) or {"id": did, "text": sent.get("text") or ""}
            pool[did] = rec if isinstance(rec, dict) else {"id": did, "text": str(rec)}

    state["curated"] = curated
    state["pool"] = pool
    state["curated_ids"] = list(curated)
    state["observed_docids"] = sorted(observed_docids)
    state["observed_sids"] = sorted(observed_sids)
    state["selected_docids"] = sorted(selected_docids)
    state["selected_sids"] = sorted(selected_sids)


def wm_text(state: dict[str, Any], *, auto_on: bool = False) -> str:
    del auto_on
    sentences = state.get("sentences") or {}
    entities = state.get("entities") or {}
    lines = [
        "[Harness-G Working Memory]",
        "Use only the target ids shown below (sid=..., eid=...). Do not invent search queries.",
        f"step={state.get('step', 0)} initialized={bool(state.get('initialized'))}",
        f"n_selected={len(state.get('selected_sids') or [])} n_visible={len(state.get('visible_sids') or [])}",
        "selected:",
    ]
    for sid in (state.get("selected_sids") or [])[:8]:
        lines.append(f"  - sid=\"{sid}\": {str((sentences.get(sid) or {}).get('text') or '')[:160]}")
    lines.append("visible:")
    for sid in (state.get("visible_sids") or [])[:8]:
        lines.append(f"  - sid=\"{sid}\": {str((sentences.get(sid) or {}).get('text') or '')[:120]}")
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
        preview = action.get("snc_preview")
        score = f" snc={preview}" if preview is not None else ""
        lines.append(f"  {aid} = {action.get('type')} {extra}{score}".rstrip())
    hist = state.get("tool_history") or []
    lines.append("tool_history: " + ", ".join(str(h.get("name")) for h in hist[-8:]))
    return "\n".join(lines)


def _lookup_query(state: dict[str, Any], eid: str) -> str:
    entities = state.get("entities") or {}
    rec = entities.get(eid) or {}
    surface = str(rec.get("surface") or eid)
    query = str(state.get("query") or "")
    selected = " ".join(
        str((state.get("sentences") or {}).get(sid, {}).get("text") or "")[:120]
        for sid in (state.get("selected_sids") or [])[-2:]
    )
    parts = [surface, query]
    if selected.strip():
        parts.append(selected.strip())
    return " ".join(parts)


def _init_visible(state: dict[str, Any], *, searcher: Any | None, search_k: int) -> list[str]:
    query = str(state.get("query") or "")
    doc_order: list[str] = list((state.get("doc_store") or {}).keys())
    if searcher is not None and getattr(searcher, "name", "none") != "none":
        hits = searcher.search(query, int(search_k))
        added = _merge_search_hits(state, hits)
        state["real_search_calls"] = int(state.get("real_search_calls") or 0) + 1
        _note_effect(state, "init_retrieval_hits", len(hits))
        _note_effect(state, "init_new_docs", added)
        doc_order = [str(h.docid) for h in hits] + [d for d in doc_order if d not in {str(h.docid) for h in hits}]
    sentences = state.get("sentences") or {}
    entities = state.get("entities") or {}
    k = 6
    if _mask_on(state, "hybrid_init_retrieve"):
        ranked = _hybrid_rank_sids(query, sentences, entities, k, doc_order=doc_order)
        _note_effect(state, "hybrid_init_used")
        return ranked
    if doc_order:
        by_doc: dict[str, list[str]] = {}
        for sid, sent in sentences.items():
            by_doc.setdefault(str(sent.get("doc_id")), []).append(sid)
        ordered: list[str] = []
        for did in doc_order:
            ordered.extend(sorted(by_doc.get(did) or []))
        if ordered:
            return ordered[:k]
    return _rank_sids(query, sentences, k)


def _lookup_sids(state: dict[str, Any], eid: str, *, new_doc_order: list[str] | None = None) -> list[str]:
    entities = state.get("entities") or {}
    sentences = state.get("sentences") or {}
    rec = entities.get(eid) or {}
    sids = list(rec.get("sids") or [])
    if _mask_on(state, "entity_synonyms"):
        for syn in rec.get("synonyms") or []:
            sids.extend((entities.get(syn) or {}).get("sids") or [])
            _note_effect(state, "entity_synonyms_expanded")
    if _mask_on(state, "sentence_neighbors"):
        extra: list[str] = []
        for sid in list(sids):
            extra.extend((sentences.get(sid) or {}).get("neighbors") or [])
        sids.extend(extra)
        _note_effect(state, "sentence_neighbors_added", len(extra))
    if new_doc_order:
        by_doc: dict[str, list[str]] = {}
        for sid, sent in sentences.items():
            by_doc.setdefault(str(sent.get("doc_id")), []).append(sid)
        for did in new_doc_order:
            for sid in sorted(by_doc.get(did) or []):
                if sid not in sids:
                    sids.append(sid)
    out: list[str] = []
    for sid in sids:
        if sid in sentences and sid not in out:
            out.append(sid)
        if len(out) >= 6:
            break
    return out


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
            "execution_ok": execution_ok,
            "args": dict(args or {}),
            "error_code": error_code,
        }
    )


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
) -> tuple[dict[str, Any], str, bool]:
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
    state["action_map"] = build_action_map(state, include_answer=include_answer)
    _sync_curated(state)
    obs = f"ERROR [{code}]: {msg}\n" + wm_text(state)
    return state, obs, False


def execute_tool(
    state: dict[str, Any],
    name: str | None,
    args: dict[str, Any] | None,
    *,
    searcher: Any | None = None,
    search_k: int = 10,
) -> tuple[dict[str, Any], str, bool]:
    st = dict(state)
    st["sentences"] = dict(state.get("sentences") or {})
    st["entities"] = dict(state.get("entities") or {})
    st["selected_sids"] = list(state.get("selected_sids") or [])
    st["visible_sids"] = list(state.get("visible_sids") or [])
    st["visited_eids"] = list(state.get("visited_eids") or [])
    st["frontier_eids"] = list(state.get("frontier_eids") or [])
    st["tool_history"] = list(state.get("tool_history") or [])
    st["harness_mask"] = dict(state.get("harness_mask") or zero_mask_for("Harness-G"))
    st["runtime_effects"] = dict(state.get("runtime_effects") or {})
    st["step"] = int(state.get("step") or 0) + 1
    st["n_tool_calls"] = int(state.get("n_tool_calls") or 0) + 1
    args = dict(args or {})

    if name and name.upper().startswith("A") and name[1:].isdigit():
        mapped = (state.get("action_map") or {}).get(name) or (state.get("action_map") or {}).get(name.upper())
        if mapped:
            name = str(mapped.get("name") or mapped.get("type") or name).lower()
            if mapped.get("sid") and "sid" not in args:
                args["sid"] = mapped["sid"]
            if mapped.get("eid") and "eid" not in args:
                args["eid"] = mapped["eid"]
            if mapped.get("sids") and "sids" not in args:
                args["sids"] = list(mapped["sids"])

    name = str(name or "").lower()
    if name == "answer_with" and not _mask_on(st, "answer_with"):
        return _fail(st, name, args, code="tool_disabled", msg="invalid tool `answer_with` (component off).")

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
        )

    include_answer = True
    if name in {"init"} or not st.get("initialized"):
        if name in {"init", "select", "lookup", "answer", "answer_with"} and not st.get("initialized"):
            st["initialized"] = True
            st["visible_sids"] = _init_visible(st, searcher=searcher, search_k=search_k)
            st["n_search_calls"] = int(st.get("n_search_calls") or 0) + 1
            st["search_count"] = int(st.get("search_count") or 0) + 1
            include_answer = False
            if name == "init":
                st["action_map"] = build_action_map(st, include_answer=False)
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
                return st, "INIT retrieved visible sentences.\n" + wm_text(st), True

    if name == "select":
        sid = str(args.get("sid") or args.get("id") or "")
        visible = set(st.get("visible_sids") or [])
        if not sid or sid not in st["sentences"]:
            return _fail(st, name, args, code="sid_not_found", msg=f"sid `{sid}` is not a known sentence.")
        if sid not in visible:
            return _fail(st, name, args, code="sid_not_visible", msg=f"sid `{sid}` is not currently visible.")
        if sid not in st["selected_sids"]:
            st["selected_sids"].append(sid)
        sent = st["sentences"].get(sid) or {}
        frontier: list[str] = []
        for eid, rec in st["entities"].items():
            if sid in (rec.get("sids") or []) and eid not in frontier:
                frontier.append(eid)
        st["frontier_eids"] = frontier
        obs = f"SELECT sid=\"{sid}\": {str(sent.get('text') or '')[:240]}"
        exec_ok = True
    elif name == "lookup":
        eid = str(args.get("eid") or args.get("id") or "")
        entities = st.get("entities") or {}
        if not eid:
            return _fail(st, name, args, code="missing_eid", msg="lookup requires eid.")
        if eid not in entities:
            return _fail(st, name, args, code="eid_not_found", msg=f"eid `{eid}` is not a known entity.")
        if _mask_on(st, "lookup_dedup") and eid in set(st.get("visited_eids") or []):
            _note_effect(st, "lookup_dedup_blocked")
            return _fail(
                st,
                name,
                args,
                code="eid_already_visited",
                msg=f"eid `{eid}` was already looked up (lookup_dedup).",
            )
        if eid not in st["visited_eids"]:
            st["visited_eids"].append(eid)
        new_doc_order: list[str] = []
        added = 0
        if searcher is not None and getattr(searcher, "name", "none") != "none":
            lookup_q = _lookup_query(st, eid)
            hits = searcher.search(lookup_q, int(search_k))
            added = _merge_search_hits(st, hits)
            st["real_search_calls"] = int(st.get("real_search_calls") or 0) + 1
            new_doc_order = [str(h.docid) for h in hits]
            _note_effect(st, "lookup_retrieval_hits", len(hits))
            _note_effect(st, "lookup_new_docs", added)
        st["n_search_calls"] = int(st.get("n_search_calls") or 0) + 1
        st["search_count"] = int(st.get("search_count") or 0) + 1
        st["visible_sids"] = _lookup_sids(st, eid, new_doc_order=new_doc_order or None)
        rec = st["entities"].get(eid) or {}
        if not st["visible_sids"]:
            obs = (
                f"LOOKUP eid=\"{eid}\" surface=\"{rec.get('surface')}\": "
                f"no new sentences (added_docs={added})."
            )
        else:
            obs = (
                f"LOOKUP eid=\"{eid}\" surface=\"{rec.get('surface')}\": "
                f"{len(st['visible_sids'])} sentences (added_docs={added})."
            )
        exec_ok = True
    elif name == "answer_with":
        sids_arg = list(args.get("sids") or [])
        sid = str(args.get("sid") or (sids_arg[0] if sids_arg else ""))
        visible = set(st.get("visible_sids") or [])
        selected = set(st.get("selected_sids") or [])
        if not sid or sid not in st["sentences"]:
            return _fail(st, name, args, code="sid_not_found", msg=f"answer_with sid `{sid}` is not valid.")
        if sid not in visible and sid not in selected:
            return _fail(st, name, args, code="sid_not_selectable", msg=f"answer_with sid `{sid}` is not selectable.")
        if sid not in st["selected_sids"]:
            st["selected_sids"].append(sid)
        st["ended"] = True
        st["end_reason"] = "answer_with"
        st["action_map"] = {}
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
        return st, f"ANSWER_WITH selected={st['selected_sids'][:8]}", True
    elif name == "answer":
        st["ended"] = True
        st["end_reason"] = str(args.get("reason") or args.get("reasoning") or "answer")
        st["action_map"] = {}
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
        return st, f"ANSWER selected={st['selected_sids'][:8]}", True
    else:
        return _fail(st, name, args, code="unhandled_tool", msg=f"unhandled tool `{name}`.")

    if not st.get("ended"):
        st["action_map"] = build_action_map(st, include_answer=include_answer)
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
    return st, obs + "\n" + wm_text(st), exec_ok


def curated_recall(state: dict[str, Any], gold_ids: list[str]) -> float | None:
    from trim.eval.local_search_env import curated_recall as _h1_recall

    return _h1_recall(state, gold_ids)


def is_g_state(state: Mapping[str, Any] | None) -> bool:
    if not state:
        return False
    if "sentences" in state or "action_map" in state:
        return True
    return is_harness_g(mask=state.get("harness_mask"))
