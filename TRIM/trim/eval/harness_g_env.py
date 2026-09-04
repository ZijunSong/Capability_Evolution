"""TRIM-native Harness-G episode over a per-query doc_store.

Mirrors ``local_search_env`` for Harness-1. The always-on runtime is
INIT / SELECT / LOOKUP / ANSWER. Advanced components (answer_with, bridges,
synonyms, neighbors, hybrid INIT, SNC previews, …) are mask-gated, the same
way V8D flags sit on top of Harness-1's search/curate/end_search tools.
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


def _sentences_from_store(doc_store: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sentences: dict[str, dict[str, Any]] = {}
    for did, rec in (doc_store or {}).items():
        text = _doc_text(rec)
        parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
        if not parts:
            parts = [text.strip()] if text.strip() else []
        for i, sent in enumerate(parts[:12]):
            sid = f"{did}:s{i}"
            sentences[sid] = {
                "sid": sid,
                "doc_id": str(did),
                "text": sent,
                "idx": i,
                "neighbors": [f"{did}:s{j}" for j in (i - 1, i + 1) if 0 <= j < min(12, len(parts))],
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
    # Cheap synonym links: shared first token.
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


def _rank_sids(query: str, sentences: Mapping[str, Mapping[str, Any]], k: int) -> list[str]:
    q = _tokenize(query)
    scored: list[tuple[float, str]] = []
    for sid, sent in sentences.items():
        toks = _tokenize(str(sent.get("text") or ""))
        inter = len(q & toks)
        scored.append((inter / max(1, len(q)), sid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [sid for _, sid in scored[:k]]


def _hybrid_rank_sids(
    query: str,
    sentences: Mapping[str, Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, Any]],
    k: int,
) -> list[str]:
    lexical = _rank_sids(query, sentences, k=max(k, 8))
    q_low = query.lower()
    entity_hits: list[str] = []
    for rec in entities.values():
        if rec["surface"].lower() in q_low:
            entity_hits.extend(rec["sids"])
    fused: list[str] = []
    for sid in lexical + entity_hits:
        if sid not in fused:
            fused.append(sid)
        if len(fused) >= k:
            break
    return fused[:k]


def new_state(
    query: str,
    doc_store: dict[str, Any],
    *,
    harness_mask: Mapping[str, bool] | None = None,
    visible_k: int = 6,
) -> dict[str, Any]:
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


def build_action_map(state: dict[str, Any], *, include_answer: bool) -> dict[str, dict[str, Any]]:
    """Build the current menu. Advanced entries appear only when their mask is on."""
    menu: dict[str, dict[str, Any]] = {}
    n = 0
    selected = set(state.get("selected_sids") or [])
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
        visible = set(state.get("visible_sids") or [])
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
    """Expose selected sentences as curated docs so TRIM reward/metrics still work."""
    curated: dict[str, Any] = {}
    store = state.get("doc_store") or {}
    sentences = state.get("sentences") or {}
    pool = dict(state.get("pool") or {})
    for sid in state.get("selected_sids") or []:
        sent = sentences.get(sid) or {}
        did = str(sent.get("doc_id") or sid)
        rec = store.get(did) or {"id": did, "text": sent.get("text") or ""}
        curated[did] = rec if isinstance(rec, dict) else {"id": did, "text": str(rec)}
        pool[sid] = {"id": sid, "text": sent.get("text") or ""}
    for sid in state.get("visible_sids") or []:
        sent = sentences.get(sid) or {}
        pool[sid] = {"id": sid, "text": sent.get("text") or ""}
    state["curated"] = curated
    state["pool"] = pool
    state["curated_ids"] = list(curated)


def wm_text(state: dict[str, Any], *, auto_on: bool = False) -> str:
    del auto_on
    sentences = state.get("sentences") or {}
    entities = state.get("entities") or {}
    lines = [
        "[Harness-G Working Memory]",
        f"step={state.get('step', 0)} initialized={bool(state.get('initialized'))}",
        f"n_selected={len(state.get('selected_sids') or [])} n_visible={len(state.get('visible_sids') or [])}",
        "selected:",
    ]
    for sid in (state.get("selected_sids") or [])[:8]:
        lines.append(f"  - {sid}: {str((sentences.get(sid) or {}).get('text') or '')[:160]}")
    lines.append("visible:")
    for sid in (state.get("visible_sids") or [])[:8]:
        lines.append(f"  - {sid}: {str((sentences.get(sid) or {}).get('text') or '')[:120]}")
    lines.append("actions:")
    for aid, action in (state.get("action_map") or {}).items():
        extra = ""
        if action.get("sid"):
            extra = action["sid"]
        elif action.get("eid"):
            extra = str(action.get("entity_surface") or action["eid"])
        preview = action.get("snc_preview")
        score = f" snc={preview}" if preview is not None else ""
        lines.append(f"  {aid} = {action.get('type')} {extra}{score}".rstrip())
    hist = state.get("tool_history") or []
    lines.append("tool_history: " + ", ".join(str(h.get("name")) for h in hist[-8:]))
    del entities
    return "\n".join(lines)


def _init_visible(state: dict[str, Any]) -> list[str]:
    sentences = state.get("sentences") or {}
    entities = state.get("entities") or {}
    query = str(state.get("query") or "")
    k = 6
    if _mask_on(state, "hybrid_init_retrieve"):
        return _hybrid_rank_sids(query, sentences, entities, k)
    return _rank_sids(query, sentences, k)


def _lookup_sids(state: dict[str, Any], eid: str) -> list[str]:
    entities = state.get("entities") or {}
    sentences = state.get("sentences") or {}
    rec = entities.get(eid) or {}
    sids = list(rec.get("sids") or [])
    if _mask_on(state, "entity_synonyms"):
        for syn in rec.get("synonyms") or []:
            sids.extend((entities.get(syn) or {}).get("sids") or [])
    if _mask_on(state, "sentence_neighbors"):
        extra: list[str] = []
        for sid in list(sids):
            extra.extend((sentences.get(sid) or {}).get("neighbors") or [])
        sids.extend(extra)
    out: list[str] = []
    for sid in sids:
        if sid in sentences and sid not in out:
            out.append(sid)
        if len(out) >= 6:
            break
    return out


def execute_tool(
    state: dict[str, Any],
    name: str | None,
    args: dict[str, Any] | None,
    *,
    searcher: Any | None = None,
    search_k: int = 10,
) -> tuple[dict[str, Any], str, bool]:
    del searcher, search_k
    st = dict(state)
    st["sentences"] = dict(state.get("sentences") or {})
    st["entities"] = dict(state.get("entities") or {})
    st["selected_sids"] = list(state.get("selected_sids") or [])
    st["visible_sids"] = list(state.get("visible_sids") or [])
    st["visited_eids"] = list(state.get("visited_eids") or [])
    st["frontier_eids"] = list(state.get("frontier_eids") or [])
    st["tool_history"] = list(state.get("tool_history") or [])
    st["harness_mask"] = dict(state.get("harness_mask") or zero_mask_for("Harness-G"))
    st["step"] = int(state.get("step") or 0) + 1
    st["n_tool_calls"] = int(state.get("n_tool_calls") or 0) + 1
    args = dict(args or {})

    # Map ephemeral menu ids (A0) onto semantic tools.
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
        st["invalid_tools"] = int(state.get("invalid_tools") or 0) + 1
        obs = "ERROR: invalid tool `answer_with` (advanced component off)."
        st["tool_history"].append({"name": name, "legal": False})
        _sync_curated(st)
        return st, obs, False

    legal = set(RUNTIME_TOOLS) | ({"answer_with"} if _mask_on(st, "answer_with") else set())
    if name not in legal:
        st["invalid_tools"] = int(state.get("invalid_tools") or 0) + 1
        obs = f"ERROR: invalid tool `{name}`."
        st["tool_history"].append({"name": name, "legal": False})
        _sync_curated(st)
        return st, obs, False

    include_answer = True
    if name in {"init"} or not st.get("initialized"):
        if name in {"init", "select", "lookup", "answer", "answer_with"} and not st.get("initialized"):
            st["initialized"] = True
            st["visible_sids"] = _init_visible(st)
            st["n_search_calls"] = int(st.get("n_search_calls") or 0) + 1
            st["search_count"] = int(st.get("search_count") or 0) + 1
            include_answer = False
            if name == "init":
                st["action_map"] = build_action_map(st, include_answer=False)
                st["tool_history"].append({"name": "init", "legal": True, "args": {}})
                _sync_curated(st)
                return st, "INIT retrieved visible sentences.\n" + wm_text(st), True

    if name == "select":
        sid = str(args.get("sid") or args.get("id") or "")
        if sid and sid not in st["selected_sids"] and sid in st["sentences"]:
            st["selected_sids"].append(sid)
        sent = st["sentences"].get(sid) or {}
        frontier: list[str] = []
        for eid, rec in st["entities"].items():
            if sid in (rec.get("sids") or []) and eid not in frontier:
                frontier.append(eid)
        st["frontier_eids"] = frontier
        include_answer = True
        obs = f"SELECT {sid}: {str(sent.get('text') or '')[:240]}"
    elif name == "lookup":
        eid = str(args.get("eid") or args.get("id") or "")
        st["n_search_calls"] = int(st.get("n_search_calls") or 0) + 1
        st["search_count"] = int(st.get("search_count") or 0) + 1
        if eid and eid not in st["visited_eids"]:
            st["visited_eids"].append(eid)
        st["visible_sids"] = _lookup_sids(st, eid)
        rec = st["entities"].get(eid) or {}
        obs = (
            f"LOOKUP {eid} ({rec.get('surface')}): "
            f"{len(st['visible_sids'])} sentences"
        )
    elif name == "answer_with":
        sids = [str(x) for x in (args.get("sids") or ([args.get("sid")] if args.get("sid") else []))]
        for sid in sids:
            if sid and sid not in st["selected_sids"] and sid in st["sentences"]:
                st["selected_sids"].append(sid)
        st["ended"] = True
        st["end_reason"] = "answer_with"
        st["action_map"] = {}
        _sync_curated(st)
        st["tool_history"].append({"name": name, "legal": True, "args": args})
        return st, f"ANSWER_WITH selected={st['selected_sids'][:8]}", True
    elif name == "answer":
        st["ended"] = True
        st["end_reason"] = str(args.get("reason") or args.get("reasoning") or "answer")
        st["action_map"] = {}
        _sync_curated(st)
        st["tool_history"].append({"name": name, "legal": True, "args": args})
        return st, f"ANSWER selected={st['selected_sids'][:8]}", True
    else:
        obs = f"ok {name}"

    if not st.get("ended"):
        st["action_map"] = build_action_map(st, include_answer=include_answer)
    st["tool_history"].append({"name": name, "legal": True, "args": args})
    _sync_curated(st)
    return st, obs + "\n" + wm_text(st), True


def curated_recall(state: dict[str, Any], gold_ids: list[str]) -> float | None:
    from trim.eval.local_search_env import curated_recall as _h1_recall

    return _h1_recall(state, gold_ids)


def is_g_state(state: Mapping[str, Any] | None) -> bool:
    if not state:
        return False
    if "sentences" in state or "action_map" in state:
        return True
    return is_harness_g(mask=state.get("harness_mask"))
