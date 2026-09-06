"""LOCAL_COMPAT_ONLY in-process tool executor over a per-query doc_store."""

from __future__ import annotations

import json
import re
from typing import Any


def _doc_text(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return str(doc.get("text") or doc.get("content") or doc.get("snippet") or json.dumps(doc)[:2000])
    return str(doc)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]{2,}", (text or "").lower()))


def rank_docs(query: str, doc_store: dict[str, Any], k: int = 8) -> list[tuple[str, str, float]]:
    q = _tokenize(query)
    scored: list[tuple[float, str, str]] = []
    for did, doc in (doc_store or {}).items():
        text = _doc_text(doc)
        toks = _tokenize(text)
        inter = len(q & toks)
        denom = max(1, len(q))
        score = inter / denom
        scored.append((score, str(did), text))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for score, did, text in scored[:k]:
        out.append((did, text, float(score)))
    return out


def format_hits(hits: list[tuple[str, str, float]], *, n_chars: int = 280) -> str:
    lines = []
    for did, text, score in hits:
        snippet = re.sub(r"\s+", " ", text)[:n_chars]
        lines.append(f"- {did} (score={score:.3f}): {snippet}")
    return "\n".join(lines) if lines else "(no hits)"


def new_state(query: str, doc_store: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": query,
        "step": 0,
        "pool": {},
        "curated": {},
        "importance": {},
        "search_count": 0,
        "tool_history": [],
        "first_search_pending": True,
        "auto_seed": None,
        "ended": False,
        "end_reason": None,
        "doc_store": doc_store or {},
        "invalid_tools": 0,
        "n_tool_calls": 0,
        "n_search_calls": 0,
    }


def apply_auto_populate(state: dict[str, Any], *, top_k: int = 8) -> dict[str, Any]:
    """State-time AUTO privilege: copy top pool docs into curated at fair."""
    st = dict(state)
    st["pool"] = dict(state.get("pool") or {})
    st["curated"] = dict(state.get("curated") or {})
    st["importance"] = dict(state.get("importance") or {})
    ranked = sorted(st["pool"].items(), key=lambda kv: -float((kv[1] or {}).get("score") or 0.0))
    seed = []
    for did, rec in ranked[:top_k]:
        if did not in st["curated"]:
            st["curated"][did] = rec
            st["importance"][did] = "fair"
            seed.append(did)
    st["auto_seed"] = seed
    st["first_search_pending"] = False
    return st


POOL_DISPLAY_FULL = 50
POOL_DISPLAY_COMPACT = 30


def wm_text(state: dict[str, Any], *, auto_on: bool) -> str:
    curated = state.get("curated") or {}
    pool = state.get("pool") or {}
    curated_set = set(curated)
    uncurated = [did for did in pool if did not in curated_set]
    lines = [
        "[Working Memory]",
        f"step={state.get('step', 0)}",
        f"first-search-pending={bool(state.get('first_search_pending'))}",
        f"prior-search-count={int(state.get('search_count') or 0)}",
        f"auto_populate_first_search={'ON' if auto_on else 'OFF'}",
        f"auto_seed={state.get('auto_seed') if auto_on else None}",
        f"mask={'full' if auto_on else 'reduced/no-AUTO'}",
        f"n_curated={len(curated)} n_pool={len(pool)}",
        "curated:",
    ]
    if curated:
        for did, rec in list(curated.items())[:12]:
            snippet = re.sub(r"\s+", " ", _doc_text(rec))[:180]
            imp = (state.get("importance") or {}).get(did)
            lines.append(f"  - {did} importance={imp}: {snippet}")
    else:
        lines.append("  (empty -- use curate tool to add relevant docs)")
    lines.append(
        f"Document Pool: {len(pool)} docs total, {len(uncurated)} uncurated"
    )
    if uncurated:
        recent = list(reversed(uncurated[-POOL_DISPLAY_FULL:]))
        for did in recent:
            snippet = re.sub(r"\s+", " ", _doc_text(pool.get(did)))[:120]
            lines.append(f"  [ ] {did}: {snippet}")
        hidden = len(uncurated) - len(recent)
        if hidden > 0:
            older = uncurated[:hidden]
            id_str = ", ".join(str(x) for x in older[:POOL_DISPLAY_COMPACT])
            if hidden > POOL_DISPLAY_COMPACT:
                id_str += f" (+{hidden - POOL_DISPLAY_COMPACT} more)"
            lines.append(f"  Earlier uncurated ({hidden}): {id_str}")
    hist = state.get("tool_history") or []
    lines.append("tool_history: " + ", ".join(str(h.get("name")) for h in hist[-8:]))
    return "\n".join(lines)


def _harness_status(state: dict[str, Any], name: str) -> str:
    """Base Harness-1 observation footer (not a v8d component)."""
    if name == "end_search":
        return ""
    names = [str(h.get("name")) for h in (state.get("tool_history") or [])] + [str(name)]
    n_pool = len(state.get("pool") or {})
    n_curated = len(state.get("curated") or {})
    step = int(state.get("step") or 0)
    turns_since_curate = 0
    for item in reversed(names):
        if item == "curate":
            break
        turns_since_curate += 1
    is_search = name in {"search_corpus", "grep_corpus", "fan_out_search"}
    lines: list[str] = []
    if n_curated == 0:
        lines.append(
            f"[WARN] Curated set is EMPTY (0 docs). You have {n_pool} docs in your pool "
            "— curate ALL promising ones now."
        )
    if is_search:
        lines.append(
            "[ACTION REQUIRED] You just searched — now curate ALL plausibly relevant "
            "docs before your next search."
        )
    if turns_since_curate >= 2 and n_pool > 0:
        lines.append(
            f"[WARN] {turns_since_curate} consecutive non-curate turns. "
            "You MUST curate before your next search."
        )
    if n_curated == 0 and n_pool >= 3:
        lines.append("[NEXT] curate promising docs from your pool.")
    elif is_search:
        lines.append("[NEXT] curate ALL relevant docs from these results NOW.")
    if n_curated > 0 and step >= 28:
        lines.append(
            "[TIP] If the curated set answers the question, call end_search."
        )
    return "\n".join(lines)


_SEARCH_QUERY_KEYS = ("query", "q", "text", "search_query", "keywords")


def _as_str_list(value: Any, *, limit: int | None = None) -> list[str]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, dict):
        items = list(value.values())
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        if isinstance(item, (list, dict, bool)) or item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _tool_search_queries(name: str, args: dict[str, Any], fallback: str) -> list[str]:
    raw: list[str] = []
    if name == "fan_out_search":
        raw.extend(_as_str_list(args.get("queries"), limit=8))
        extra = args.get("query") or args.get("q")
        if extra:
            raw.append(str(extra))
    elif name == "grep_corpus":
        raw.append(str(args.get("pattern") or args.get("query") or args.get("q") or ""))
    else:
        q = ""
        for key in _SEARCH_QUERY_KEYS:
            if args.get(key):
                q = str(args.get(key))
                break
        raw.append(q)
    cleaned = [item.strip() for item in raw if str(item or "").strip()]
    fb = str(fallback or "").strip()
    return cleaned or ([fb] if fb else [])


def _merge_hit(hits_all: dict[str, tuple[str, float]], did: str, text: str, score: float) -> None:
    if not did:
        return
    prev = hits_all.get(did)
    if prev is None or score > prev[1]:
        hits_all[did] = (text, score)


def execute_tool(
    state: dict[str, Any],
    name: str | None,
    args: dict[str, Any] | None,
    *,
    searcher: Any | None = None,
    search_k: int = 10,
) -> tuple[dict[str, Any], str, bool]:
    st = dict(state)
    st["pool"] = dict(state.get("pool") or {})
    st["curated"] = dict(state.get("curated") or {})
    st["importance"] = dict(state.get("importance") or {})
    st["tool_history"] = list(state.get("tool_history") or [])
    st["step"] = int(state.get("step") or 0) + 1
    st["n_tool_calls"] = int(state.get("n_tool_calls") or 0) + 1
    args = args or {}
    legal = name in {
        "fan_out_search",
        "search_corpus",
        "grep_corpus",
        "read_document",
        "review_docs",
        "curate",
        "verify",
        "end_search",
    }
    if not legal:
        st["invalid_tools"] = int(state.get("invalid_tools") or 0) + 1
        obs = f"ERROR: invalid tool `{name}`."
        st["tool_history"].append({"name": name, "legal": False})
        return st, obs, False

    st["doc_store"] = dict(state.get("doc_store") or {})
    store = st["doc_store"]
    obs = ""
    if name in {"search_corpus", "grep_corpus", "fan_out_search"}:
        st["n_search_calls"] = int(state.get("n_search_calls") or 0) + 1
        st["search_count"] = int(state.get("search_count") or 0) + 1
        st["first_search_pending"] = False
        queries = _tool_search_queries(name, args, str(state.get("query") or ""))
        hits_all: dict[str, tuple[str, float]] = {}
        live = searcher is not None and getattr(searcher, "name", "none") != "none"
        for q in queries:
            got_live = False
            if live:
                for hit in searcher.search(str(q), int(search_k)):
                    did = str(getattr(hit, "docid", "") or "")
                    text = str(getattr(hit, "text", "") or "")
                    score = float(getattr(hit, "score", 0.0) or 0.0)
                    if not did:
                        continue
                    _merge_hit(hits_all, did, text, score)
                    got_live = True
            if not got_live:
                for did, text, score in rank_docs(q, store, k=int(search_k)):
                    _merge_hit(hits_all, str(did), text, float(score))
        ranked = sorted(hits_all.items(), key=lambda item: -item[1][1])
        for did, (text, score) in ranked:
            rec = {"id": did, "text": text[:4000], "score": score}
            st["pool"][did] = rec
            store[did] = rec
        shown = ranked[: int(search_k)]
        obs = "Search results:\n" + format_hits([(d, t, s) for d, (t, s) in shown])
    elif name == "read_document":
        did = str(args.get("doc_id") or args.get("id") or "")
        rec = store.get(did) or st["pool"].get(did) or st["curated"].get(did)
        obs = f"Document {did}:\n{_doc_text(rec)[:4000]}" if rec is not None else f"Document {did} not found."
    elif name == "review_docs":
        ids = _as_str_list(args.get("doc_ids") or args.get("ids"), limit=8)
        parts = []
        for did in ids:
            rec = st["curated"].get(did) or st["pool"].get(did) or store.get(did)
            parts.append(f"{did}: {_doc_text(rec)[:800]}" if rec is not None else f"{did}: missing")
        obs = "Review:\n" + "\n".join(parts)
    elif name == "curate":
        add_ids = _as_str_list(args.get("add_ids") or args.get("doc_ids") or args.get("ids"))
        remove_ids = _as_str_list(args.get("remove_ids"))
        imp = args.get("importance") or {}
        for did in add_ids:
            rec = st["pool"].get(did) or store.get(did)
            if rec is not None:
                if not isinstance(rec, dict):
                    rec = {"id": did, "text": _doc_text(rec)}
                st["curated"][str(did)] = rec
                if isinstance(imp, dict) and str(did) in imp:
                    st["importance"][str(did)] = imp[str(did)]
        for did in remove_ids:
            st["curated"].pop(str(did), None)
        obs = f"Curated n={len(st['curated'])} ids={list(st['curated'])[:12]}"
    elif name == "verify":
        ids = _as_str_list(args.get("doc_ids") or args.get("ids"), limit=5)
        claim = str(args.get("claim") or "")
        ctoks = _tokenize(claim)
        parts = []
        for did in ids:
            rec = st["curated"].get(did) or st["pool"].get(did) or store.get(did)
            text = _doc_text(rec)
            hit = len(ctoks & _tokenize(text)) >= max(1, len(ctoks) // 4)
            parts.append(f"{did}: {'yes' if hit else 'no'}")
        obs = f"Verify claim={claim[:200]}\n" + "\n".join(parts)
    elif name == "end_search":
        st["ended"] = True
        st["end_reason"] = str(args.get("reasoning") or args.get("reason") or "")
        obs = f"end_search accepted. curated={list(st['curated'])[:12]}"
    footer = _harness_status(st, name)
    if footer:
        obs = f"{obs}\n{footer}".strip()
    st["tool_history"].append({"name": name, "legal": True, "args": args})
    return st, obs, True


def curated_recall(state: dict[str, Any], gold_ids: list[str]) -> float | None:
    if not gold_ids:
        return None
    gold = {str(x) for x in gold_ids}
    got: set[str] = set()
    for x in state.get("curated") or {}:
        got.add(str(x))
        text = str(x)
        if "::c" in text:
            got.add(text.rsplit("::c", 1)[0])
    return len(gold & got) / max(1, len(gold))
