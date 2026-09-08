"""Live Harness-1 component effects gated by ``state['harness_mask']``.

These are the in-process equivalents of V8D flags. A config-only mask is
not enough: search / curate / verify / WM rendering must actually branch.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from trim.adapters.components import full_mask, zero_mask

MAX_CURATED_DOCS = 30
AUTO_POPULATE_TOP_K = 8
VALID_IMPORTANCE = ("very_high", "high", "fair", "low")
_IMPORTANCE_RANK = {"very_high": 0, "high": 1, "fair": 2, "low": 3}
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|\d{4}(?:s)?|\d{1,2}/\d{1,2}/\d{2,4})\b"
)
_STOPWORDS = frozenset(
    {
        "The",
        "This",
        "That",
        "A",
        "An",
        "It",
        "He",
        "She",
        "In",
        "On",
        "At",
        "For",
        "By",
        "With",
        "To",
        "From",
        "I",
        "We",
        "You",
        "They",
        "But",
        "Page",
        "Section",
        "Document",
    }
)
BASE_TOOLS = frozenset(
    {
        "fan_out_search",
        "search_corpus",
        "grep_corpus",
        "read_document",
        "review_docs",
        "curate",
        "end_search",
    }
)
RERANK_INSTRUCTION = (
    "Given a hard multi-hop web query, retrieve passages that contain the specific "
    "entities, dates, quantities, or relationships asked about. Prefer passages "
    "that directly match multiple constraints simultaneously."
)


def resolve_mask(state: Mapping[str, Any] | None, override: Mapping[str, bool] | None = None) -> dict[str, bool]:
    if override is not None:
        return dict(override)
    raw = (state or {}).get("harness_mask")
    if isinstance(raw, Mapping):
        return dict(raw)
    return zero_mask()


def mask_on(state: Mapping[str, Any], component_id: str) -> bool:
    return bool(resolve_mask(state).get(component_id, False))


def mask_label(mask: Mapping[str, bool] | None) -> str:
    mask = dict(mask or {})
    if mask and all(mask.get(k) == v for k, v in full_mask().items()):
        return "full"
    if not mask or all(not bool(v) for v in mask.values()):
        return "zero"
    return "mixed"


def note_effect(state: dict[str, Any], component_id: str, *, n: int = 1) -> None:
    effects = dict(state.get("runtime_effects") or {})
    effects[component_id] = int(effects.get(component_id) or 0) + int(n)
    state["runtime_effects"] = effects


def legal_tool_set(mask: Mapping[str, bool] | None) -> set[str]:
    tools = set(BASE_TOOLS)
    if mask and mask.get("verify_tool"):
        tools.add("verify")
    return tools


def empty_effects(mask: Mapping[str, bool] | None = None) -> dict[str, int]:
    keys = list((mask or zero_mask()).keys()) or list(zero_mask().keys())
    return {cid: 0 for cid in keys}


def content_fingerprint(text: str) -> str:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    blob = " ".join(toks[:240])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def compress_snippet(query: str, text: str, *, max_chars: int = 280) -> str:
    from trim.training.sentence_compress_teacher import compress_text

    return compress_text(query, text or "", max_sents=4, max_chars=max_chars)


def extract_entities(text: str) -> set[str]:
    ents: set[str] = set()
    for match in _ENTITY_RE.finditer((text or "")[:8000]):
        ent = match.group(0).strip()
        if len(ent) < 2 or ent in _STOPWORDS:
            continue
        ents.add(ent)
    return ents


def update_evidence_graph(state: dict[str, Any], doc_id: str, text: str) -> None:
    graph = dict(state.get("evidence_graph") or {})
    entity_to_docs = {k: set(v) for k, v in (graph.get("entity_to_docs") or {}).items()}
    doc_to_entities = {k: set(v) for k, v in (graph.get("doc_to_entities") or {}).items()}
    if doc_id in doc_to_entities:
        return
    ents = extract_entities(text)
    if not ents:
        return
    doc_to_entities[doc_id] = ents
    for ent in ents:
        entity_to_docs.setdefault(ent, set()).add(doc_id)
    state["evidence_graph"] = {
        "entity_to_docs": {k: sorted(v) for k, v in entity_to_docs.items()},
        "doc_to_entities": {k: sorted(v) for k, v in doc_to_entities.items()},
    }
    note_effect(state, "evidence_graph")


def render_evidence_graph(state: Mapping[str, Any], *, max_entities: int = 8) -> str:
    graph = state.get("evidence_graph") or {}
    entity_to_docs = graph.get("entity_to_docs") or {}
    if not entity_to_docs:
        return ""
    ranked = sorted(entity_to_docs.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    bridge = [(e, docs) for e, docs in ranked if len(docs) >= 2][:max_entities]
    singleton = sum(1 for _, docs in ranked if len(docs) == 1)
    if not bridge:
        return f"[Evidence Graph] 0 bridge entities, {singleton} singleton entities."
    lines = ["[Evidence Graph] Entities appearing in multiple docs (bridges):"]
    for ent, docs in bridge:
        shown = list(docs)[:5]
        extra = f" (+{len(docs) - 5} more)" if len(docs) > 5 else ""
        lines.append(f"  {ent}: {', '.join(str(x) for x in shown)}{extra}")
    if singleton:
        lines.append(f"  ({singleton} entities in only 1 doc — potential hops)")
    return "\n".join(lines)


def neighbor_ids(doc_id: str, store: Mapping[str, Any]) -> list[str]:
    keys = [str(k) for k in store]
    if doc_id in keys:
        idx = keys.index(doc_id)
        out = []
        if idx > 0:
            out.append(keys[idx - 1])
        if idx + 1 < len(keys):
            out.append(keys[idx + 1])
        return out
    m = re.search(r"(::c)(\d+)$", str(doc_id))
    if not m:
        return []
    prefix, n = str(doc_id)[: m.start(2)], int(m.group(2))
    candidates = [f"{prefix}{n - 1}", f"{prefix}{n + 1}"]
    return [c for c in candidates if c in store]


def format_token_budget_marker(used_tokens: int, budget: int = 32768) -> str:
    used = max(0, int(used_tokens))
    pct = int(100.0 * used / max(budget, 1))
    flag = ""
    if pct >= 90:
        flag = " CRITICAL — end_search NOW"
    elif pct >= 75:
        flag = " warning: finish up soon"
    elif pct >= 60:
        flag = " over halfway"
    return f"[Context: {used}/{budget}{flag}]"


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def maybe_compress(query: str, text: str, state: Mapping[str, Any], *, max_chars: int) -> str:
    if not mask_on(state, "sentence_compress"):
        return text
    return compress_snippet(query, text, max_chars=max_chars)


def filter_dedup_hits(
    state: dict[str, Any],
    ranked: list[tuple[str, tuple[str, float]]],
) -> list[tuple[str, tuple[str, float]]]:
    if not mask_on(state, "content_dedup"):
        return ranked
    seen = set(state.get("dedup_fingerprints") or [])
    kept: list[tuple[str, tuple[str, float]]] = []
    dropped: list[str] = list(state.get("dedup_dropped") or [])
    for did, (text, score) in ranked:
        fp = content_fingerprint(text)
        if fp in seen:
            dropped.append(str(did))
            note_effect(state, "content_dedup")
            continue
        seen.add(fp)
        kept.append((did, (text, score)))
    state["dedup_fingerprints"] = sorted(seen)
    state["dedup_dropped"] = dropped
    return kept


def apply_auto_populate(state: dict[str, Any], *, top_k: int = AUTO_POPULATE_TOP_K) -> dict[str, Any]:
    """Copy top pool docs into curated at fair. No-op if already seeded or curated."""
    st = dict(state)
    st["pool"] = dict(state.get("pool") or {})
    st["curated"] = dict(state.get("curated") or {})
    st["importance"] = dict(state.get("importance") or {})
    st["runtime_effects"] = dict(state.get("runtime_effects") or empty_effects(resolve_mask(state)))
    if st.get("auto_seed") or st["curated"]:
        st["first_search_pending"] = False
        return st
    ranked = sorted(st["pool"].items(), key=lambda kv: -float((kv[1] or {}).get("score") or 0.0))
    seed: list[str] = []
    for did, rec in ranked[: int(top_k)]:
        if did in st["curated"]:
            continue
        st["curated"][did] = rec
        if mask_on(st, "importance_tagging"):
            st["importance"][did] = "fair"
            note_effect(st, "importance_tagging")
        seed.append(str(did))
    st["auto_seed"] = seed
    st["first_search_pending"] = False
    if seed:
        note_effect(st, "auto_populate_first_search", n=len(seed))
    return st


def curate_with_mask(
    state: dict[str, Any],
    *,
    add_ids: list[str],
    remove_ids: list[str],
    importance: Mapping[str, Any] | None,
    store: Mapping[str, Any],
) -> str:
    st = state
    for did in remove_ids:
        key = str(did)
        st["curated"].pop(key, None)
        st["importance"].pop(key, None)
    imp_norm: dict[str, str] = {}
    if importance and mask_on(st, "importance_tagging"):
        for key, value in dict(importance).items():
            tag = str(value).strip().lower()
            if tag not in VALID_IMPORTANCE:
                tag = "fair"
            imp_norm[str(key)] = tag
            note_effect(st, "importance_tagging")
    evicted: list[str] = []
    dropped: list[str] = []
    for raw in add_ids:
        did = str(raw)
        rec = st["pool"].get(did) or store.get(did)
        if rec is None:
            continue
        if not isinstance(rec, dict):
            rec = {"id": did, "text": str(rec)}
        incoming = imp_norm.get(did, "fair")
        if did in st["curated"]:
            if did in imp_norm:
                st["importance"][did] = incoming
            continue
        if len(st["curated"]) < MAX_CURATED_DOCS:
            st["curated"][did] = rec
            if mask_on(st, "importance_tagging"):
                st["importance"][did] = incoming
            continue
        if mask_on(st, "subtractive_curation"):
            incoming_rank = _IMPORTANCE_RANK.get(incoming, 2)
            worst_id = None
            worst_rank = -1
            for cid in list(st["curated"]):
                rank = _IMPORTANCE_RANK.get(st["importance"].get(cid, "fair"), 2)
                if rank > worst_rank:
                    worst_rank = rank
                    worst_id = cid
            if worst_id is not None and worst_rank > incoming_rank:
                st["curated"].pop(worst_id, None)
                st["importance"].pop(worst_id, None)
                evicted.append(str(worst_id))
                st["curated"][did] = rec
                if mask_on(st, "importance_tagging"):
                    st["importance"][did] = incoming
                note_effect(st, "subtractive_curation")
                continue
        dropped.append(did)
    rendered = list(st["curated"])[:12]
    if mask_on(st, "importance_tagging") and st["importance"]:
        rendered = [
            f"{did}[{st['importance'].get(did, 'fair')}]" for did in list(st["curated"])[:12]
        ]
    obs = f"Curated n={len(st['curated'])} ids={rendered}"
    if evicted:
        obs += f"\n[EVICTED low-importance] {evicted[:5]}"
    if dropped:
        obs += f"\n[CAPACITY] not added: {dropped[:5]}"
    return obs


def decorate_observation(state: dict[str, Any], obs: str, *, query: str, name: str) -> str:
    parts = [obs]
    if mask_on(state, "adaptive_rerank_instruction") and name in {
        "search_corpus",
        "grep_corpus",
        "fan_out_search",
    }:
        state["rerank_instruction"] = RERANK_INSTRUCTION
        parts.append("[Rerank instruction] " + RERANK_INSTRUCTION)
        note_effect(state, "adaptive_rerank_instruction")
    if mask_on(state, "evidence_graph"):
        graph_txt = render_evidence_graph(state)
        if graph_txt:
            parts.append(graph_txt)
    if mask_on(state, "chunk_neighbors") and name in {"read_document", "search_corpus", "review_docs"}:
        store = state.get("doc_store") or {}
        ids = list((state.get("pool") or {}))[:6]
        neighbor_lines = []
        for did in ids:
            neigh = neighbor_ids(str(did), store)
            if neigh:
                neighbor_lines.append(f"  {did} -> {', '.join(neigh)}")
        if neighbor_lines:
            parts.append("[Chunk neighbors]\n" + "\n".join(neighbor_lines))
            note_effect(state, "chunk_neighbors")
    if mask_on(state, "token_budget_marker"):
        used = estimate_tokens("\n".join(parts)) + 32 * int(state.get("step") or 0)
        marker = format_token_budget_marker(used)
        state["token_budget_marker"] = marker
        parts.append(marker)
        note_effect(state, "token_budget_marker")
    del query
    return "\n".join(p for p in parts if p).strip()
