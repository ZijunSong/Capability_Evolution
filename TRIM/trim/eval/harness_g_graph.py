"""Offline / incremental Harness-G sentence–entity graph for BC+ adapters.

The official Harness-G index is a corpus-level graph. TRIM can attach a
prebuilt index (``scope=corpus``) or build an episode graph from the current
doc_store (``scope=episode_doc_store``). LOOKUP ranks graph candidates; it
does not truncate by document prefix.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from trim.eval.local_search_env import _doc_text, _tokenize

MAX_UNIT_CHARS = 400
QUESTION_WORD_BUDGET = 256
EVIDENCE_WORD_BUDGET = 64

_SENT_SPLIT = re.compile(r"(?<=[.!?])[ \t]+")
_YAML_FRONT_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)
_METADATA_ONLY_RE = re.compile(
    r"^(title|author|date|published|copyright|table of contents|references):\s*.+\s*$",
    re.I,
)
_NAV_LINE_RE = re.compile(
    r"^(skip to (main )?content|toggle (navigation|menu)|our mission|"
    r"e-?newsletter|subscribe( now)?|sign[- ]?in|log[- ]?in|"
    r"cookie(s)?( policy| settings)?|back to top|table of contents|"
    r"main menu|home|search|share this|follow us|all rights reserved)$",
    re.I,
)
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2}|[A-Z]{2,}|[A-Z][a-z]+(?:[-'][A-Za-z]+)?(?:[ \t]+[A-Z][a-z]+(?:[-'][A-Za-z]+)?){0,3})\b"
)
_ENTITY_STOP = frozenset(
    {
        "the", "a", "an", "for", "from", "and", "or", "but", "with", "without",
        "about", "into", "over", "after", "before", "between", "under", "this",
        "that", "these", "those", "our", "your", "their", "his", "her", "its",
        "article", "year", "page", "home", "menu", "news", "more", "click",
        "here", "read", "share", "follow", "subscribe", "copyright", "all",
        "rights", "reserved", "table", "contents", "references", "related",
        "links", "next", "previous", "back", "top", "search", "login", "sign",
        "contact", "privacy", "terms", "cookie", "cookies", "skip", "content",
        "mission", "newsletter", "january", "february", "march", "april",
        "june", "july", "august", "september", "october", "november",
        "december", "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday",
    }
)
_TWO_LETTER_KEEP = frozenset({"uk", "us", "un", "eu", "ai"})
_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("us", "usa", "united_states", "u.s.", "u.s.a."),
    ("uk", "united_kingdom", "britain", "great_britain"),
    ("un", "united_nations"),
    ("eu", "european_union"),
    ("nasa", "national_aeronautics_and_space_administration"),
)


def text_hash(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:16]


def strip_yaml_front_matter(text: str) -> str:
    return _YAML_FRONT_RE.sub("", str(text or ""), count=1)


def is_metadata_only_line(text: str) -> bool:
    line = str(text or "").strip()
    if not line:
        return True
    if _NAV_LINE_RE.match(line):
        return True
    if _METADATA_ONLY_RE.match(line) and (len(line) < 80) and (line.count(".") + line.count("!") + line.count("?") <= 1):
        return True
    return False


def split_long_unit(text: str, *, max_chars: int = MAX_UNIT_CHARS) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for word in words:
        extra = len(word) + (1 if cur else 0)
        if cur and n + extra > max_chars:
            chunks.append(" ".join(cur))
            cur = [word]
            n = len(word)
        else:
            cur.append(word)
            n += extra
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


def text_to_sentence_parts(text: str, *, max_unit_chars: int = MAX_UNIT_CHARS) -> list[str]:
    text = strip_yaml_front_matter(text)
    parts: list[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        for piece in _SENT_SPLIT.split(block):
            piece = piece.strip()
            if piece:
                parts.append(piece)
    if not parts and text.strip():
        parts = [text.strip()]
    kept = [p for p in parts if not is_metadata_only_line(p)]
    source = kept or parts
    out: list[str] = []
    for part in source:
        out.extend(split_long_unit(part, max_chars=max_unit_chars))
    return out


def canonical_eid(surface: str) -> str:
    return "e:" + re.sub(r"\s+", "_", surface.strip().lower())


def _alias_map() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for group in _ALIAS_GROUPS:
        members = {f"e:{name}" for name in group}
        for eid in members:
            out[eid] = set(members)
    return out


_ALIAS_MAP = _alias_map()


def keep_entity_surface(surface: str) -> bool:
    token = surface.strip()
    if not token:
        return False
    low = token.lower()
    if low in _ENTITY_STOP:
        return False
    if len(token) == 2:
        return token.isupper() and low in _TWO_LETTER_KEEP
    if len(token) < 3:
        return False
    return True


def normalize_entity_surface(surface: str) -> str:
    toks = [t for t in str(surface or "").split() if t]
    while toks and toks[0].lower() in _ENTITY_STOP:
        toks = toks[1:]
    return " ".join(toks).strip()


def lexical_score(query: str, text: str) -> float:
    q = _tokenize(query)
    if not q:
        return 0.0
    t = _tokenize(text)
    return len(q & t) / max(1, len(q))


def mixquery_text(
    question: str,
    evidence_texts: Iterable[str],
    *,
    question_word_budget: int = QUESTION_WORD_BUDGET,
    evidence_word_budget: int = EVIDENCE_WORD_BUDGET,
) -> dict[str, Any]:
    q_words = str(question or "").split()[: max(1, int(question_word_budget))]
    ev_words: list[str] = []
    for text in evidence_texts:
        ev_words.extend(str(text or "").split())
        if len(ev_words) >= int(evidence_word_budget):
            break
    ev_words = ev_words[: max(0, int(evidence_word_budget))]
    query = " ".join(q_words + ev_words).strip()
    return {
        "query": query,
        "question_words_used": len(q_words),
        "evidence_words_used": len(ev_words),
        "question_word_budget": int(question_word_budget),
        "evidence_word_budget": int(evidence_word_budget),
    }


GRAPH_FORMAT = "harness_g_graph_v1"
GRAPH_MAX_ENTITY_DOCS = 200
GRAPH_MAX_ENTITY_SIDS = GRAPH_MAX_ENTITY_DOCS * 400


class HarnessGGraphIndex:
    """Sentence / entity graph with incremental document ingest."""

    def __init__(self, *, scope: str = "episode_doc_store") -> None:
        self.scope = str(scope or "episode_doc_store")
        self.docs: dict[str, dict[str, Any]] = {}
        self.sentences: dict[str, dict[str, Any]] = {}
        self.entities: dict[str, dict[str, Any]] = {}
        self.sentence_to_entities: dict[str, list[str]] = {}
        self.parent_docids: dict[str, str] = {}
        self.doc_to_sids: dict[str, list[str]] = {}
        self.source_corpus: str | None = None
        self.source_path: str | None = None
        self._fingerprint: str | None = None

    def clone_overlay(self) -> "HarnessGGraphIndex":
        """Episode view over a graph. Top-level maps are copied; nested records are shared."""
        other = HarnessGGraphIndex(scope=self.scope)
        other.docs = dict(self.docs)
        other.sentences = dict(self.sentences)
        other.entities = dict(self.entities)
        other.sentence_to_entities = dict(self.sentence_to_entities)
        other.parent_docids = dict(self.parent_docids)
        other.doc_to_sids = dict(self.doc_to_sids)
        other.source_corpus = self.source_corpus
        other.source_path = getattr(self, "source_path", None)
        other._fingerprint = self._fingerprint
        return other

    def content_fingerprint(self) -> str:
        if self._fingerprint:
            return self._fingerprint
        payload = {
            "scope": self.scope,
            "n_docs": len(self.docs),
            "n_sents": len(self.sentences),
            "n_ents": len(self.entities),
            "doc_hashes": sorted((did, rec.get("text_hash")) for did, rec in self.docs.items()),
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._fingerprint = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return self._fingerprint

    def metadata(self) -> dict[str, Any]:
        n_sent_edges = 0
        for sids in (self.doc_to_sids or {}).values():
            n = len(sids)
            if n > 1:
                n_sent_edges += 2 * (n - 1)
        n_syn_edges = sum(len(rec.get("synonyms") or []) for rec in self.entities.values())
        return {
            "graph_scope": self.scope,
            "graph_fingerprint": self.content_fingerprint(),
            "graph_num_docs": len(self.docs),
            "graph_num_sentences": len(self.sentences),
            "graph_num_entities": len(self.entities),
            "graph_num_edges": n_sent_edges + n_syn_edges,
            "graph_build_version": GRAPH_FORMAT,
            "graph_source_corpus": getattr(self, "source_corpus", None),
            "graph_source_path": getattr(self, "source_path", None),
        }

    def docs_for_entity(self, eid: str) -> list[str]:
        cache = self.__dict__.setdefault("_eid_to_docs", {})
        cached = cache.get(eid)
        if cached is not None:
            return cached
        rec = self.entities.get(eid) or {}
        out: list[str] = []
        seen: set[str] = set()
        for sid in rec.get("sids") or []:
            did = self.parent_docid(str(sid))
            if did and did not in seen:
                seen.add(did)
                out.append(did)
        cache[eid] = out
        return out

    def expand_docs(
        self,
        start_docids: Iterable[str],
        *,
        hops: int = 1,
        max_entity_docs: int | None = None,
        stop_docids: Iterable[str] | None = None,
    ) -> set[str]:
        frontier = {str(x) for x in start_docids if str(x)}
        reached = set(frontier)
        stop = {str(x) for x in (stop_docids or []) if str(x)}
        cap = None if max_entity_docs is None else max(1, int(max_entity_docs))
        for _ in range(max(0, int(hops))):
            if stop and reached & stop:
                break
            nxt: set[str] = set()
            eids: set[str] = set()
            for did in frontier:
                for sid in self.doc_to_sids.get(did) or []:
                    eids.update(self.sentence_to_entities.get(sid) or [])
            for eid in list(eids):
                rec = self.entities.get(eid) or {}
                for syn in rec.get("synonyms") or []:
                    eids.add(str(syn))
            for eid in eids:
                docs = self.docs_for_entity(eid)
                if cap is not None and len(docs) > cap:
                    continue
                for did in docs:
                    if did not in reached:
                        nxt.add(did)
                        if stop and did in stop:
                            reached.update(nxt)
                            return reached
            if not nxt:
                break
            reached.update(nxt)
            frontier = nxt
        return reached

    def ingest_documents(self, doc_store: Mapping[str, Any] | None) -> int:
        changed = 0
        for did, rec in (doc_store or {}).items():
            did = str(did)
            text = _doc_text(rec)
            digest = text_hash(text)
            prev = self.docs.get(did)
            if prev and prev.get("text_hash") == digest:
                continue
            if did in self.docs:
                self._drop_doc(did)
            self._add_doc(did, text, rec if isinstance(rec, Mapping) else {"id": did, "text": text})
            changed += 1
        if changed:
            self._fingerprint = None
            self.__dict__.pop("_eid_to_docs", None)
            self._link_aliases()
        return changed

    def _drop_doc(self, did: str) -> None:
        if did not in self.docs:
            return
        drop_sids = list(self.doc_to_sids.get(did) or [])
        drop_set = set(drop_sids)
        for sid in drop_sids:
            self.sentences.pop(sid, None)
            self.sentence_to_entities.pop(sid, None)
        stale: list[str] = []
        for eid, rec in self.entities.items():
            rec["sids"] = [sid for sid in rec.get("sids") or [] if sid not in drop_set]
            if "_sid_index" in rec:
                rec["_sid_index"] = set(rec["sids"])
            if not rec["sids"]:
                stale.append(eid)
        for eid in stale:
            self.entities.pop(eid, None)
        self.docs.pop(did, None)
        self.parent_docids.pop(did, None)
        self.doc_to_sids.pop(did, None)

    def _add_doc(self, did: str, text: str, rec: Mapping[str, Any]) -> None:
        parent = str(rec.get("parent_docid") or rec.get("id") or did)
        self.docs[did] = {"id": did, "text": text, "text_hash": text_hash(text), "parent_docid": parent}
        self.parent_docids[did] = parent
        parts = text_to_sentence_parts(text)
        offset = 0
        raw = str(text or "")
        sids: list[str] = []
        for i, part in enumerate(parts):
            start = raw.find(part, offset)
            if start < 0:
                start = offset
            end = start + len(part)
            offset = end
            sid = f"{did}:s{i}"
            neighbors = [f"{did}:s{j}" for j in (i - 1, i + 1) if 0 <= j < len(parts)]
            self.sentences[sid] = {
                "sid": sid,
                "doc_id": did,
                "parent_docid": parent,
                "text": part,
                "idx": i,
                "start": start,
                "end": end,
                "text_hash": text_hash(part),
                "neighbors": neighbors,
            }
            sids.append(sid)
            for surface in _ENTITY_RE.findall(part):
                surface = normalize_entity_surface(surface)
                if not keep_entity_surface(surface):
                    continue
                eid = canonical_eid(surface)
                ent = self.entities.setdefault(
                    eid,
                    {"eid": eid, "surface": surface.strip(), "sids": [], "synonyms": [], "_sid_index": set()},
                )
                index = ent.get("_sid_index")
                if not isinstance(index, set):
                    index = set(ent.get("sids") or [])
                    ent["_sid_index"] = index
                if sid not in index:
                    index.add(sid)
                    ent["sids"].append(sid)
                self.sentence_to_entities.setdefault(sid, [])
                if eid not in self.sentence_to_entities[sid]:
                    self.sentence_to_entities[sid].append(eid)
        self.doc_to_sids[did] = sids

    def _link_aliases(self) -> None:
        seen: set[str] = set()
        for eid, group in _ALIAS_MAP.items():
            if eid in seen:
                continue
            present = [member for member in sorted(group) if member in self.entities]
            for member in present:
                seen.add(member)
                rec = self.entities.get(member)
                if rec is not None:
                    rec["synonyms"] = [other for other in present if other != member]

    def get_entities_for_sentence(self, sid: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for eid in self.sentence_to_entities.get(sid) or []:
            rec = self.entities.get(eid)
            if rec:
                out.append(rec)
        return out

    def parent_docid(self, sid: str) -> str:
        sent = self.sentences.get(sid) or {}
        did = str(sent.get("parent_docid") or sent.get("doc_id") or "")
        if did:
            return did
        if ":" in sid:
            return sid.split(":", 1)[0]
        return sid

    def similar_entities(self, eid: str, *, limit: int = 6) -> list[str]:
        rec = self.entities.get(eid) or {}
        syns = [str(x) for x in (rec.get("synonyms") or []) if str(x) and str(x) != eid]
        # Official corpus has millions of entities; never scan them for token overlap.
        if len(self.entities) > 50_000:
            return syns[: max(0, int(limit))]
        surface_toks = _tokenize(str(rec.get("surface") or ""))
        scored: list[tuple[float, str]] = []
        for other_id, other in self.entities.items():
            if other_id == eid:
                continue
            toks = _tokenize(str(other.get("surface") or ""))
            if not toks or not surface_toks:
                continue
            inter = len(surface_toks & toks)
            if inter <= 0:
                continue
            scored.append((inter / max(1, len(surface_toks)), other_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [oid for _, oid in scored[:limit]]

    def propose_bridge_entities(
        self,
        frontier_eids: Iterable[str],
        question: str,
        selected_sids: Iterable[str],
        *,
        topm: int = 5,
    ) -> list[dict[str, Any]]:
        selected = set(selected_sids)
        frontier = list(dict.fromkeys(frontier_eids))
        scored: list[tuple[float, str, str]] = []
        for source in frontier:
            for target in self.similar_entities(source, limit=8):
                if any(target in (self.sentence_to_entities.get(sid) or []) for sid in selected):
                    continue
                score = lexical_score(question, str(rec.get("surface") or ""))
                scored.append((score, source, target))
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for score, source, target in scored:
            if target in seen or target in frontier:
                continue
            seen.add(target)
            out.append({"source_eid": source, "target_eid": target, "eid": target, "score": score})
            if len(out) >= topm:
                break
        return out

    def sids_for_docs(self, doc_ids: Iterable[str]) -> list[str]:
        out: list[str] = []
        for did in doc_ids:
            out.extend(self.doc_to_sids.get(str(did)) or [])
        return out

    def _rank_sids(self, query: str, sids: Iterable[str]) -> list[tuple[str, float]]:
        scored = [
            (sid, lexical_score(query, str((self.sentences.get(sid) or {}).get("text") or "")))
            for sid in sids
            if sid in self.sentences
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def hybrid_initial_retrieve(
        self,
        query: str,
        *,
        topk: int = 6,
        doc_order: list[str] | None = None,
        use_entity_channel: bool = True,
    ) -> list[dict[str, Any]]:
        allowed = [str(x) for x in (doc_order or []) if str(x)]
        if allowed:
            lexical = self._rank_sids(query, self.sids_for_docs(allowed))
        else:
            lexical = self._rank_all(query)
        entity_hits: list[str] = []
        if use_entity_channel:
            q_eids: set[str] = set()
            for surface in _ENTITY_RE.findall(query or ""):
                surface = normalize_entity_surface(surface)
                if keep_entity_surface(surface):
                    q_eids.add(canonical_eid(surface))
            sid_iter = self.sids_for_docs(allowed) if allowed else list(self.sentences)
            for sid in sid_iter:
                ents = self.sentence_to_entities.get(sid) or []
                if q_eids and any(eid in q_eids for eid in ents):
                    entity_hits.append(sid)
        bm25_sids: list[str] = []
        if allowed:
            for did in allowed:
                sids = list(self.doc_to_sids.get(did) or [])
                sids.sort(key=lambda s: int((self.sentences.get(s) or {}).get("idx") or 0))
                bm25_sids.extend(sids)
        fused = _rrf_fuse([[sid for sid, _ in lexical], entity_hits, bm25_sids])
        return [
            self._row(
                sid,
                score=lexical_score(query, str((self.sentences.get(sid) or {}).get("text") or "")),
                source="hybrid_init",
            )
            for sid in fused[:topk]
            if sid in self.sentences
        ]

    def rank_init_sids(
        self,
        query: str,
        *,
        topk: int = 6,
        doc_order: list[str] | None = None,
        hybrid: bool = False,
    ) -> list[str]:
        if hybrid:
            return [row["sid"] for row in self.hybrid_initial_retrieve(query, topk=topk, doc_order=doc_order)]
        if doc_order:
            ranked = self._rank_sids(query, self.sids_for_docs(doc_order))
        else:
            ranked = self._rank_all(query)
        positive = [(sid, score) for sid, score in ranked if score > 0]
        if not positive:
            return self._coverage_fallback(doc_order, topk)
        picked: list[str] = []
        used_docs: set[str] = set()
        # Prefer relevant sentences, with light per-doc coverage among positive hits.
        by_doc: dict[str, list[tuple[str, float]]] = {}
        for sid, score in positive:
            did = str((self.sentences.get(sid) or {}).get("doc_id") or "")
            by_doc.setdefault(did, []).append((sid, score))
        order = list(doc_order or []) + [d for d in by_doc if d not in (doc_order or [])]
        idx = 0
        while len(picked) < topk:
            progressed = False
            for did in order:
                rows = by_doc.get(did) or []
                if idx < len(rows):
                    sid = rows[idx][0]
                    if sid not in picked:
                        picked.append(sid)
                        used_docs.add(did)
                        progressed = True
                    if len(picked) >= topk:
                        break
            if not progressed:
                break
            idx += 1
        if len(picked) < topk:
            for sid, _score in positive:
                if sid not in picked:
                    picked.append(sid)
                if len(picked) >= topk:
                    break
        return picked[:topk]

    def lookup_entity(
        self,
        eid: str,
        mixquery: str,
        *,
        topk: int = 6,
        use_synonyms: bool = False,
        use_neighbors: bool = False,
        extra_sids: Iterable[str] | None = None,
        observed_sids: Iterable[str] | None = None,
        new_doc_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        if eid not in self.entities and not extra_sids and not new_doc_ids:
            return []
        candidate_sids: list[str] = []
        similar_eids = self.similar_entities(eid) if use_synonyms else []
        neighbor_sids: set[str] = set()
        mention_eids = [eid]
        if use_synonyms:
            mention_eids.extend((self.entities.get(eid) or {}).get("synonyms") or [])
            mention_eids.extend(similar_eids)
        seen: set[str] = set()
        sources: dict[str, str] = {}

        def add(sid: str, source: str) -> None:
            if not sid or sid not in self.sentences or sid in seen:
                return
            seen.add(sid)
            candidate_sids.append(sid)
            sources[sid] = source

        observed_docs = {
            str((self.sentences.get(sid) or {}).get("doc_id") or "")
            for sid in (observed_sids or [])
            if sid
        }
        observed_docs.discard("")
        for mention_eid in mention_eids:
            rec = self.entities.get(mention_eid) or {}
            src = "target" if mention_eid == eid else "synonym"
            sids = rec.get("sids") or []
            hub = len(sids) > GRAPH_MAX_ENTITY_SIDS
            if not hub:
                hub = len(self.docs_for_entity(mention_eid)) > GRAPH_MAX_ENTITY_DOCS
            if hub:
                for did in observed_docs:
                    for sid in self.doc_to_sids.get(did) or []:
                        if mention_eid in (self.sentence_to_entities.get(sid) or []):
                            add(sid, src)
            else:
                for sid in sids:
                    add(sid, src)
        if use_neighbors:
            for sid in list(candidate_sids):
                for nb in (self.sentences.get(sid) or {}).get("neighbors") or []:
                    neighbor_sids.add(nb)
                    add(nb, "sentence_neighbor")
        for sid in extra_sids or []:
            add(str(sid), "extra")
        if new_doc_ids:
            for did in {str(x) for x in new_doc_ids if str(x)}:
                for sid in self.doc_to_sids.get(did) or []:
                    add(sid, "fresh_retrieval")
        scored: list[dict[str, Any]] = []
        observed = set(observed_sids or [])
        for sid in candidate_sids:
            sent = self.sentences[sid]
            score = lexical_score(mixquery, str(sent.get("text") or ""))
            source = sources.get(sid, "graph_candidate")
            if source == "target":
                score += 0.10
            scored.append(
                {
                    **sent,
                    "score": round(float(score), 6),
                    "entity_source": source,
                    "rank": 0,
                    "unobserved": sid not in observed,
                }
            )
        scored.sort(key=lambda row: (-float(row["score"]), 0 if row.get("unobserved") else 1, row["sid"]))
        observed_docs_set = {
            str((self.sentences.get(sid) or {}).get("doc_id") or "")
            for sid in observed
        }
        observed_docs_set.discard("")
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for row in scored:
            by_doc.setdefault(str(row.get("doc_id") or ""), []).append(row)
        for did in by_doc:
            by_doc[did].sort(key=lambda row: (-float(row["score"]), row["sid"]))
        doc_order = sorted(
            by_doc,
            key=lambda d: (
                0 if d not in observed_docs_set else 1,
                -float(by_doc[d][0]["score"]) if by_doc[d] else 0.0,
                d,
            ),
        )
        selected: list[dict[str, Any]] = []
        k = max(1, int(topk))
        idx = 0
        while len(selected) < k:
            progressed = False
            for did in doc_order:
                rows = by_doc[did]
                if idx < len(rows):
                    selected.append(rows[idx])
                    progressed = True
                    if len(selected) >= k:
                        break
            if not progressed:
                break
            idx += 1
        if not selected:
            selected = scored[:k]
        if new_doc_ids:
            new_set = {str(x) for x in new_doc_ids}
            if selected and not any(str(row.get("doc_id")) in new_set for row in selected):
                replacement = next((row for row in scored if str(row.get("doc_id")) in new_set), None)
                if replacement is not None:
                    selected[-1] = replacement
        for i, row in enumerate(selected):
            row["rank"] = i
        return selected

    def _rank_all(self, query: str) -> list[tuple[str, float]]:
        scored = [(sid, lexical_score(query, str(sent.get("text") or ""))) for sid, sent in self.sentences.items()]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def _coverage_fallback(self, doc_order: list[str] | None, topk: int) -> list[str]:
        by_doc: dict[str, list[str]] = {}
        if doc_order:
            order = [str(x) for x in doc_order if str(x)]
            for did in order:
                sids = list(self.doc_to_sids.get(did) or [])
                by_doc[did] = sorted(
                    sids, key=lambda s: int((self.sentences.get(s) or {}).get("idx") or 0)
                )
        else:
            for sid, sent in self.sentences.items():
                by_doc.setdefault(str(sent.get("doc_id")), []).append(sid)
            for did in list(by_doc):
                by_doc[did] = sorted(
                    by_doc[did], key=lambda s: int((self.sentences.get(s) or {}).get("idx") or 0)
                )
            order = list(by_doc)
        picked: list[str] = []
        idx = 0
        docs = [by_doc[d] for d in order if by_doc.get(d)]
        while len(picked) < topk and docs:
            progressed = False
            for doc_sids in docs:
                if idx < len(doc_sids):
                    sid = doc_sids[idx]
                    if sid not in picked:
                        picked.append(sid)
                        progressed = True
                    if len(picked) >= topk:
                        break
            if not progressed:
                break
            idx += 1
        return picked[:topk]

    def _row(self, sid: str, *, score: float, source: str) -> dict[str, Any]:
        sent = dict(self.sentences.get(sid) or {"sid": sid})
        sent["score"] = round(float(score), 6)
        sent["entity_source"] = source
        return sent


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


def build_graph_from_documents(
    doc_store: Mapping[str, Any] | None,
    *,
    scope: str = "episode_doc_store",
) -> HarnessGGraphIndex:
    graph = HarnessGGraphIndex(scope=scope)
    graph.ingest_documents(doc_store)
    return graph


def graph_to_payload(graph: HarnessGGraphIndex) -> dict[str, Any]:
    for rec in graph.entities.values():
        if isinstance(rec, dict):
            rec.pop("_sid_index", None)
    return {
        "format": GRAPH_FORMAT,
        "scope": graph.scope,
        "source_corpus": getattr(graph, "source_corpus", None),
        "docs": graph.docs,
        "sentences": graph.sentences,
        "entities": graph.entities,
        "sentence_to_entities": graph.sentence_to_entities,
        "parent_docids": graph.parent_docids,
        "doc_to_sids": getattr(graph, "doc_to_sids", {}),
    }


def graph_from_payload(payload: Mapping[str, Any]) -> HarnessGGraphIndex:
    fmt = str(payload.get("format") or "")
    if fmt and fmt != GRAPH_FORMAT:
        raise ValueError(f"unsupported Harness-G graph format: {fmt}")
    graph = HarnessGGraphIndex(scope=str(payload.get("scope") or "corpus"))
    graph.source_corpus = payload.get("source_corpus")
    # Take ownership of the payload maps. Copying 10M+ nested lists on load
    # makes official corpus startup unusable.
    graph.docs = payload.get("docs") or {}
    graph.sentences = payload.get("sentences") or {}
    graph.entities = payload.get("entities") or {}
    graph.sentence_to_entities = payload.get("sentence_to_entities") or {}
    graph.parent_docids = payload.get("parent_docids") or {}
    graph.doc_to_sids = payload.get("doc_to_sids") or {}
    if not graph.doc_to_sids and graph.sentences:
        rebuilt: dict[str, list[str]] = {}
        for sid, sent in graph.sentences.items():
            did = str(sent.get("doc_id") or "")
            if did:
                rebuilt.setdefault(did, []).append(sid)
        graph.doc_to_sids = rebuilt
    return graph


def save_graph_index(graph: HarnessGGraphIndex, path: str | Any) -> None:
    from pathlib import Path

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = graph_to_payload(graph)
    if dest.suffix.lower() == ".json":
        dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return
    import pickle

    dest.write_bytes(pickle.dumps(payload, protocol=4))


def load_graph_index(path: str | Any) -> HarnessGGraphIndex:
    from pathlib import Path

    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Harness-G graph index not found: {src}")
    size = src.stat().st_size
    if size >= 100_000_000:
        print(f"[harness_g_graph] loading {src} ({size / 1e9:.1f} GB)", flush=True)
    if src.suffix.lower() == ".json":
        payload = json.loads(src.read_text(encoding="utf-8"))
    else:
        import pickle

        payload = pickle.loads(src.read_bytes())
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid Harness-G graph payload in {src}")
    graph = graph_from_payload(payload)
    graph.source_path = str(src)
    if size >= 100_000_000:
        print(
            f"[harness_g_graph] loaded docs={len(graph.docs)} sents={len(graph.sentences)} ents={len(graph.entities)}",
            flush=True,
        )
    return graph


def resolve_graph_index(path: str | Any | None) -> HarnessGGraphIndex | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return load_graph_index(text)
