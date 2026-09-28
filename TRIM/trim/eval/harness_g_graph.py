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
import threading
import time
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


def lexical_score_tokens(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokenize(text)) / max(1, len(query_tokens))


class GraphFullScanRefused(RuntimeError):
    """Corpus retrieval refused to walk every sentence."""


_scan_tls = threading.local()


def begin_scan_bucket() -> dict[str, dict[str, int]]:
    bucket: dict[str, dict[str, int]] = {}
    _scan_tls.bucket = bucket
    return bucket


def drain_scan_bucket() -> dict[str, dict[str, int]]:
    bucket = getattr(_scan_tls, "bucket", None) or {}
    _scan_tls.bucket = {}
    return {key: dict(value) for key, value in bucket.items()}


def add_scan(kind: str, **fields: int) -> None:
    bucket = getattr(_scan_tls, "bucket", None)
    if bucket is None:
        return
    rec = bucket.setdefault(str(kind), {})
    for key, value in fields.items():
        rec[key] = int(rec.get(key) or 0) + int(value or 0)


_lookup_tls = threading.local()


def publish_lookup_stats(stats: Mapping[str, Any]) -> None:
    _lookup_tls.stats = dict(stats)


def take_lookup_stats() -> dict[str, Any]:
    stats = getattr(_lookup_tls, "stats", None) or {}
    _lookup_tls.stats = {}
    return dict(stats)


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
GRAPH_FINGERPRINT_VERSION = "harness_g_graph_fp_v2"
GRAPH_MAX_ENTITY_DOCS = 200
GRAPH_MAX_ENTITY_SIDS = GRAPH_MAX_ENTITY_DOCS * 400
HYBRID_RETRIEVAL_BACKEND = "lexical_postings"
SIMILAR_NAME_ENTITY_SCAN_LIMIT = 50_000
SENTENCE_INDEX_SCAN_LIMIT = 500_000
MAX_POSTINGS_PER_TOKEN = 64


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
        self._token_to_sids: dict[str, list[str]] | None = None
        self._token_to_eids: dict[str, list[str]] | None = None
        self._retrieval_index_ready: bool = False
        self._retrieval_backend: str = HYBRID_RETRIEVAL_BACKEND

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
        other._token_to_sids = self._token_to_sids
        other._token_to_eids = self._token_to_eids
        other._retrieval_index_ready = self._retrieval_index_ready
        other._retrieval_backend = getattr(self, "_retrieval_backend", HYBRID_RETRIEVAL_BACKEND)
        for attr in ("_doc_eid_counts", "_bridge_static", "_surface_tok_cache", "_sid_tok_cache", "_graph_cache_lock_obj"):
            if attr in self.__dict__:
                setattr(other, attr, self.__dict__[attr])
        return other

    def content_fingerprint(self) -> str:
        if self._fingerprint:
            return self._fingerprint
        doc_xor = 0
        for did, rec in self.docs.items():
            piece = hashlib.sha1(f"{did}|{rec.get('text_hash') or ''}".encode("utf-8")).digest()
            doc_xor ^= int.from_bytes(piece[:8], "little")
        synonym_xor = 0
        similar_xor = 0
        related_xor = 0
        parent_xor = 0
        for eid, rec in self.entities.items():
            for syn in rec.get("synonyms") or []:
                piece = hashlib.sha1(f"{eid}|syn|{syn}".encode("utf-8")).digest()
                synonym_xor ^= int.from_bytes(piece[:8], "little")
            for other in rec.get("similar_names") or []:
                piece = hashlib.sha1(f"{eid}|sim|{other}".encode("utf-8")).digest()
                similar_xor ^= int.from_bytes(piece[:8], "little")
            for other in rec.get("related") or []:
                piece = hashlib.sha1(f"{eid}|rel|{other}".encode("utf-8")).digest()
                related_xor ^= int.from_bytes(piece[:8], "little")
        for sid, rec in self.sentences.items():
            parent = str(rec.get("parent_docid") or rec.get("doc_id") or "")
            piece = hashlib.sha1(f"{sid}|p|{parent}".encode("utf-8")).digest()
            parent_xor ^= int.from_bytes(piece[:8], "little")
            for nb in rec.get("neighbors") or []:
                piece = hashlib.sha1(f"{sid}|nb|{nb}".encode("utf-8")).digest()
                parent_xor ^= int.from_bytes(piece[:8], "little")
        alias_digest = hashlib.sha1(json.dumps(_ALIAS_GROUPS, sort_keys=False).encode("utf-8")).hexdigest()[:16]
        payload = {
            "scope": self.scope,
            "n_docs": len(self.docs),
            "n_sents": len(self.sentences),
            "n_ents": len(self.entities),
            "doc_xor": doc_xor,
            "synonym_xor": synonym_xor,
            "similar_xor": similar_xor,
            "related_xor": related_xor,
            "parent_neighbor_xor": parent_xor,
            "build_version": GRAPH_FINGERPRINT_VERSION,
            "max_unit_chars": MAX_UNIT_CHARS,
            "alias_groups": alias_digest,
            "retrieval_backend": getattr(self, "_retrieval_backend", HYBRID_RETRIEVAL_BACKEND),
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._fingerprint = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return self._fingerprint

    def metadata(self) -> dict[str, Any]:
        fingerprint = self.content_fingerprint()
        cached = self.__dict__.get("_metadata_cache")
        if isinstance(cached, dict) and cached.get("graph_fingerprint") == fingerprint:
            return dict(cached)
        n_sent_edges = 0
        for sids in (self.doc_to_sids or {}).values():
            n = len(sids)
            if n > 1:
                n_sent_edges += 2 * (n - 1)
        n_syn_edges = sum(len(rec.get("synonyms") or []) for rec in self.entities.values())
        payload = {
            "graph_scope": self.scope,
            "graph_fingerprint": fingerprint,
            "graph_num_docs": len(self.docs),
            "graph_num_sentences": len(self.sentences),
            "graph_num_entities": len(self.entities),
            "graph_num_edges": n_sent_edges + n_syn_edges,
            "graph_build_version": GRAPH_FINGERPRINT_VERSION,
            "graph_retrieval_backend": getattr(self, "_retrieval_backend", HYBRID_RETRIEVAL_BACKEND),
            "graph_format": GRAPH_FORMAT,
            "graph_source_corpus": getattr(self, "source_corpus", None),
            "graph_source_path": getattr(self, "source_path", None),
        }
        self._metadata_cache = dict(payload)
        return dict(payload)

    def full_scan_allowed(self) -> bool:
        if getattr(self, "_refuse_full_scan", False):
            return False
        return len(self.sentences) <= SENTENCE_INDEX_SCAN_LIMIT

    def _graph_cache_lock(self):
        lock = self.__dict__.get("_graph_cache_lock_obj")
        if lock is None:
            lock = threading.Lock()
            self.__dict__["_graph_cache_lock_obj"] = lock
        return lock

    def entity_mention_counts(self, doc_id: str) -> dict[str, int]:
        """Per-document entity mention counts. Cached by doc id, returned as a copy."""
        doc_id = str(doc_id)
        cache = self.__dict__.setdefault("_doc_eid_counts", {})
        lock = self._graph_cache_lock()
        with lock:
            hit = cache.get(doc_id)
            if hit is not None:
                add_scan("frontier", cache_hit=1)
                return dict(hit)
        counts: dict[str, int] = {}
        n_sent = 0
        n_mentions = 0
        for sid in self.doc_to_sids.get(doc_id) or []:
            n_sent += 1
            for eid in self.sentence_to_entities.get(sid) or []:
                key = str(eid)
                counts[key] = counts.get(key, 0) + 1
                n_mentions += 1
        with lock:
            if doc_id not in cache and len(cache) < 4096:
                cache[doc_id] = dict(counts)
            add_scan(
                "frontier",
                cache_miss=1,
                sentences_scanned=n_sent,
                entity_mentions_scanned=n_mentions,
            )
        return dict(counts)

    def _sid_tokens(self, sid: str) -> set[str]:
        cache = self.__dict__.setdefault("_sid_tok_cache", {})
        hit = cache.get(sid)
        if hit is not None:
            add_scan("lookup_rank", token_cache_hit=1)
            return hit
        text = str((self.sentences.get(sid) or {}).get("text") or "")
        toks = _tokenize(text)
        if len(cache) < 20000:
            cache[sid] = toks
        add_scan("lookup_rank", token_cache_miss=1)
        return toks

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
            self.__dict__.pop("_metadata_cache", None)
            self.__dict__.pop("_doc_eid_counts", None)
            self.__dict__.pop("_bridge_static", None)
            self.__dict__.pop("_bridge_result_cache", None)
            self.__dict__.pop("_surface_tok_cache", None)
            self.__dict__.pop("_sid_tok_cache", None)
            self._retrieval_index_ready = False
            self._token_to_sids = None
            self._token_to_eids = None
            self._link_aliases()
            self._link_similar_names()
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
                    rec.setdefault("edge_types", {})
                    for other in rec["synonyms"]:
                        rec["edge_types"][other] = "alias"

    def _link_similar_names(self, *, limit_per_entity: int = 8) -> None:
        """Offline name-similarity edges. Query path never scans the whole entity table."""
        if len(self.entities) > SIMILAR_NAME_ENTITY_SCAN_LIMIT:
            return
        token_to_eids: dict[str, list[str]] = {}
        surfaces: dict[str, set[str]] = {}
        for eid, rec in self.entities.items():
            toks = _tokenize(str(rec.get("surface") or ""))
            surfaces[eid] = toks
            for tok in toks:
                token_to_eids.setdefault(tok, []).append(eid)
        for eid, toks in surfaces.items():
            rec = self.entities.get(eid)
            if rec is None or not toks or rec.get("similar_names"):
                continue
            scores: dict[str, int] = {}
            for tok in toks:
                for other in token_to_eids.get(tok) or []:
                    if other != eid:
                        scores[other] = scores.get(other, 0) + 1
            ranked = sorted(scores, key=lambda o: (-(scores[o] / max(1, len(toks))), o))
            kept: list[str] = []
            rec.setdefault("edge_types", {})
            for other in ranked:
                if len(kept) >= limit_per_entity:
                    break
                if scores[other] / max(1, len(toks)) < 0.5:
                    continue
                kept.append(other)
                rec["edge_types"].setdefault(other, "similar_name")
            rec["similar_names"] = kept

    def graph_quality_report(self) -> dict[str, Any]:
        n_syn = 0
        n_sim = 0
        n_related = 0
        n_no_sids = 0
        degrees: list[int] = []
        for rec in self.entities.values():
            n_syn += len(rec.get("synonyms") or [])
            n_sim += len(rec.get("similar_names") or [])
            n_related += len(rec.get("related") or [])
            n_sids = len(rec.get("sids") or [])
            degrees.append(n_sids)
            if n_sids == 0:
                n_no_sids += 1
        n_nav = 0
        n_body = 0
        for sent in self.sentences.values():
            text = str(sent.get("text") or "")
            if is_metadata_only_line(text):
                n_nav += 1
            else:
                n_body += 1
        parent_mismatch = 0
        for sid, sent in self.sentences.items():
            parent = str(sent.get("parent_docid") or sent.get("doc_id") or "")
            if ":" in sid and sid.split(":", 1)[0] != parent.split(":", 1)[0] and parent not in self.docs:
                parent_mismatch += 1
        degrees_sorted = sorted(degrees)
        def _pct(p: float) -> int:
            if not degrees_sorted:
                return 0
            idx = min(len(degrees_sorted) - 1, max(0, int(p / 100.0 * (len(degrees_sorted) - 1))))
            return degrees_sorted[idx]
        return {
            "n_docs": len(self.docs),
            "n_sentences": len(self.sentences),
            "n_entities": len(self.entities),
            "n_alias_edges": n_syn,
            "n_similar_name_edges": n_sim,
            "n_related_edges": n_related,
            "n_entities_without_sids": n_no_sids,
            "entity_degree_p50": _pct(50),
            "entity_degree_p95": _pct(95),
            "n_nav_or_metadata_sentences": n_nav,
            "n_body_sentences": n_body,
            "n_parent_mismatch_sids": parent_mismatch,
            "graph_fingerprint": self.content_fingerprint(),
            "retrieval_backend": getattr(self, "_retrieval_backend", HYBRID_RETRIEVAL_BACKEND),
        }

    def ensure_retrieval_index(self, *, max_postings: int = MAX_POSTINGS_PER_TOKEN) -> str:
        if self._retrieval_index_ready and self._token_to_eids is not None:
            return str(self._retrieval_backend)
        token_to_eids: dict[str, list[str]] = {}
        for eid, rec in self.entities.items():
            for tok in _tokenize(str(rec.get("surface") or "")):
                lst = token_to_eids.setdefault(tok, [])
                if len(lst) < max_postings:
                    lst.append(eid)
        self._token_to_eids = token_to_eids
        if len(self.sentences) <= SENTENCE_INDEX_SCAN_LIMIT:
            token_to_sids: dict[str, list[str]] = {}
            for sid, sent in self.sentences.items():
                for tok in _tokenize(str(sent.get("text") or "")):
                    lst = token_to_sids.setdefault(tok, [])
                    if len(lst) < max_postings:
                        lst.append(sid)
            self._token_to_sids = token_to_sids
            self._retrieval_backend = "lexical_postings"
        else:
            self._token_to_sids = None
            self._retrieval_backend = "lexical_entity_postings"
        self._retrieval_index_ready = True
        return self._retrieval_backend

    def _posting_sids_for_query(self, query: str, *, limit: int) -> list[str]:
        self.ensure_retrieval_index()
        scores: dict[str, float] = {}
        toks = _tokenize(query)
        if self._token_to_sids:
            for tok in toks:
                for sid in self._token_to_sids.get(tok) or []:
                    scores[sid] = scores.get(sid, 0.0) + 1.0
        else:
            eids = self._posting_eids_for_query(query, limit=max(16, limit))
            for eid in eids:
                for sid in (self.entities.get(eid) or {}).get("sids") or []:
                    scores[sid] = scores.get(sid, 0.0) + 1.0
                    if len(scores) >= max(256, limit * 8):
                        break
        return sorted(scores, key=lambda s: (-scores[s], s))[: max(1, int(limit))]

    def _posting_eids_for_query(self, query: str, *, limit: int) -> list[str]:
        self.ensure_retrieval_index()
        scores: dict[str, float] = {}
        for tok in _tokenize(query):
            for eid in (self._token_to_eids or {}).get(tok) or []:
                scores[eid] = scores.get(eid, 0.0) + 1.0
        for surface in _ENTITY_RE.findall(query or ""):
            surface = normalize_entity_surface(surface)
            if keep_entity_surface(surface):
                eid = canonical_eid(surface)
                if eid in self.entities:
                    scores[eid] = scores.get(eid, 0.0) + 2.0
        return sorted(scores, key=lambda e: (-scores[e], e))[: max(1, int(limit))]

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
        out: list[str] = []
        seen: set[str] = {eid}

        def _add(other: str) -> None:
            other = str(other or "")
            if other and other not in seen and other in self.entities:
                seen.add(other)
                out.append(other)

        for syn in rec.get("synonyms") or []:
            _add(str(syn))
        for other in rec.get("similar_names") or []:
            _add(str(other))
        for other in rec.get("related") or []:
            _add(str(other))
        if not rec.get("similar_names") and len(self.entities) <= SIMILAR_NAME_ENTITY_SCAN_LIMIT:
            surface_toks = _tokenize(str(rec.get("surface") or ""))
            if surface_toks:
                scored: list[tuple[float, str]] = []
                for other_id, other in self.entities.items():
                    if other_id in seen:
                        continue
                    toks = _tokenize(str(other.get("surface") or ""))
                    if not toks:
                        continue
                    inter = len(surface_toks & toks)
                    if inter <= 0:
                        continue
                    scored.append((inter / max(1, len(surface_toks)), other_id))
                scored.sort(key=lambda x: (-x[0], x[1]))
                rec["similar_names"] = [oid for _, oid in scored[: max(0, int(limit))]]
                rec.setdefault("edge_types", {})
                for oid in rec["similar_names"]:
                    rec["edge_types"].setdefault(oid, "similar_name")
                for oid in rec["similar_names"]:
                    _add(oid)
        return out[: max(0, int(limit))]

    def _bridge_graph_local(self, source_eid: str) -> dict[str, frozenset[str]]:
        """Query-independent co-sentence, neighbor, and synonym candidates for one source."""
        cache = self.__dict__.setdefault("_bridge_static", {})
        lock = self._graph_cache_lock()
        with lock:
            hit = cache.get(source_eid)
        if hit is not None:
            add_scan("bridge", static_cache_hit=1)
            return hit
        add_scan("bridge", static_cache_miss=1)
        reasons: dict[str, set[str]] = {}
        source_rec = self.entities.get(source_eid) or {}
        n_sids = 0
        for sid in list(source_rec.get("sids") or [])[:32]:
            n_sids += 1
            for target_eid in self.sentence_to_entities.get(sid) or []:
                target_eid = str(target_eid)
                if target_eid == source_eid or target_eid not in self.entities:
                    continue
                reasons.setdefault(target_eid, set()).add("co_sentence")
            for nb in (self.sentences.get(sid) or {}).get("neighbors") or []:
                for target_eid in self.sentence_to_entities.get(nb) or []:
                    target_eid = str(target_eid)
                    if target_eid == source_eid or target_eid not in self.entities:
                        continue
                    reasons.setdefault(target_eid, set()).add("sentence_neighbor")
        for similar in self.similar_entities(source_eid, limit=8):
            similar = str(similar)
            if similar == source_eid or similar not in self.entities:
                continue
            reason = "synonym_entity"
            edge_type = str((source_rec.get("edge_types") or {}).get(similar) or "")
            if edge_type == "similar_name":
                reason = "similar_name"
            reasons.setdefault(similar, set()).add(reason)
        frozen = {eid: frozenset(rs) for eid, rs in reasons.items()}
        add_scan("bridge", source_sids_scanned=n_sids, static_targets=len(frozen))
        with lock:
            if source_eid not in cache and len(cache) < 8192:
                cache[source_eid] = frozen
        return frozen

    def propose_bridge_entities(
        self,
        frontier_eids: Iterable[str],
        question: str,
        selected_sids: Iterable[str],
        *,
        topm: int = 5,
    ) -> list[dict[str, Any]]:
        source_eid_list = [eid for eid in dict.fromkeys(frontier_eids) if eid in self.entities]
        if not source_eid_list:
            return []
        selected_sid_set = {sid for sid in selected_sids if sid in self.sentences}
        result_key = (
            self.content_fingerprint(),
            tuple(source_eid_list),
            str(question or ""),
            tuple(sorted(selected_sid_set)),
            int(topm),
        )
        result_cache = self.__dict__.setdefault("_bridge_result_cache", {})
        cached_rows = result_cache.get(result_key)
        if cached_rows is not None:
            add_scan("bridge", result_cache_hit=1)
            return [dict(row) for row in cached_rows]
        add_scan("bridge", result_cache_miss=1)
        source_set = set(source_eid_list)
        source_priority = {
            "selected_sentence": 0,
            "co_sentence": 1,
            "sentence_neighbor": 2,
            "synonym_entity": 3,
            "similar_name": 4,
        }
        q_tokens = _tokenize(question)
        scored: list[tuple[float, int, str, str, list[str]]] = []
        for source_eid in source_eid_list:
            candidate_reasons: dict[str, set[str]] = {}
            candidate_priority: dict[str, int] = {}

            def add_candidate(target_eid: str, reason: str) -> None:
                target_eid = str(target_eid or "")
                if target_eid not in self.entities or target_eid == source_eid:
                    return
                if self.entities.get(target_eid) is None:
                    return
                if target_eid in source_set and reason not in {"selected_sentence", "co_sentence"}:
                    return
                candidate_reasons.setdefault(target_eid, set()).add(reason)
                candidate_priority[target_eid] = min(
                    candidate_priority.get(target_eid, 99),
                    source_priority.get(reason, 99),
                )

            for selected_sid in selected_sid_set:
                ents = self.sentence_to_entities.get(selected_sid) or []
                if source_eid not in ents:
                    continue
                for target_eid in ents:
                    add_candidate(str(target_eid), "selected_sentence")

            for target_eid, reasons in self._bridge_graph_local(source_eid).items():
                for reason in reasons:
                    add_candidate(target_eid, reason)

            for target_eid, reasons in candidate_reasons.items():
                target_rec = self.entities.get(target_eid)
                if target_rec is None:
                    continue
                pri = candidate_priority.get(target_eid, 99)
                lex = lexical_score_tokens(q_tokens, str(target_rec.get("surface") or ""))
                score = 1.0 / (1.0 + pri) + 0.5 * lex
                scored.append((score, pri, source_eid, target_eid, sorted(reasons)))
        scored.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for score, _pri, source, target, reasons in scored:
            if target in seen:
                continue
            seen.add(target)
            out.append(
                {
                    "source_eid": source,
                    "target_eid": target,
                    "eid": target,
                    "score": round(float(score), 6),
                    "reasons": list(reasons),
                    "source": "bridge_candidate:" + ",".join(reasons),
                }
            )
            if len(out) >= topm:
                break
        if len(result_cache) < 128:
            result_cache[result_key] = [dict(row) for row in out]
        return [dict(row) for row in out]

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
        paragraph_topk: int = 20,
        sentence_topk: int = 12,
        entity_topk: int = 8,
    ) -> list[dict[str, Any]]:
        """Three independent channels: local paragraph, global sentence, global entity mention.

        ``doc_order`` is a local/BM25 whitelist for the paragraph channel only.
        Global channels never use it as a hard filter. Retrieval backend is lexical
        postings, not dense retrieval.
        """
        local_docs = [str(x) for x in (doc_order or []) if str(x)]
        local_sids = self.sids_for_docs(local_docs[: max(1, int(paragraph_topk))]) if local_docs else []
        paragraph_ranked = self._rank_sids(query, local_sids) if local_sids else []
        paragraph_sids = [sid for sid, _ in paragraph_ranked[: max(1, int(sentence_topk))]]

        global_pool = self._posting_sids_for_query(query, limit=max(32, int(sentence_topk) * 8))
        if not global_pool and self.full_scan_allowed():
            global_pool = [sid for sid, _ in self._rank_all(query)[: max(32, int(sentence_topk) * 4)]]
        elif not global_pool:
            self.last_retrieval_block = {
                "reason": "empty_posting_full_scan_refused",
                "n_sentences": len(self.sentences),
                "channel": "global_sentence",
            }
            add_scan("init_retrieve", full_scan_refused=1)
        global_ranked = self._rank_sids(query, global_pool)
        global_sids = [sid for sid, _ in global_ranked[: max(1, int(sentence_topk))]]

        entity_sids: list[str] = []
        if use_entity_channel:
            q_eids = list(self._posting_eids_for_query(query, limit=max(1, int(entity_topk))))
            for surface in _ENTITY_RE.findall(query or ""):
                surface = normalize_entity_surface(surface)
                if keep_entity_surface(surface):
                    eid = canonical_eid(surface)
                    if eid in self.entities and eid not in q_eids:
                        q_eids.append(eid)
            seen_ent: set[str] = set()
            for eid in q_eids[: max(1, int(entity_topk))]:
                for sid in (self.entities.get(eid) or {}).get("sids") or []:
                    if sid in seen_ent or sid not in self.sentences:
                        continue
                    seen_ent.add(sid)
                    entity_sids.append(sid)
                    if len(entity_sids) >= max(8, int(sentence_topk) * 4):
                        break
            entity_sids = [sid for sid, _ in self._rank_sids(query, entity_sids)[: max(1, int(sentence_topk))]]

        fused = _rrf_fuse([paragraph_sids, global_sids, entity_sids])
        source_of: dict[str, str] = {}
        for sid in paragraph_sids:
            source_of[sid] = "paragraph_sentence"
        for sid in global_sids:
            source_of[sid] = "global_sentence" if sid not in source_of else source_of[sid] + ",global_sentence"
        for sid in entity_sids:
            source_of[sid] = "entity_mention" if sid not in source_of else source_of[sid] + ",entity_mention"
        self.last_hybrid_channels = {
            "backend": getattr(self, "_retrieval_backend", HYBRID_RETRIEVAL_BACKEND),
            "paragraph_sids": list(paragraph_sids),
            "global_sentence_sids": list(global_sids),
            "entity_mention_sids": list(entity_sids),
            "local_doc_whitelist": list(local_docs),
            "global_outside_whitelist": [
                sid for sid in global_sids + entity_sids
                if local_docs and str((self.sentences.get(sid) or {}).get("doc_id") or "") not in set(local_docs)
            ],
        }
        return [
            self._row(
                sid,
                score=lexical_score(query, str((self.sentences.get(sid) or {}).get("text") or "")),
                source=source_of.get(sid, "hybrid_init"),
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
        elif not self.full_scan_allowed():
            self.last_retrieval_block = {
                "reason": "empty_doc_order_full_scan_refused",
                "n_sentences": len(self.sentences),
            }
            raise GraphFullScanRefused(
                "corpus graph refused a full-sentence scan because doc_order is empty "
                f"(n_sentences={len(self.sentences)})"
            )
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
        window_offset: int = 0,
        candidates_out: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if eid not in self.entities and not extra_sids and not new_doc_ids:
            return []
        t_collect = time.perf_counter()
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
        t_rank = time.perf_counter()
        add_scan(
            "lookup_candidates",
            n_candidates=len(candidate_sids),
            synonym_mentions=max(0, len(mention_eids) - 1),
        )
        scored: list[dict[str, Any]] = []
        observed = set(observed_sids or [])
        q_tokens = _tokenize(mixquery)
        for sid in candidate_sids:
            sent = self.sentences[sid]
            if not q_tokens:
                score = 0.0
            else:
                score = len(q_tokens & self._sid_tokens(sid)) / max(1, len(q_tokens))
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
        while True:
            progressed = False
            for did in doc_order:
                rows = by_doc[did]
                if idx < len(rows):
                    selected.append(rows[idx])
                    progressed = True
            if not progressed:
                break
            idx += 1
        if not selected:
            selected = list(scored)
        if candidates_out is not None:
            candidates_out.extend(dict(row) for row in selected)
        offset = max(0, int(window_offset))
        window = selected[offset : offset + k]
        if not window:
            window = selected[:k]
        if new_doc_ids:
            new_set = {str(x) for x in new_doc_ids}
            if window and not any(str(row.get("doc_id")) in new_set for row in window):
                replacement = next((row for row in selected if str(row.get("doc_id")) in new_set), None)
                if replacement is not None:
                    window = list(window)
                    window[-1] = replacement
        for i, row in enumerate(window):
            row["rank"] = i
            row["window_offset"] = offset
            row["n_candidates"] = len(selected)
        publish_lookup_stats(
            {
                "lookup_candidates_sec": max(0.0, t_rank - t_collect),
                "lookup_rank_sec": max(0.0, time.perf_counter() - t_rank),
                "n_candidates": len(candidate_sids),
                "n_selected": len(selected),
            }
        )
        return window

    def _rank_all(self, query: str) -> list[tuple[str, float]]:
        if not self.full_scan_allowed():
            raise GraphFullScanRefused(
                f"refusing full-sentence scan (n_sentences={len(self.sentences)})"
            )
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
        elif not self.full_scan_allowed():
            raise GraphFullScanRefused(
                "corpus graph refused coverage fallback over every sentence "
                f"(n_sentences={len(self.sentences)})"
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
    else:
        import pickle

        dest.write_bytes(pickle.dumps(payload, protocol=4))
    report_path = dest.with_name(dest.name + ".quality.json")
    try:
        report_path.write_text(json.dumps(graph.graph_quality_report(), indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[harness_g_graph] quality report skipped: {exc}", flush=True)


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

        with src.open("rb") as handle:
            payload = pickle.load(handle)
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


def graph_sidecar_path(path: str | Any):
    from pathlib import Path

    src = Path(path)
    return src.with_name(src.name + ".meta.json")


def graph_file_identity(path: str | Any) -> dict[str, Any]:
    from pathlib import Path

    src = Path(path)
    stat = src.stat()
    digest = hashlib.sha256()
    with src.open("rb") as handle:
        digest.update(handle.read(1 << 20))
        if stat.st_size > (1 << 20):
            handle.seek(max(0, stat.st_size - (1 << 20)))
            digest.update(handle.read(1 << 20))
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "edge_sha256": digest.hexdigest(),
    }


def read_graph_metadata_sidecar(path: str | Any) -> dict[str, Any] | None:
    """Parent-process metadata. None when the sidecar is missing or does not match the file."""
    side = graph_sidecar_path(path)
    if not side.is_file():
        return None
    try:
        payload = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not str(payload.get("graph_fingerprint") or ""):
        return None
    try:
        identity = graph_file_identity(path)
    except OSError:
        return None
    if payload.get("file_identity") != identity:
        return None
    out = dict(payload)
    out.pop("file_identity", None)
    out["graph_metadata_source"] = "sidecar"
    return out


def write_graph_metadata_sidecar(path: str | Any, meta: Mapping[str, Any]) -> None:
    side = graph_sidecar_path(path)
    payload = dict(meta)
    payload["file_identity"] = graph_file_identity(path)
    payload["sidecar_version"] = 1
    side.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_graph_index(path: str | Any | None) -> HarnessGGraphIndex | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return load_graph_index(text)
