"""Original-schema search / grep / read tools over a local corpus."""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Mapping

from trim.local_backend.bm25 import LocalBm25Backend, RankHit
from trim.local_backend.corpus_store import LocalCorpusStore
from trim.local_backend.format_obs import format_search_observation
from trim.upstream_harness1.token_count import prefix_to_token_budget

GREP_LIMIT = 5
GREP_TIMEOUT_S = 10.0
SEARCH_DESCRIPTION_LOCAL = (
    "Searches a local BM25 (Lucene) index for relevant documents. "
    "This is not the original Chroma hybrid sparse+dense retriever."
)


class _LocalToolBase:
    def __init__(self, tool_schema: Any):
        self.tool_schema = tool_schema

    def get_format(self, provider: Any) -> dict[str, Any]:
        return self.tool_schema.to_provider_format(provider)

    def __repr__(self) -> str:
        return f"LocalTool(name={self.tool_schema.name!r})"


def _copy_schema(schema: Any, description: str | None = None) -> Any:
    payload = schema.model_dump() if hasattr(schema, "model_dump") else schema.dict()
    if description:
        payload["description"] = description
    return type(schema)(**payload)


class LocalSearchCorpusTool(_LocalToolBase):
    def __init__(
        self,
        *,
        backend: LocalBm25Backend,
        store: LocalCorpusStore,
        metadata_cls: Any,
        schema: Any,
        reranker: Any | None = None,
        display_limit: int = 10,
        search_limit: int = 50,
        token_counter: Callable[[str], int] | None = None,
        cache: dict[tuple[Any, ...], tuple[str, Any]] | None = None,
        capability_log: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(_copy_schema(schema, SEARCH_DESCRIPTION_LOCAL))
        self.backend = backend
        self.store = store
        self._metadata_cls = metadata_cls
        self._reranker = reranker
        self._display_limit = int(display_limit)
        self._search_limit = int(search_limit)
        self._token_counter = token_counter
        self._cache = cache if cache is not None else {}
        self.capability_log = capability_log if capability_log is not None else {}

    def __call__(self, params: Mapping[str, Any], overrides: Mapping[str, Any] | None = None):
        if not isinstance(params, dict) or "query" not in params:
            raise ValueError(f"Invalid params type: {type(params)}")
        query = str(params["query"])
        ignore_ids = list((overrides or {}).get("ignore_ids") or [])
        rerank_instruction = (overrides or {}).get("rerank_instruction")
        # Locked upstream SearchCorpusTool.__call__ does not pass instruction to the reranker.
        self.capability_log["adaptive_rerank_instruction_received"] = bool(rerank_instruction)
        self.capability_log["adaptive_rerank_instruction_consumed_by_search"] = False
        key = (
            self.backend.index_version,
            query,
            tuple(sorted(str(x) for x in ignore_ids)),
            self._search_limit,
            getattr(self._reranker, "identity", None),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        hits = self.backend.search(query, k=self._search_limit, ignore_ids=ignore_ids)
        ids = [h.chunk_id for h in hits]
        docs = [h.text for h in hits]
        pre_ids = list(ids)
        token_counts: list[int | None] = [None] * len(ids)
        if self._reranker is not None and docs:
            max_tokens = (overrides or {}).get("max_tokens")
            ranked = self._reranker(query, docs, max_tokens=max_tokens)
            ids = [ids[r.original_index] for r in ranked]
            docs = [r.document for r in ranked]
            token_counts = [getattr(r, "tokens", None) for r in ranked]
            self.capability_log["reranker_called"] = True
        shown = ids[: self._display_limit] if ids else []
        doc_texts = {cid: doc for cid, doc in zip(shown, docs[: len(shown)])}
        text = format_search_observation(ids, docs, token_counts, display_limit=self._display_limit)
        meta = self._metadata_cls(
            returned_chunk_ids=list(shown),
            pre_rerank_chunk_ids=pre_ids if self._reranker else None,
            doc_texts=doc_texts or None,
        )
        result = (text, meta)
        self._cache[key] = result
        return result


class LocalGrepCorpusTool(_LocalToolBase):
    def __init__(
        self,
        *,
        store: LocalCorpusStore,
        metadata_cls: Any,
        schema: Any,
        token_counter: Callable[[str], int] | None = None,
        limit: int = GREP_LIMIT,
        timeout_s: float = GREP_TIMEOUT_S,
    ) -> None:
        super().__init__(schema)
        self.store = store
        self._metadata_cls = metadata_cls
        self._token_counter = token_counter
        self._limit = int(limit)
        self._timeout_s = float(timeout_s)
        self.capability_log: dict[str, Any] = {}

    def __call__(self, params: Mapping[str, Any], overrides: Mapping[str, Any] | None = None):
        del overrides
        if not isinstance(params, dict) or "pattern" not in params:
            raise ValueError(f"Invalid params type: {type(params)}")
        pattern = str(params["pattern"])
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            if isinstance(self.capability_log, dict):
                self.capability_log["grep_invalid_regex"] = int(
                    self.capability_log.get("grep_invalid_regex") or 0
                ) + 1
            return (f"grep: invalid regex ({exc})", self._metadata_cls(returned_chunk_ids=[]))
        started = time.perf_counter()
        ids: list[str] = []
        docs: list[str] = []
        timed_out = False
        for chunk in self.store.iter_chunks():
            if time.perf_counter() - started >= self._timeout_s:
                timed_out = True
                break
            if compiled.search(chunk.text or ""):
                ids.append(chunk.chunk_id)
                docs.append(chunk.text)
                if len(ids) >= self._limit:
                    break
        if timed_out:
            if isinstance(self.capability_log, dict):
                self.capability_log["grep_timeout"] = int(self.capability_log.get("grep_timeout") or 0) + 1
        if timed_out and not ids:
            return (
                f"grep: scan timed out after {self._timeout_s:.1f}s; not a confirmed empty corpus",
                self._metadata_cls(returned_chunk_ids=[]),
            )
        token_counts = [self._token_counter(d) for d in docs] if self._token_counter else [None] * len(docs)
        doc_texts = {cid: doc for cid, doc in zip(ids, docs)}
        text = format_search_observation(ids, docs, token_counts, display_limit=self._limit)
        if isinstance(self.capability_log, dict):
            if ids:
                self.capability_log["grep_success"] = int(self.capability_log.get("grep_success") or 0) + 1
            else:
                self.capability_log["grep_no_results"] = int(self.capability_log.get("grep_no_results") or 0) + 1
        return (text, self._metadata_cls(returned_chunk_ids=list(ids), doc_texts=doc_texts or None))


class LocalReadDocumentTool(_LocalToolBase):
    def __init__(
        self,
        *,
        store: LocalCorpusStore,
        schema: Any,
        reranker: Any | None = None,
        token_counter: Callable[[str], int] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(schema)
        self.store = store
        self._reranker = reranker
        self._token_counter = token_counter
        self._max_tokens = max_tokens
        self.capability_log: dict[str, Any] = {}

    def __call__(self, params: Mapping[str, Any], overrides: Mapping[str, Any] | None = None):
        if not isinstance(params, dict) or ("doc_id" not in params and "id" not in params):
            raise ValueError(f"Invalid params type: {type(params)}")
        raw_id = str(params["doc_id"] if "doc_id" in params else params["id"])
        checksum_before = self.store.checksum_of(raw_id)
        chunks = self.store.chunks_for(raw_id)
        if not chunks:
            doc = self.store.get_document(raw_id)
            if doc is None:
                log = getattr(self, "capability_log", None)
                if isinstance(log, dict):
                    log["read_unknown_id"] = int(log.get("read_unknown_id") or 0) + 1
                return (f"read_document: unknown id {raw_id}", None)
            from trim.local_backend.corpus_store import Chunk

            chunks = [
                Chunk(
                    chunk_id=self.store.id_map.chunk_id(doc.doc_id, 0),
                    doc_id=doc.doc_id,
                    ordinal=0,
                    text=doc.text,
                )
            ]
        documents = [c.text for c in chunks]
        assembled = "".join(documents)
        query = (overrides or {}).get("query") if overrides else None
        max_tokens = ((overrides or {}).get("max_tokens") if overrides else None) or self._max_tokens
        if self._reranker is not None and query is not None and max_tokens is not None:
            ranked = self._reranker(query, documents, max_tokens=max_tokens)
            selected = {r.original_index for r in ranked}
            assembled = "".join(documents[i] for i in range(len(documents)) if i in selected)
        elif self._token_counter is not None and max_tokens is not None:
            if self._token_counter(assembled) > max_tokens:
                kept: list[str] = []
                used = 0
                for doc in documents:
                    n = self._token_counter(doc)
                    remaining = int(max_tokens) - used
                    if remaining <= 0:
                        break
                    if n <= remaining:
                        kept.append(doc)
                        used += n
                        continue
                    prefix = prefix_to_token_budget(doc, remaining, self._token_counter)
                    if prefix:
                        kept.append(prefix)
                    break
                assembled = "".join(kept)
        self.store.assert_immutable(raw_id, checksum_before or None)
        if isinstance(self.capability_log, dict):
            self.capability_log["read_success"] = int(self.capability_log.get("read_success") or 0) + 1
        if self._token_counter is not None:
            return (f"# Document ({self._token_counter(assembled)} tokens)\n{assembled}", None)
        return (assembled, None)
