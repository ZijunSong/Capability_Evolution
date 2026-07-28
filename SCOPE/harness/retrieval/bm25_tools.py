"""Harness tools backed by BrowseComp+ local BM25 index."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from harness.rerank import Reranker
from harness.retrieval.bm25_backend import BrowseCompBm25Backend
from harness.tools import (
    DOC_TRUNCATION,
    GREP_CORPUS_SCHEMA,
    GrepCorpusToolCallMetadata,
    READ_DOCUMENT_SCHEMA,
    SEARCH_CORPUS_SCHEMA,
    SearchCorpusToolCallMetadata,
    Tool,
    ToolCallMetadata,
)

DEFAULT_SNIPPET_MAX_CHARS = 2048


def _format_doc_blocks(
    doc_ids: list[str],
    documents: list[str],
    *,
    token_counter: Callable[[str], int] | None,
    snippet_max_chars: int | None,
) -> tuple[str, list[str]]:
    formatted: list[str] = []
    ids: list[str] = []
    for doc_id, doc in zip(doc_ids, documents):
        if snippet_max_chars is not None:
            doc = doc[:snippet_max_chars]
        tokens = token_counter(doc) if token_counter is not None else None
        formatted.append(
            "\n# DOCUMENT ID: {}{} \n{}".format(
                doc_id,
                f" ({tokens} tokens)" if tokens is not None else "",
                doc[:DOC_TRUNCATION],
            )
        )
        ids.append(doc_id)
    body = "\n".join(formatted) if formatted else "No results found"
    return body, ids


class Bm25SearchCorpusTool(Tool):
    """BM25 search_corpus drop-in for Harness envs (no Chroma / embeddings)."""

    def __init__(
        self,
        backend: BrowseCompBm25Backend,
        *,
        reranker: Reranker | None = None,
        token_counter: Callable[[str], int] | None = None,
        snippet_max_chars: int | None = DEFAULT_SNIPPET_MAX_CHARS,
        search_limit: int = 50,
        display_limit: int = 10,
    ) -> None:
        super().__init__(tool_schema=SEARCH_CORPUS_SCHEMA)
        self._backend = backend
        self._reranker = reranker
        self._token_counter = token_counter
        self._snippet_max_chars = snippet_max_chars
        self._search_limit = search_limit
        self._display_limit = display_limit

    def __call__(
        self,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        if not isinstance(params, dict) or "query" not in params:
            raise ValueError(f"Invalid params type: {type(params)}")

        query = str(params["query"])
        ignore_ids: set[str] = set()
        if overrides and "ignore_ids" in overrides:
            ignore_ids = {str(x) for x in overrides["ignore_ids"]}

        hits = self._backend.search(
            query,
            k=self._search_limit,
            ignore_ids=ignore_ids,
        )
        documents = [h.text for h in hits]
        ids = [h.doc_id for h in hits]

        max_tokens_override = (
            overrides.get("max_tokens") if overrides and "max_tokens" in overrides else None
        )
        if self._reranker is not None and documents:
            rerank_results = self._reranker(
                query, cast(List[str], documents), max_tokens=max_tokens_override
            )
            ids = [ids[r.original_index] for r in rerank_results]
            documents = [r.document for r in rerank_results]

        ids = ids[: self._display_limit]
        documents = documents[: self._display_limit]
        body, returned_ids = _format_doc_blocks(
            ids,
            documents,
            token_counter=self._token_counter,
            snippet_max_chars=self._snippet_max_chars,
        )
        return body, SearchCorpusToolCallMetadata(returned_chunk_ids=returned_ids)


class Bm25GrepCorpusTool(Tool):
    """Regex grep via BM25 prefetch + filter."""

    def __init__(
        self,
        backend: BrowseCompBm25Backend,
        *,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        super().__init__(tool_schema=GREP_CORPUS_SCHEMA)
        self._backend = backend
        self._token_counter = token_counter

    def __call__(
        self,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        if not isinstance(params, dict) or "pattern" not in params:
            raise ValueError(f"Invalid params type: {type(params)}")
        pattern = str(params["pattern"])
        hits = self._backend.grep(pattern, k=5)
        body, ids = _format_doc_blocks(
            [h.doc_id for h in hits],
            [h.text for h in hits],
            token_counter=self._token_counter,
            snippet_max_chars=DEFAULT_SNIPPET_MAX_CHARS,
        )
        return body, GrepCorpusToolCallMetadata(returned_chunk_ids=ids)


class Bm25ReadDocumentTool(Tool):
    """Read full document text from BM25 index by doc id."""

    def __init__(
        self,
        backend: BrowseCompBm25Backend,
        *,
        reranker: Reranker | None = None,
        token_counter: Callable[[str], int] | None = None,
        max_tokens: int | None = 4096,
    ) -> None:
        if max_tokens is not None and token_counter is None:
            raise ValueError("token_counter is required when max_tokens is specified")
        super().__init__(tool_schema=READ_DOCUMENT_SCHEMA)
        self._backend = backend
        self._reranker = reranker
        self._token_counter = token_counter
        self._max_tokens = max_tokens

    def __call__(
        self,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        if not isinstance(params, dict) or ("doc_id" not in params and "id" not in params):
            raise ValueError(f"Invalid params type: {type(params)}")
        doc_id = str(params.get("doc_id") or params.get("id"))
        text = self._backend.get_document(doc_id)
        if not text:
            return f"Document {doc_id} not found.", None

        query = overrides.get("query") if overrides else None
        max_tokens = (
            overrides.get("max_tokens")
            if overrides and "max_tokens" in overrides
            else None
        ) or self._max_tokens

        if (
            self._reranker is not None
            and query is not None
            and max_tokens is not None
            and self._token_counter is not None
        ):
            rerank_results = self._reranker(
                query, [text], max_tokens=max_tokens
            )
            if rerank_results:
                text = rerank_results[0].document
        elif (
            max_tokens is not None
            and self._token_counter is not None
            and self._token_counter(text) > max_tokens
        ):
            # Simple char-level truncation fallback.
            text = text[: max_tokens * 4]

        return text[:DOC_TRUNCATION], None
